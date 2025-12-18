import copy
from typing import Dict, List, Tuple, Union, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import numpy as np
import os
import time
import json
import random
import re
from pathlib import Path
from tqdm import tqdm
from itertools import islice

from util import nethook
from .unke_AnchorEdit_hparams import unkeAnchorEditHyperParams
from .compute_z import compute_z

# ------------------------------
# 1. 统一路径构建函数
# ------------------------------

def construct_context(
    tok: AutoTokenizer,
    question: str,
    base: str,
    window_idx: int,
    window_size: int,
    overlap: int,
) -> List[int]:
    """
    【修正版】支持 BOS (add_special_tokens=True) 的全局切片逻辑。
    解决首个锚点失效和输出乱码问题。
    """
    # 1. 构建全局文本
    full_text = question + base
    
    # 2. 全局分词：【改动】开启 add_special_tokens=True
    # 结果: [BOS, Q_tokens..., A_tokens...]
    full_ids = tok(full_text, add_special_tokens=True)["input_ids"]
    
    # 3. 确定 Answer 起始位置：【改动】开启 add_special_tokens=True
    # 结果: [BOS, Q_tokens...]
    # len(q_ids) 正好指向 Answer 的第一个 Token 在 full_ids 中的索引
    q_ids = tok(question, add_special_tokens=True)["input_ids"]
    answer_start_idx = len(q_ids)
    
    # --- 增加一个安全检查 (可选，但推荐) ---
    # 确保没有发生 token merge 导致的边界错位
    # 如果 full_ids 的长度小于 q_ids，说明发生了极其诡异的截断或合并
    if len(full_ids) < len(q_ids):
        # 回退策略：如果合并导致变短，强制截断
        return full_ids
    # ------------------------------------

    # 4. 计算当前窗口的截断位置 (保持不变)
    step = max(1, window_size - overlap)
    
    # prefix_tokens_in_answer: 在 Answer 部分我们需要保留多少个 Token
    prefix_tokens_in_answer = 0 if window_idx <= 0 else window_idx * step
    
    # 5. 计算绝对截断索引 (保持不变)
    # cutoff_idx 指向 Context 的最后一个 Token 的下一个位置
    cutoff_idx = answer_start_idx + prefix_tokens_in_answer
    
    # 6. 边界保护 (保持不变)
    if cutoff_idx > len(full_ids):
        cutoff_idx = len(full_ids)
        
    # 7. 切片返回
    context_ids = full_ids[:cutoff_idx]
    
    return context_ids





def pad_to_batch_tensor(
    tok: AutoTokenizer,
    batch_ids: List[List[int]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        try: tok.pad_token = tok.eos_token
        except: pass
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    
    if not batch_ids: return {}
    
    bs = len(batch_ids)
    max_len = max(len(x) for x in batch_ids)
    
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
    
    for i, ids in enumerate(batch_ids):
        L = len(ids)
        if L > 0:
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, :L] = 1
            
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# ------------------------------
# 2. Flatten Data 准备 (替代 compute_ks 的显式调用)
# ------------------------------

def get_anchor_input_ids(
    tok: AutoTokenizer,
    raw_question: str,
    raw_unsafe_answer: str,
    anchor_win_idx: int,
    hparams: unkeAnchorEditHyperParams,
    piece_text: str = None  # <--- 新增参数
) -> List[int]:
    """
    【修正版】构建 Context + Safe Sequence[:-1] 以匹配 compute_z 的 Teacher Forcing 逻辑。
    """
    # 1. 获取 Context IDs (Question + Unsafe Prefix)
    context_ids = construct_context(
        tok, raw_question, raw_unsafe_answer, anchor_win_idx,
        hparams.window_size, hparams.overlap
    )
    
    # 如果没有 piece_text，只能返回 context (通常不应发生，除非是无编辑窗口)
    if not piece_text:
        return context_ids

    # 2. 还原 Context 文本
    context_text = tok.decode(context_ids, skip_special_tokens=False)
    
    # 3. 拼接 Context + Safe Piece (解决 Tokenizer 边界)
    full_safe_text = context_text + piece_text
    full_safe_ids = tok(full_safe_text, add_special_tokens=False)["input_ids"]
    
    # 4. 提取 Safe IDs
    # full_safe_ids 包含了 Context + Safe
    # compute_z 中的 input_ids_seq 是 context + safe[:-1]
    # 所以我们直接取 full_safe_ids[:-1] 即可
    
    if len(full_safe_ids) > len(context_ids):
        return full_safe_ids[:-1]
    else:
        return context_ids



def prepare_flattened_anchor_data(
    tok: AutoTokenizer,
    batch_data: list,
    idxs_dict: dict,
    zs_dict: dict,
    inputs_dict: dict, # <--- 新增参数
    hparams: unkeAnchorEditHyperParams
):
    flat_input_ids = []
    flat_edit_indices = []
    flat_target_zs = []
    
    for k_idx, ex in enumerate(batch_data):
        if k_idx not in idxs_dict or not idxs_dict[k_idx]:
            continue
            
        sample_idxs = idxs_dict[k_idx]
        sample_zs = zs_dict[k_idx]
        sample_inputs = inputs_dict[k_idx] # 获取 inputs 列表
        
        # 简单的 1-to-1 映射
        # compute_z 保证了 inputs, idxs, zs 三个列表长度一致且一一对应
        for i in range(len(sample_idxs)):
            
            # sample_inputs[i] 是 tensor (cpu)，需要转成 list[int] 给 pad_to_batch 使用
            inp_tensor = sample_inputs[i]
            inp_list = inp_tensor.tolist()
            
            flat_input_ids.append(inp_list)
            flat_edit_indices.append(sample_idxs[i])
            flat_target_zs.append(sample_zs[i])
            
    return flat_input_ids, flat_edit_indices, flat_target_zs




# ------------------------------
# 3. 评估函数 (修正版)
# ------------------------------

def eval_anchor_windows_avg_prob(
    model: AutoModelForCausalLM,
    inputs_dict: Dict[int, List[torch.Tensor]],
    val_targets_dict: Dict[int, List[int]], # 接收 Target ID
) -> Tuple[float, List[float]]:
    """
    【终极极简版】直接评估：给定 Input，模型预测 Target ID 的概率是多少？
    无拼接，无分词，纯 Tensor 操作。
    """
    device = next(model.parameters()).device
    model.eval()
    
    per_example_probs = []
    
    with torch.no_grad():
        for k in sorted(inputs_dict.keys()):
            inputs_list = inputs_dict[k]
            targets_list = val_targets_dict.get(k, [])
            
            if not inputs_list or not targets_list:
                per_example_probs.append(float("nan"))
                continue
            
            sample_probs = []
            
            for i, inp_cpu in enumerate(inputs_list):
                target_id = targets_list[i] # int
                
                # 1. 准备 Input
                inp_gpu = inp_cpu.unsqueeze(0).to(device) # [1, Seq_Len]
                
                # 2. 前向传播
                logits = model(inp_gpu).logits[0, -1, :] # 取最后一个位置的 Logits
                
                # 3. 计算 Target 的概率
                prob = torch.softmax(logits, dim=-1)[target_id].item()
                
                sample_probs.append(prob)
            
            if sample_probs:
                # 这里使用的是 Mean，但对于编辑效果，这就是关键点的平均成功率
                per_example_probs.append(np.mean(sample_probs))
            else:
                per_example_probs.append(float("nan"))
                
    final_avg = float(np.nanmean(per_example_probs))
    return final_avg, per_example_probs





def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
    param_optimizer = list(model.named_parameters())
    no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
    ]
    return optimizer_parameters

# ------------------------------
# 4. 核心优化函数 (Apply)
# ------------------------------

# 1. 添加 unpack_output 辅助函数
def unpack_output(output):
    return output[0] if isinstance(output, tuple) else output

# 2. 修正 apply_unke_AnchorEdit_to_model
def apply_unke_AnchorEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: unkeAnchorEditHyperParams,
    batch_data: list,
    ex_data: list
):
    # pad_token check
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        try: tok.pad_token = tok.eos_token
        except: pass
    
    device = next(model.parameters()).device
    
    # 1. Freeze
    preserve_params = []
    for name, params in model.named_parameters():
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    weights = {param: nethook.get_parameter(model, param) for param in preserve_params}
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    
    z_layer = hparams.layers[-1]
    zs_dict = {} 
    idxs_dict = {} 
    inputs_dict = {} 
    val_targets_dict = {}
    
    # 2. Compute Z (Targets)
    for k, data in enumerate(batch_data):
        original_ans = data.get("answer", "")
        if "pre_edit" in data and "response" in data["pre_edit"]:
            data["answer"] = data["pre_edit"]["response"]
            
        inputs_list, idxs_list, zs_list, val_targets = compute_z(model, tok, data, z_layer, hparams)
        
        data["answer"] = original_ans
        idxs_dict[k] = idxs_list
        zs_dict[k] = zs_list 
        inputs_dict[k] = inputs_list
        val_targets_dict[k] = val_targets
    
    if not idxs_dict:
        print("[AnchorEdit] No anchors found, skip.")
        return weights_copy
    
    # 3. Eval Before
    try:
        base_avg, base_per_ex = eval_anchor_windows_avg_prob(model, inputs_dict, val_targets_dict)
        print(f"[Debug-Eval] BEFORE edit | avg_prob={base_avg:.4f}")
    except Exception as e:
        print(f"[Debug-Eval] BEFORE edit failed: {e}")
        base_avg = 0.0

    # 4. Flatten Inputs
    flat_input_ids, flat_edit_indices, flat_target_zs = prepare_flattened_anchor_data(
        tok, batch_data, idxs_dict, zs_dict, inputs_dict, hparams
    )
    
    if not flat_input_ids: return weights_copy
    
    # Pad Batch
    contexts_tok = pad_to_batch_tensor(tok, flat_input_ids, device)
    all_target_zs = torch.stack(flat_target_zs).to(device)
    
    # 5. Preserve Data
    if isinstance(ex_data, str): ex_data = [ex_data]
    ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(device)
    
    # 6. Loop
    for i, layer in enumerate(hparams.layers):
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        # --- A. Get Current Keys & Initial V (Edit Data) ---
        captured_in_edit = []
        captured_out_edit = []
        def hook_fn_edit(m, inp, out):
            captured_in_edit.append(inp[0].detach())
            captured_out_edit.append(unpack_output(out).detach()) # 使用 unpack
            
        h1 = _layer.register_forward_hook(hook_fn_edit)
        with torch.no_grad():
            _ = model(**contexts_tok)
        h1.remove()
        
        layer_in_edit = captured_in_edit[0]
        layer_out_edit_initial = captured_out_edit[0]
        
        # Extract Vectors
        current_ks_list = []
        init_out_vectors_list = []
        
        for row, idx in enumerate(flat_edit_indices):
            # 维度保护
            actual_idx = idx if idx < layer_in_edit.shape[1] else layer_in_edit.shape[1] - 1
            current_ks_list.append(layer_in_edit[row, actual_idx, :])
            init_out_vectors_list.append(layer_out_edit_initial[row, actual_idx, :])
            
        current_ks = torch.stack(current_ks_list, dim=0)
        init_out_vectors = torch.stack(init_out_vectors_list, dim=0)
        
        # Compute Target
        # 【修正】如果是单层编辑，直接用 Z；多层则分摊
        if len(hparams.layers) == 1:
            targets = all_target_zs.detach() # 直接对齐 Z
        else:
            residue = (all_target_zs - current_ks) / (len(hparams.layers) - i)
            targets = (init_out_vectors + residue).detach()
            
        # --- B. Get Preserve Stats ---
        captured_in_stat = []
        captured_out_stat = []
        def hook_fn_stat(m, inp, out):
            captured_in_stat.append(inp[0].detach())
            captured_out_stat.append(unpack_output(out).detach())
            
        h2 = _layer.register_forward_hook(hook_fn_stat)
        with torch.no_grad():
            _ = model(**ex_tok)
        h2.remove()
        
        stat_in = captured_in_stat[0]
        stat_out = captured_out_stat[0]
        
        # --- C. Optimize ---
        criterion_mse = nn.MSELoss() 
        for n, m in _layer.named_parameters(): m.requires_grad = True
        params = get_optimizer_params(_layer, hparams.lr)
        optimizer = optim.AdamW(params, lr=hparams.lr, eps=1e-8)
        
        # Mask Helpers
        if 'llama' in hparams.model_name.lower():
            input_causal_mask, input_position_ids, _ = get_causal_mask(layer_in_edit, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids, _ = get_causal_mask(stat_in, ex_tok['attention_mask'])
        elif 'qwen' in hparams.model_name.lower():
            input_causal_mask, input_position_ids = get_qwen2_causal_mask(layer_in_edit, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids = get_qwen2_causal_mask(stat_in, ex_tok['attention_mask'])
        
        ex_pos_emb, in_pos_emb = None, None
        if 'Qwen3' in hparams.model_name or 'Qwen2' in hparams.model_name:
            if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                rotary_emb = model.model.rotary_emb
            else:
                rotary_emb = model.transformer.rotary_emb
            ex_pos_emb = rotary_emb(stat_in, position_ids=ex_position_ids)
            in_pos_emb = rotary_emb(layer_in_edit, position_ids=input_position_ids)
            
        # 手动计算 Llama RoPE (Fix for NoneType error)
        if 'llama' in hparams.model_name.lower():
             if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                rotary_emb = model.model.rotary_emb
             else:
                rotary_emb = model.model.layers[0].self_attn.rotary_emb
             
             # Calculate Embeddings
             ex_pos_emb = rotary_emb(stat_in, position_ids=ex_position_ids)
             in_pos_emb = rotary_emb(layer_in_edit, position_ids=input_position_ids)

        # --- D. Loop ---
        for step in range(hparams.optim_num_step):
            optimizer.zero_grad()
            
            # Preserve Forward
            if 'qwen' in hparams.model_name.lower():
                raw_stat = _layer(stat_in, attention_mask=ex_causal_mask, position_embeddings=ex_pos_emb)
            elif 'llama' in hparams.model_name.lower():
                # Llama 传入 position_embeddings
                raw_stat = _layer(stat_in, attention_mask=ex_causal_mask, position_embeddings=ex_pos_emb)
            else:
                raw_stat = _layer(stat_in)
            
            curr_stat = unpack_output(raw_stat)
            
            # Masked MSE for Preserve
            mask_expanded = ex_tok['attention_mask'].unsqueeze(-1).expand_as(curr_stat).bool()
            # 简单的 MSE 也可以，但带 mask 更准
            loss_preserve = criterion_mse(curr_stat, stat_out)
            
            # Edit Forward
            if 'qwen' in hparams.model_name.lower():
                raw_edit = _layer(layer_in_edit, attention_mask=input_causal_mask, position_embeddings=in_pos_emb)
            elif 'llama' in hparams.model_name.lower():
                raw_edit = _layer(layer_in_edit, attention_mask=input_causal_mask, position_embeddings=in_pos_emb)
            else:
                raw_edit = _layer(layer_in_edit)
                
            curr_edit = unpack_output(raw_edit)
            
            # Extract Vectors
            curr_vectors_list = []
            for row, idx in enumerate(flat_edit_indices):
                actual_idx = idx if idx < curr_edit.shape[1] else curr_edit.shape[1] - 1
                curr_vectors_list.append(curr_edit[row, actual_idx, :])
            curr_vectors = torch.stack(curr_vectors_list, dim=0)
            
            loss_edit = criterion_mse(curr_vectors, targets)
            
            # Reweighting
            alpha = 10 # Preserve 权重
            loss = loss_edit + alpha * loss_preserve
            
            loss.backward()
            
            # Gradient Clip
            torch.nn.utils.clip_grad_norm_(_layer.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if (step + 1) % 5 == 0 or step == 0:
                 print(f"L{layer} Step {step+1} | Total: {loss.item():.5f} (Edit: {loss_edit.item():.5f}, Prsv: {loss_preserve.item():.5f})")
            
            if loss_edit.item() < 2e-3:
                print("Early stopping due to low edit loss.")
                break
        
        torch.cuda.empty_cache()
        
    try:
        final_avg, final_per_ex = eval_anchor_windows_avg_prob(model, inputs_dict, val_targets_dict)
        print(f"[Debug-Eval] AFTER ALL | avg_prob={final_avg:.4f}")
        if 'base_avg' in locals():
            print(f"[Debug-Eval] AFTER ALL | Delta avg_prob={final_avg - base_avg:.4f}")
    except Exception as e:
        print(f"[Debug-Eval] AFTER ALL failed: {e}")
        
    return weights_copy


# ... (Mask Helpers 保持不变) ...
def get_qwen2_causal_mask(input_tensor, attention_mask, past_key_values_length=0):
    device = input_tensor.device
    batch_size, seq_length = input_tensor.shape[0], input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    causal_mask = torch.full(
        (seq_length, seq_length),
        fill_value=torch.finfo(input_tensor.dtype).min,
        dtype=input_tensor.dtype,
        device=device,
    )
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
    expanded_causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)
    if attention_mask is not None:
        padding_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
        final_mask = expanded_causal_mask.masked_fill(padding_mask == 0, torch.finfo(input_tensor.dtype).min)
    else:
        final_mask = expanded_causal_mask
    return final_mask, position_ids

def get_causal_mask(input_tensor,attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length
    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.clone() 
    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask,position_ids,cache_position
