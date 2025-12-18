import copy
from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from util import nethook
from util.globals import *
import numpy as np
import os
from .memit_AnchorEdit_hparams import MEMITAnchorEditHyperParams

COV_CACHE = {}

# ------------------------------
# 1. 辅助函数
# ------------------------------

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

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape: return matrix
    elif matrix.T.shape == shape: return matrix.T
    else: raise ValueError("Update matrix shape mismatch.")

def unpack_output(output):
    return output[0] if isinstance(output, tuple) else output

def get_qwen2_causal_mask(input_tensor, attention_mask, past_key_values_length=0):
    device = input_tensor.device
    batch_size, seq_length = input_tensor.shape[0], input_tensor.shape[1]
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    causal_mask = torch.full((seq_length, seq_length), torch.finfo(input_tensor.dtype).min, dtype=input_tensor.dtype, device=device)
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
    expanded_causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)
    if attention_mask is not None:
        padding_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
        final_mask = expanded_causal_mask.masked_fill(padding_mask == 0, torch.finfo(input_tensor.dtype).min)
    else:
        final_mask = expanded_causal_mask
    return final_mask, position_ids

def get_causal_mask(input_tensor, attention_mask):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = sequence_length
    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
    if sequence_length != 1: causal_mask = torch.triu(causal_mask, diagonal=1)
    cache_position = torch.arange(0, 0 + input_tensor.shape[1], device=device)
    position_ids = cache_position.unsqueeze(0)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1).clone()
    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask, position_ids, cache_position

# ------------------------------
# 2. Covariance (MEMIT Core)
# ------------------------------

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    from util.layer_stats import layer_stats
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model, tok, layer_name, STATS_DIR, mom2_dataset, to_collect=["mom2"],
            sample_size=mom2_n_samples, precision=mom2_dtype, force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    return torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")

# ------------------------------
# 3. Eval Function (New)
# ------------------------------

def eval_anchor_windows_avg_prob(
    model: AutoModelForCausalLM,
    inputs_dict: Dict[int, List[torch.Tensor]],
    val_targets_dict: Dict[int, List[int]], 
) -> Tuple[float, List[float]]:
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
                target_id = targets_list[i]
                inp_gpu = inp_cpu.unsqueeze(0).to(device)
                logits = model(inp_gpu).logits[0, -1, :] 
                prob = torch.softmax(logits, dim=-1)[target_id].item()
                sample_probs.append(prob)
            if sample_probs:
                per_example_probs.append(np.mean(sample_probs))
            else:
                per_example_probs.append(float("nan"))
    final_avg = float(np.nanmean(per_example_probs))
    return final_avg, per_example_probs

# ------------------------------
# 4. Compute KS (New)
# ------------------------------

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: MEMITAnchorEditHyperParams,
    layer: int,
    idxs_dict: dict,
    inputs_dict: dict, 
):
    """
    计算 Key (k) = Layer Input
    使用 inputs_dict 中的 Tensor，保证与 compute_z 的输入绝对一致。
    """
    device = next(model.parameters()).device
    
    flat_inputs = []
    flat_indices = []
    
    for k in sorted(idxs_dict.keys()):
        idxs_list = idxs_dict[k]
        inputs_list = inputs_dict[k]
        for i, idx in enumerate(idxs_list):
            if i >= len(inputs_list): break
            inp_tensor = inputs_list[i]
            inp_list = inp_tensor.tolist()
            flat_inputs.append(inp_list)
            flat_indices.append(len(inp_list) - 1)
            
    if not flat_inputs:
        raise ValueError("No inputs found in compute_ks")
    model_inputs = pad_to_batch_tensor(tok, flat_inputs, device)
    expected_bs = len(flat_inputs)
    expected_seq = model_inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            _ = model(**model_inputs)
            # layer_in_ks: [Batch, Seq, Hidden]
            layer_in_ks = tr.input 
            
    layer_in_ks = layer_in_ks[0] if isinstance(layer_in_ks, tuple) else layer_in_ks
    
    # 维度还原 (Handle Flattened Batch)
    if layer_in_ks.dim() == 2:
        hidden_dim = layer_in_ks.shape[-1]
        if layer_in_ks.shape[0] == expected_bs * expected_seq:
            layer_in_ks = layer_in_ks.view(expected_bs, expected_seq, hidden_dim)
        elif layer_in_ks.shape[0] == expected_seq and expected_bs == 1:
            layer_in_ks = layer_in_ks.unsqueeze(0)
            
    if layer_in_ks.dim() != 3 or layer_in_ks.shape[0] != expected_bs:
        print(f"Warning: compute_ks input shape mismatch. Expected ({expected_bs}, {expected_seq}, ...), got {layer_in_ks.shape}")
        if expected_bs == 1 and layer_in_ks.dim() == 2:
             layer_in_ks = layer_in_ks.unsqueeze(0)
    # 提取 K
    ks_extracted_list = []
    for i, idx in enumerate(flat_indices):
        ks_extracted_list.append(layer_in_ks[i, idx, :])
        
    zs_out = torch.stack(ks_extracted_list, dim=1) 
    
    return zs_out, flat_indices, model_inputs, layer_in_ks

# ------------------------------
# 5. Apply MEMIT AnchorEdit
# ------------------------------

def apply_memit_AnchorEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: MEMITAnchorEditHyperParams,
    batch_data: list,
):
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        try: tok.pad_token = tok.eos_token
        except: pass
    
    device = next(model.parameters()).device
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    z_layer = hparams.layers[-1]
    
    # 2. Compute Z
    all_zs_list = []
    idxs_dict = {}
    inputs_dict = {}
    val_targets_dict = {}
    
    for k, data in enumerate(batch_data):
        original_ans = data.get("answer", "")
        if "pre_edit" in data and "response" in data["pre_edit"]:
            data["answer"] = data["pre_edit"]["response"]
            
        inputs_list, idxs_list, zs_list, val_targets = compute_z(
            model, tok, data, z_layer, hparams
        )
        
        data["answer"] = original_ans
        idxs_dict[k] = idxs_list
        inputs_dict[k] = inputs_list
        val_targets_dict[k] = val_targets
        all_zs_list.extend(zs_list)
    if not all_zs_list:
        print("[AnchorEdit] No anchors found, skip.")
        return weights_copy
    
    # Eval Before
    try:
        base_avg, _ = eval_anchor_windows_avg_prob(model, inputs_dict, val_targets_dict)
        print(f"[Debug-Eval] BEFORE edit | avg_prob={base_avg:.4f}")
    except Exception as e:
        print(f"[Debug-Eval] BEFORE edit failed: {e}")
        base_avg = 0.0
    # 堆叠 Target Z
    zs = torch.stack(all_zs_list, dim=1).to(device)
    
    # ===== 层级插入更新 =====
    
    for i, layer in enumerate(hparams.layers):
        print(f"\nEditing Layer {layer}...")
        
        # 1. 准备 Batch Input
        flat_inputs = []
        flat_indices = []
        for k in sorted(idxs_dict.keys()):
            idxs_list = idxs_dict[k]
            inputs_list = inputs_dict[k]
            for j, idx in enumerate(idxs_list):
                if j >= len(inputs_list): break
                flat_inputs.append(inputs_list[j].tolist())
                flat_indices.append(len(inputs_list[j]) - 1)
        
        if not flat_inputs: break
        model_inputs = pad_to_batch_tensor(tok, flat_inputs, device)
        
        # 2. Dual Trace: 获取 Layer Input (4096) 和 Module Input (12288)
        layer_name = hparams.layer_module_tmp.format(layer)
        rewrite_name = hparams.rewrite_module_tmp.format(layer)
        
        with torch.no_grad():
            with nethook.TraceDict(
                model, 
                layers=[layer_name, rewrite_name], 
                retain_input=True
            ) as tr:
                _ = model(**model_inputs)
                
                # Layer Input [Batch, Seq, 4096] (用于计算 Residual)
                layer_in_batch = tr[layer_name].input
                layer_in_batch = layer_in_batch[0] if isinstance(layer_in_batch, tuple) else layer_in_batch
                
                # Module Input [Batch, Seq, 12288] (用于 MEMIT 公式)
                module_in_batch = tr[rewrite_name].input
                module_in_batch = module_in_batch[0] if isinstance(module_in_batch, tuple) else module_in_batch
        # 3. 计算 Residual (使用 Layer Input)
        with torch.no_grad():
            _layer = nethook.get_module(model, layer_name)
            attention_mask = model_inputs['attention_mask']
            
            # 手动处理 Mask 和 Pos Embed
            if 'qwen' in hparams.model_name.lower():
                causal_mask, position_ids = get_qwen2_causal_mask(layer_in_batch, attention_mask)
                if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                    rotary_emb = model.model.rotary_emb
                else:
                    rotary_emb = model.transformer.rotary_emb
                
                cos, sin = rotary_emb(layer_in_batch, position_ids=position_ids)
                
                raw_out = _layer(
                    layer_in_batch, 
                    attention_mask=causal_mask, 
                    position_embeddings=(cos, sin)
                )
            elif 'llama' in hparams.model_name.lower():
                causal_mask, position_ids, _ = get_causal_mask(layer_in_batch, attention_mask)
                
                # --- 修复代码开始: 手动计算 Llama RoPE ---
                if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                    rotary_emb = model.model.rotary_emb
                elif hasattr(model, 'rotary_emb'):
                    rotary_emb = model.rotary_emb
                else:
                    rotary_emb = None
                
                position_embeddings = None
                if rotary_emb is not None:
                     cos, sin = rotary_emb(layer_in_batch, position_ids=position_ids)
                     position_embeddings = (cos, sin)
                
                raw_out = _layer(
                    layer_in_batch,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings # 修复点
                )
                # --- 修复代码结束 ---
            else:
                raw_out = _layer(layer_in_batch)
            
            layer_out = unpack_output(raw_out)
            
            # 维度还原
            if layer_out.dim() == 2:
                bs, seq, hidden = layer_in_batch.shape
                if layer_out.shape[0] == bs * seq:
                    layer_out = layer_out.view(bs, seq, hidden)
                elif layer_out.shape[0] == seq and bs == 1:
                    layer_out = layer_out.unsqueeze(0)
            
            curr_out_list = []
            for batch_idx, seq_idx in enumerate(flat_indices):
                curr_out_list.append(layer_out[batch_idx, seq_idx, :])
            curr_out_stacked = torch.stack(curr_out_list, dim=1).to(device)
            
            # Residual: [4096, M]
            targets_resid = (zs - curr_out_stacked) / (len(hparams.layers) - i)
            
        # 4. MEMIT 公式准备 (使用 Module Input 12288)
        
        # 提取 K (12288)
        ks_list = []
        for batch_idx, seq_idx in enumerate(flat_indices):
            ks_list.append(module_in_batch[batch_idx, seq_idx, :])
        layer_ks = torch.stack(ks_list, dim=1).to(device) # [12288, M]
        
        # 获取 Cov (12288)
        force_recompute = False
        cov = get_cov(
            model, tok, rewrite_name,
            hparams.mom2_dataset, hparams.mom2_n_samples
            if not force_recompute else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype, force_recompute=force_recompute,
        )
        
        # 5. MEMIT Update
        # Solve: (Cov_new) X = K -> X = Cov^-1 K
        # Cov_new = Weight * Cov_old + K K^T
        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov + layer_ks @ layer_ks.T, 
            layer_ks
        )
        
        # Delta W = Residue @ adj_K^T
        # [4096, M] @ [M, 12288] = [4096, 12288]
        resid = targets_resid 
        upd_matrix = resid @ adj_k.T 
        
        # 更新权重
        weight_name = f"{rewrite_name}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        
        print(f"Layer {layer} Update: Norm {torch.linalg.norm(upd_matrix).item():.4f}")
        
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            
        del layer_ks, resid, adj_k, upd_matrix, cov
        torch.cuda.empty_cache()
        
        # 6. Eval
        try:
            cur_avg, _ = eval_anchor_windows_avg_prob(model, inputs_dict, val_targets_dict)
            print(f"[Debug-Eval] AFTER layer {layer} | avg_prob={cur_avg:.4f}")
        except Exception as e:
            print(f"Eval Error: {e}")
    try:
        final_avg, _ = eval_anchor_windows_avg_prob(model, inputs_dict, val_targets_dict)
        print(f"[Debug-Eval] AFTER ALL | avg_prob={final_avg:.4f}")
        if 'base_avg' in locals():
            print(f"[Debug-Eval] AFTER ALL | Delta avg_prob={final_avg - base_avg:.4f}")
    except Exception as e:
        print(f"Final Eval Error: {e}")
    return weights_copy
