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
from .AnchorEdit_hparams import AnchorEditHyperParams

COV_CACHE = {}

# ------------------------------
# 1. 辅助函数：Pad & Match Shape & Masks
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
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )

def unpack_output(output):
    return output[0] if isinstance(output, tuple) else output

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

def get_causal_mask(input_tensor, attention_mask):
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
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask, position_ids, cache_position

# ------------------------------
# 2. Covariance Statistics (AlphaEdit Core)
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
    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(f"Layer {layer} nullspace dim: {len(small_singular_indices)}")
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

# ------------------------------
# 3. Eval Function
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
# 4. Compute KS
# ------------------------------

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: AnchorEditHyperParams,
    layer: int,
    idxs_dict: dict,
    inputs_dict: dict, 
):
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
            layer_in_ks = tr.input 
            
    layer_in_ks = layer_in_ks[0] if isinstance(layer_in_ks, tuple) else layer_in_ks
    
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
    ks_extracted_list = []
    for i, idx in enumerate(flat_indices):
        ks_extracted_list.append(layer_in_ks[i, idx, :])
        
    zs_out = torch.stack(ks_extracted_list, dim=1) 
    return zs_out, flat_indices, model_inputs, layer_in_ks

# ------------------------------
# 5. Core Apply Function
# ------------------------------

def apply_AnchorEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: AnchorEditHyperParams,
    batch_data: list,
    P=None,
):
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        try: tok.pad_token = tok.eos_token
        except: pass
    
    device = next(model.parameters()).device
    
    # 1. 准备权重
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
    anchor_weight_list = []
    
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
        
        for _ in range(len(idxs_list)):
            anchor_weight_list.append(1.0) 
    if not all_zs_list:
        print("[AnchorEdit] No anchors found, skip.")
        return weights_copy
    
    try:
        base_avg, _ = eval_anchor_windows_avg_prob(model, inputs_dict, val_targets_dict)
        print(f"[Debug-Eval] BEFORE edit | avg_prob={base_avg:.4f}")
    except Exception as e:
        print(f"[Debug-Eval] BEFORE edit failed: {e}")
        base_avg = 0.0
    zs = torch.stack(all_zs_list, dim=1).to(device)
    anchor_weights = torch.tensor(anchor_weight_list, dtype=torch.float32, device=device)
    anchor_sqrt_w = anchor_weights.clamp(min=1e-8).sqrt().unsqueeze(0) 
    
    # ===== 层级插入更新 =====
    for i, layer in enumerate(hparams.layers):
        print(f"\nEditing Layer {layer}...")
        
        # --- 准备 Batch Input ---
        # (原 compute_ks 的逻辑移到这里，以便同时 trace 两个目标)
        flat_inputs = []
        flat_indices = []
        for k in sorted(idxs_dict.keys()):
            idxs_list = idxs_dict[k]
            inputs_list = inputs_dict[k]
            for j, idx in enumerate(idxs_list):
                if j >= len(inputs_list): break
                inp_list = inputs_list[j].tolist()
                flat_inputs.append(inp_list)
                flat_indices.append(len(inp_list) - 1)
        
        if not flat_inputs: break
        model_inputs = pad_to_batch_tensor(tok, flat_inputs, device)
        
        # --- Dual Trace: 获取 Layer Input (4096) 和 Module Input (12288) ---
        layer_name = hparams.layer_module_tmp.format(layer)
        rewrite_name = hparams.rewrite_module_tmp.format(layer)
        
        with torch.no_grad():
            with nethook.TraceDict(
                model, 
                layers=[layer_name, rewrite_name], 
                retain_input=True
            ) as tr:
                _ = model(**model_inputs)
                
                # 1. Layer Input [Batch, Seq, 4096]
                layer_in_batch = tr[layer_name].input
                layer_in_batch = layer_in_batch[0] if isinstance(layer_in_batch, tuple) else layer_in_batch
                
                # 2. Module Input [Batch, Seq, 12288] -> 这是 Key!
                module_in_batch = tr[rewrite_name].input
                module_in_batch = module_in_batch[0] if isinstance(module_in_batch, tuple) else module_in_batch
        
        # --- 计算 Residual (需要 Layer Output) ---
        # 使用 Layer Input 手动 Forward
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
                
                # Forward
                raw_out = _layer(
                    layer_in_batch, 
                    attention_mask=causal_mask, 
                    position_embeddings=(cos, sin)
                )
            elif 'llama' in hparams.model_name.lower():
                causal_mask, position_ids, _ = get_causal_mask(layer_in_batch, attention_mask)
                
                # --- 修复代码开始: 手动计算 Llama 的 RoPE ---
                if hasattr(model, 'model') and hasattr(model.model, 'rotary_emb'):
                    rotary_emb = model.model.rotary_emb
                elif hasattr(model, 'rotary_emb'):
                    # Fallback for some variants
                    rotary_emb = model.rotary_emb
                else:
                    rotary_emb = None
                
                position_embeddings = None
                if rotary_emb is not None:
                     # 计算 cos, sin
                     cos, sin = rotary_emb(layer_in_batch, position_ids=position_ids)
                     position_embeddings = (cos, sin)
                
                # 传入 position_embeddings 以兼容新版 transformers
                raw_out = _layer(
                    layer_in_batch,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )
                # --- 修复代码结束 ---

            else:
                raw_out = _layer(layer_in_batch)
            
            layer_out = unpack_output(raw_out)
            
            # 维度还原 [Batch, Seq, 4096]
            if layer_out.dim() == 2:
                bs, seq, hidden = layer_in_batch.shape
                if layer_out.shape[0] == bs * seq:
                    layer_out = layer_out.view(bs, seq, hidden)
                elif layer_out.shape[0] == seq and bs == 1:
                    layer_out = layer_out.unsqueeze(0)
            
            # 提取 Output Vectors
            curr_out_list = []
            for batch_idx, seq_idx in enumerate(flat_indices):
                curr_out_list.append(layer_out[batch_idx, seq_idx, :])
            curr_out_stacked = torch.stack(curr_out_list, dim=1).to(device) # [4096, M]
            
            # 计算 Residual (目标 Z - 当前输出)
            # 注意：Z 是 4096 维的
            targets_resid = (zs - curr_out_stacked) / (len(hparams.layers) - i)
            
        # --- 准备 AlphaEdit 变量 (使用 Module Input 12288) ---
        
        # 1. 提取 Key (Cur KS) from Module Input
        ks_list = []
        for batch_idx, seq_idx in enumerate(flat_indices):
            ks_list.append(module_in_batch[batch_idx, seq_idx, :])
        layer_ks = torch.stack(ks_list, dim=1).to(device) # [12288, M]
        
        # 2. 提取 Context Keys (KP) from Module Input (12288)
        kp_list = []
        for batch_idx in range(module_in_batch.size(0)):
            seq_len = module_in_batch.size(1)
            kp_list.append(module_in_batch[batch_idx, 0, :])
            if seq_len > 5:
                kp_list.append(module_in_batch[batch_idx, seq_len // 2, :])
        layer_kp = torch.stack(kp_list, dim=1).to(device) # [12288, N_bg]
        
        resid = targets_resid # [4096, M]
        
        # 4. AlphaEdit 核心公式
        # Apply weights:
        layer_ks_w = layer_ks * anchor_sqrt_w
        resid_w    = resid * anchor_sqrt_w
        
        P_layer = P[i, :, :].to(device)
        
        # Covariance of Input (12k)
        Cov = layer_ks_w @ layer_ks_w.T + layer_kp @ layer_kp.T
        
        # LHS (12k x 12k)
        LHS = P_layer @ Cov + hparams.L2 * torch.eye(layer_ks.shape[0], dtype=torch.float, device=device)
        
        # RHS (12k x 4k)
        # resid_w is [4096, M]. layer_ks_w is [12288, M].
        # We need K R^T.
        RHS = P_layer @ layer_ks_w @ resid_w.T 
        
        # Solve
        upd_matrix_T = torch.linalg.solve(LHS, RHS)
        upd_matrix = upd_matrix_T.T # [4096, 12288]
        
        # 5. 更新参数
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        
        # 再次检查维度匹配 (Weight 是 4096 x 12288)
        if upd_matrix.shape != weights[weight_name].shape:
             # 如果形状不对，尝试转置 (虽然逻辑上已经是 4k x 12k 了)
             upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        
        print(f"Layer {layer} Update: Norm {torch.linalg.norm(upd_matrix).item():.4f}")
        
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            
        del layer_ks, layer_kp, resid, LHS, RHS, upd_matrix, P_layer, Cov
        torch.cuda.empty_cache()
        
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
