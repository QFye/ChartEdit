from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .AnchorEdit_hparams import AnchorEditHyperParams
from util import nethook

@torch.no_grad()
def _tokenize_anchors(tok: AutoTokenizer, anchors_raw: Dict[int, str], device: torch.device) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    if not anchors_raw: return out
    if tok.pad_token_id is None and tok.eos_token is not None:
        try: tok.pad_token = tok.eos_token
        except: pass
    for k, v in anchors_raw.items():
        enc = tok(v, return_tensors="pt")
        out[k] = enc["input_ids"][0].to(device)
    return out

def construct_context(
    tok: AutoTokenizer,
    question: str,
    base: str,
    window_idx: int,
    window_size: int,
    overlap: int,
) -> List[int]:
    """
    【修正版】与 compute_z 保持严格一致的 "Text Concatenation" 策略。
    """
    # 1. 计算 base 截断位置 (字符/Token 估算)
    base_ids_full = tok(base, add_special_tokens=False)["input_ids"]
    
    step = max(1, window_size - overlap)
    prefix_token_count = 0 if window_idx <= 0 else window_idx * step
    prefix_token_count = min(prefix_token_count, len(base_ids_full))
    
    # 2. 还原文本
    current_base_ids = base_ids_full[:prefix_token_count]
    current_base_text = tok.decode(current_base_ids, skip_special_tokens=False)
    
    # 3. 文本拼接 (关键步骤)
    full_text = question + current_base_text
    
    # 4. 整体分词
    context_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    
    return context_ids


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: AnchorEditHyperParams,
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
    
    # === 0. 模型参数获取 ===
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # === 1. 准备基础文本 (Global Slicing) ===
    raw_question = data.get("adversarial prompt") or data.get("adversarial_prompt") or data.get("question", "")
    raw_unsafe_answer = data.get("answer", "") 
    
    full_unsafe_text_global = raw_question + raw_unsafe_answer
    full_unsafe_ids_global = tok(full_unsafe_text_global, add_special_tokens=False)["input_ids"]
    full_unsafe_tensor_global = torch.LongTensor(full_unsafe_ids_global).to("cuda")
    
    q_ids = tok(raw_question, add_special_tokens=False)["input_ids"]
    answer_start_idx = len(q_ids)
    
    current_token_idx = 0 
    win_idx = 0
    answer_len = len(full_unsafe_ids_global) - answer_start_idx
    step = max(1, hparams.window_size - hparams.overlap)
    
    all_delta = []
    all_target = []
    all_idxs = []
    all_inputs = [] 
    all_validation_targets = []
    
    piece_dict = data.get("piece", {}) or {}
    anchor_win_set = {int(k) for k in piece_dict.keys()}
    
    while current_token_idx < answer_len:
        if win_idx not in anchor_win_set:
            current_token_idx += step
            win_idx += 1
            continue
            
        # === 2. 构建 Input ===
        context_end_pos = answer_start_idx + current_token_idx
        input_ids_context = full_unsafe_tensor_global[:context_end_pos].unsqueeze(0)
        context_ids = input_ids_context[0]
        base_lookup_idx = len(context_ids) - 1
        
        full_context_text = tok.decode(context_ids, skip_special_tokens=False)
        
        # === 3. 构建 Target & Auto-Align ===
        base_win_end_pos = min(len(full_unsafe_ids_global), context_end_pos + hparams.window_size)
        b_list = full_unsafe_ids_global[context_end_pos : base_win_end_pos + hparams.window_size]
        base_win_text = tok.decode(b_list, skip_special_tokens=False)
        
        mismatch_offset_tokens = 0
        p_list = []
        piece_ids_full = None
        piece_text = piece_dict.get(str(win_idx)) or piece_dict.get(win_idx)
        
        if isinstance(piece_text, str) and len(piece_text) > 0:
            print("cut")
            full_safe_text = raw_question + piece_text
            full_safe_ids = tok(full_safe_text, add_special_tokens=False)["input_ids"]
            
            if len(full_safe_ids) > len(context_ids):
                piece_ids_full = full_safe_ids[len(context_ids):]
            else:
                piece_ids_full = []
            
            best_shift = 0
            max_match_len = 0
            limit = min(len(b_list), len(piece_ids_full))
            for k in range(limit):
                if b_list[k] == piece_ids_full[k]: max_match_len += 1
                else: break
            
            print(f"  [Auto-Align] Best shift found: {best_shift} (Match len: {max_match_len})")
            p_list = piece_ids_full
            len_cut = min(len(p_list), len(b_list))
            p_list = p_list[:len_cut]
            b_list = b_list[:len_cut]
            
            # Padding Fix
            if len(b_list) > 1 and len(p_list) > 0:
                if b_list[0] != p_list[0] and b_list[1] == p_list[0]:
                    print(f"  [Padding Fix] p_list starts late. Prepending b_list[0] ({b_list[0]}) to align.")
                    p_list.insert(0, b_list[0])
            
            piece_text_debug = tok.decode(p_list, skip_special_tokens=False)
            print(f"piece_text (aligned slice) = {repr(piece_text_debug)}")
            print(f"base_win_text = {repr(base_win_text)}")
            print(f"p_list = {p_list}")
            print(f"b_list = {b_list}")
            
            def check_match(list1, list2):
                limit = min(len(list1), len(list2))
                for k in range(limit):
                    if list1[k] != list2[k]: return k
                return limit

            idx_0 = check_match(b_list, p_list)
            mismatch_offset_tokens = idx_0
            if mismatch_offset_tokens >= len(p_list):
                mismatch_offset_tokens = max(0, len(p_list) - 1)

        print(mismatch_offset_tokens)
        
        # === 4. Target & Edit Point ===
        lookup_idx_val = base_lookup_idx + mismatch_offset_tokens
        safe_tensor = torch.LongTensor(p_list).unsqueeze(0).to("cuda")
        
        if len(p_list) > 0:
            if safe_tensor.size(1) > 1:
                input_ids_seq = torch.cat([input_ids_context, safe_tensor[:, :-1]], dim=1)
            else:
                input_ids_seq = input_ids_context
        else:
            input_ids_seq = input_ids_context

        ex_len = input_ids_seq.size(1)
        lookup_idx_val = min(lookup_idx_val, ex_len - 1)
        lookup_idxs = [lookup_idx_val]
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(1, ex_len)
        target_token_id = None
        if len(p_list) > 0:
            start_fill = lookup_idx_val
            tokens_to_fill = safe_tensor[0][mismatch_offset_tokens:]
            if len(tokens_to_fill) > 0:
                fill_len = min(len(tokens_to_fill), ex_len - start_fill)
                rewriting_targets[0, start_fill : start_fill + fill_len] = tokens_to_fill[:fill_len]
                target_token_id = tokens_to_fill[0]
            else:
                target_token_id = tok.eos_token_id
                rewriting_targets[0, start_fill] = target_token_id
        else:
            target_token_id = tok.eos_token_id
            rewriting_targets[0, lookup_idx_val] = target_token_id
        all_validation_targets.append(target_token_id)
        # --- Debug Info ---
        print(f"\n>>> [Debug] Anchor Window: {win_idx}")
        print(f"    [FULL RAW INPUT]     : {repr(full_context_text)}")
        
        safe_debug_text = tok.decode(p_list, skip_special_tokens=True) if piece_ids_full is not None else ""
        full_safe_target_path = full_context_text + " >>> " + safe_debug_text
        print(f"    [FULL TARGET NEW]    : {repr(full_safe_target_path)}")
        
        edit_input_token = tok.decode([input_ids_seq[0][lookup_idx_val].item()])
        target_token_str = tok.decode([target_token_id])
        print(f"    [Editing At]: Index {lookup_idx_val} (Skip {mismatch_offset_tokens}) | Input '{edit_input_token}' -> Predicts '{target_token_str}'")
        
        # === 5. 优化过程 ===
        loss_layer = max(hparams.v_loss_layer, layer)
        if hasattr(model.config, 'n_embd'):
            delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
        elif hasattr(model.config, 'hidden_size'):
            delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
        else: raise NotImplementedError
            
        target_init = None
        
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init
            if cur_layer == hparams.layer_module_tmp.format(layer):
                is_tensor = torch.is_tensor(cur_out)
                layer_out = cur_out if is_tensor else cur_out[0]
                if layer_out.dim() == 3:
                    batch_first = layer_out.size(0) == 1
                    if target_init is None:
                        target_init = (
                            layer_out[0, lookup_idxs[0], :] if batch_first
                            else layer_out[lookup_idxs[0], 0, :]
                        ).detach().clone()
                    for idxs_pre, delta_pre in all_delta:
                        for idx in idxs_pre:
                            if batch_first: layer_out[0, idx, :] += delta_pre
                            else: layer_out[idx, 0, :] += delta_pre
                    for idx in lookup_idxs:
                        if batch_first: layer_out[0, idx, :] += delta
                        else: layer_out[idx, 0, :] += delta
                elif layer_out.dim() == 2:
                    if target_init is None:
                        target_init = layer_out[lookup_idxs[0], :].detach().clone()
                    for idxs_pre, delta_pre in all_delta:
                        for idx in idxs_pre: layer_out[idx, :] += delta_pre
                    for idx in lookup_idxs: layer_out[idx, :] += delta
                if not is_tensor:
                    tmp = list(cur_out); tmp[0] = layer_out; cur_out = type(cur_out)(tmp)
            return cur_out

        opt = torch.optim.Adam([delta], lr=hparams.v_lr)
        nethook.set_requires_grad(False, model)
        
        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(input_ids_seq).logits
            
            log_probs = torch.log_softmax(logits, dim=2)
            
            # 1. 计算原始 Cross Entropy (NLL)
            # shape: [batch, seq]
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            
            # 2. 创建 Mask (忽略 -100)
            mask = (rewriting_targets != -100).float()
            
            # =====================================================
            # 【核心修改】EOS 加权逻辑
            # =====================================================
            # 默认权重全是 1.0
            weight_mask = torch.ones_like(loss)
            
            # 找到 Target 是 EOS 的位置
            # 注意：你的 tokenizer.eos_token_id 必须正确
            eos_positions = (rewriting_targets == tok.eos_token_id)
            
            # 如果找到了 EOS，给它加权 (比如 5.0 倍或 10.0 倍)
            if eos_positions.any():
                # print(f"    [EOS Boost] Boosting EOS weight at indices: {eos_positions.nonzero(as_tuple=True)[1].tolist()}")
                weight_mask[eos_positions] = 2.0 
            
            # 3. 计算加权后的 NLL
            # Loss 本身是负的 log_prob (因为我们取了 log_softmax)，所以要取负号变成正 Loss
            # 注意：torch.gather(log_probs) 得到的是 log(p)，是负数。
            # 我们要最小化的是 -log(p)。
            
            # loss 是 log_probs (负数)
            # 我们要的是 - (loss * mask * weight)
            nll_loss_each = -(loss * mask * weight_mask).sum(1) / mask.sum(1).clamp(min=1)
            
            # =====================================================
            
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (
                (torch.norm(delta) / torch.norm(target_init)) ** 2
            )
            
            total_loss = nll_loss + weight_decay.to(nll_loss.device)
            
            print(
                f"    [Optim] Step {it}: loss {np.round(total_loss.item(), 3)} = "
                f"{np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"Prob: {torch.exp(-nll_loss).item():.4f}"
            )
            
            if total_loss < 1e-2: break
            if it == hparams.v_num_grad_steps - 1: break
                
            total_loss.backward()
            opt.step()
            
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad(): delta[...] = delta * max_norm / delta.norm()
        
        target = target_init + delta
        all_delta.append((lookup_idxs, delta.clone()))
        all_target.append(target)
        all_idxs.append(lookup_idxs[0])
        
        
        # 【核心修复】保存 input_ids_seq 直到编辑点
        # 这样 main.py 拿到的 Input 就包含了 Context + Skipped Prefix (Safe/Unsafe Shared)
        # 确保 main.py 的计算不会越界，且 Key 的计算语境正确
        saved_input = input_ids_seq[0, :lookup_idx_val+1].cpu()
        all_inputs.append(saved_input)
        
        print(
            f"    [Result] Iter {len(all_delta)}: Init norm {target_init.norm():.2f} | "
            f"Delta norm {delta.norm():.2f} | Target norm {target.norm():.2f}"
        )
        print(f"    [Result] Saved Input Len: {len(saved_input)}, Val Target ID: {target_token_id}")
        # 加一句解码，看看它是啥
        print(f"    [Debug Decode] Target is: '{tok.decode([target_token_id])}'")

        current_token_idx += step
        win_idx += 1
        
    return all_inputs, all_idxs, all_target, all_validation_targets
