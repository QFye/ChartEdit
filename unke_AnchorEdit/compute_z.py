from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .unke_AnchorEdit_hparams import unkeAnchorEditHyperParams
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
    【修正版】支持 BOS (add_special_tokens=True) 的全局切片逻辑。
    解决首个锚点失效和输出乱码问题。
    """
    # 1. 构建全局文本
    full_text = question + base
    
    # 2. 全局分词：开启 add_special_tokens=True
    # 结果: [BOS, Q_tokens..., A_tokens...]
    full_ids = tok(full_text, add_special_tokens=True)["input_ids"]
    
    # 3. 确定 Answer 起始位置：开启 add_special_tokens=True
    # 结果: [BOS, Q_tokens...]
    # len(q_ids) 正好指向 Answer 的第一个 Token 在 full_ids 中的索引
    q_ids = tok(question, add_special_tokens=True)["input_ids"]
    answer_start_idx = len(q_ids)
    
    # --- 安全检查 ---
    if len(full_ids) < len(q_ids):
        return full_ids
    # ----------------
    
    # 4. 计算当前窗口的截断位置
    step = max(1, window_size - overlap)
    
    # prefix_tokens_in_answer: 在 Answer 部分我们需要保留多少个 Token
    prefix_tokens_in_answer = 0 if window_idx <= 0 else window_idx * step
    
    # 5. 计算绝对截断索引
    cutoff_idx = answer_start_idx + prefix_tokens_in_answer
    
    # 6. 边界保护
    if cutoff_idx > len(full_ids):
        cutoff_idx = len(full_ids)
        
    # 7. 切片返回
    context_ids = full_ids[:cutoff_idx]
    
    return context_ids

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: unkeAnchorEditHyperParams,
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
    
    # 【核心修改 1】: 开启 BOS
    full_unsafe_ids_global = tok(full_unsafe_text_global, add_special_tokens=True)["input_ids"]
    full_unsafe_tensor_global = torch.LongTensor(full_unsafe_ids_global).to("cuda")
    
    # 【核心修改 2】: Q 也开启 BOS
    q_ids = tok(raw_question, add_special_tokens=True)["input_ids"]
    
    # 【核心修改 3】: 计算 Answer 起始位置
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
            
            # =================================================================
            # 【核心修复】: 生成 full_safe_ids 时也必须开启 BOS (True)
            # 否则它会比带 BOS 的 context_ids 少一个 token，导致切片时多切掉一个！
            # =================================================================
            full_safe_ids = tok(full_safe_text, add_special_tokens=True)["input_ids"]
            
            if len(full_safe_ids) > len(context_ids):
                # 现在 full_safe_ids 和 context_ids 都有 BOS，len(context_ids) 切片是安全的
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
            if len(b_list) > len(p_list):
                b_list = b_list[:len(p_list)]
            
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
        
         # === 4. Target & Edit Point (核心修改部分) ===
        
        # 4.1 构造 Input: Context + Safe[:-1] (Teacher Forcing)
        # 将整个 Safe 序列（除了最后一个词）拼接到 Context 后面，作为输入历史
        safe_tensor = torch.LongTensor(p_list).unsqueeze(0).to("cuda")
        if safe_tensor.size(1) > 0:
             input_ids_seq = torch.cat([input_ids_context, safe_tensor[:, :-1]], dim=1)
        else:
             input_ids_seq = input_ids_context
             
        # 4.2 计算编辑点的绝对索引 (保持原有 Mismatch 定位逻辑)
        # 编辑点 = Context最后一个词 + Mismatch偏移量
        # 这确保了 delta 注入在模型开始犯错的精确位置
        lookup_idx_val = base_lookup_idx + mismatch_offset_tokens
        
        # 边界保护
        ex_len = input_ids_seq.size(1)
        if lookup_idx_val >= ex_len: lookup_idx_val = ex_len - 1
        
        # lookup_idxs 依然只包含一个点，即我们在单点注入 delta
        lookup_idxs = [lookup_idx_val]
        
        # 4.3 设置全窗口 Targets (全轨迹监督)
        # 初始化全为 -100 (忽略 Loss)
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(1, ex_len)
        target_token_id = tok.eos_token_id # Default
        
        if len(p_list) > 0:
            # 获取从 mismatch 点开始的所有剩余 Safe Tokens
            tokens_to_fill = safe_tensor[0][mismatch_offset_tokens:]
            
            # 起始填充位置 (Input Index)
            start_fill = lookup_idx_val
            
            # 计算可填充长度 (受限于 Input 长度)
            fill_len = min(len(tokens_to_fill), ex_len - start_fill)
            
            # 【关键修改】填充 Rewriting Targets
            # 从 start_fill (编辑点) 开始，一直监督到序列结束
            # input[start_fill]   -> 预测 tokens_to_fill[0] (即 p_list[mismatch])
            # input[start_fill+1] -> 预测 tokens_to_fill[1]
            # ...
            rewriting_targets[0, start_fill : start_fill + fill_len] = tokens_to_fill[:fill_len]
            
            if len(tokens_to_fill) > 0:
                target_token_id = tokens_to_fill[0].item()
        else:
            # Fallback: 如果没有 Safe 内容，只在编辑点预测 EOS
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
            
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            
            mask = (rewriting_targets != -100).float()
            
            # EOS 加权
            weight_mask = torch.ones_like(loss)
            eos_positions = (rewriting_targets == tok.eos_token_id)
            if eos_positions.any():
                weight_mask[eos_positions] = 1
            
            nll_loss_each = -(loss * mask * weight_mask).sum(1) / mask.sum(1).clamp(min=1)
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
        
        saved_input = input_ids_seq[0, :lookup_idx_val+1].cpu()
        all_inputs.append(saved_input)
        
        print(
            f"    [Result] Iter {len(all_delta)}: Init norm {target_init.norm():.2f} | "
            f"Delta norm {delta.norm():.2f} | Target norm {target.norm():.2f}"
        )
        print(f"    [Result] Saved Input Len: {len(saved_input)}, Val Target ID: {target_token_id}")
        print(f"    [Debug Decode] Target is: '{tok.decode([target_token_id])}'")
        current_token_idx += step
        win_idx += 1
        
    return all_inputs, all_idxs, all_target, all_validation_targets
