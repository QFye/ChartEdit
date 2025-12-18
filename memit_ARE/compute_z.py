from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .memit_ARE_hparams import MEMITAREHyperParams
from util import nethook
import nltk


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: MEMITAREHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    #print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(data["answer"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]  
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    
    input_tok = tok(
        [data["question"]],  
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    cur_input_ids = input_tok['input_ids'] 
    all_delta = []
    all_target = []
    all_idxs = []
    # ans_sen = nltk.tokenize.sent_tokenize(data["answer"], language='english')
    # cur_sen = ""
    # for i,sen in enumerate(ans_sen):
    #     if i > 0  and sen[0]!=" ":
    #         sen = " " + sen
    #     cur_sen = cur_sen + sen
    #     current_target_ids = tok(cur_sen, return_tensors="pt").to("cuda")["input_ids"][0]
    #     if current_target_ids[0] == tok.bos_token_id or current_target_ids[0] == tok.unk_token_id:
    #         current_target_ids = current_target_ids[1:]
    #     # if len(current_target_ids) < hparams.window_size:
    #     #     continue
    #     input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[:-1], dim=0)], dim=1)
    #     cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids, dim=0)], dim=1)
        

    start = 0
    while start < len(target_ids):
        end = start + hparams.window_size
        if end > len(target_ids):
            end = len(target_ids)
        current_target_ids = target_ids[start:end]
        if start > 0:
            input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[hparams.overlap:-1], dim=0)], dim=1)
            cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[hparams.overlap:], dim=0)], dim=1)
        else:
            input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[:-1], dim=0)], dim=1)
            cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids, dim=0)], dim=1)
        
        start += hparams.window_size - hparams.overlap
        
        rewriting_targets = torch.tensor(-100, device="cuda").repeat(
            1, len(input_ids[0])
        )
   
        ex_len = len(input_ids[0])
    
        rewriting_targets[0, ex_len - len(current_target_ids) : ex_len] = current_target_ids


        lookup_idxs = [ex_len - len(current_target_ids)]
    
        loss_layer = max(hparams.v_loss_layer, layer)
    
        if hasattr(model.config, 'n_embd'):
            delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
        elif hasattr(model.config, 'hidden_size'):
            delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
        else:
            raise NotImplementedError
        target_init = None
    
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init  

            if cur_layer == hparams.layer_module_tmp.format(layer):
                # Normalize output container: some models return Tensor, some (Tensor, ...)
                is_tensor = torch.is_tensor(cur_out)
                layer_out = cur_out if is_tensor else cur_out[0]

                # Determine indexing strategy based on dims
                # Expect one of: (batch, seq, hidden), (seq, batch, hidden), or (seq, hidden)
                if layer_out.dim() == 3:
                    batch_first = layer_out.size(0) == 1  # our batch size is 1
                    if target_init is None:
                        if batch_first:
                            target_init = layer_out[0, lookup_idxs[0], :].detach().clone()
                        else:
                            target_init = layer_out[lookup_idxs[0], 0, :].detach().clone()

                    # Apply previous deltas
                    for idxs_pre, delta_pre in all_delta:
                        for i, idx in enumerate(idxs_pre):
                            if batch_first:
                                layer_out[0, idx, :] += delta_pre
                            else:
                                layer_out[idx, 0, :] += delta_pre

                    # Apply current delta
                    for i, idx in enumerate(lookup_idxs):
                        if batch_first:
                            layer_out[0, idx, :] += delta
                        else:
                            layer_out[idx, 0, :] += delta

                elif layer_out.dim() == 2:
                    # Shape (seq, hidden), batch dim collapsed (batch=1). Index along seq.
                    if target_init is None:
                        target_init = layer_out[lookup_idxs[0], :].detach().clone()

                    for idxs_pre, delta_pre in all_delta:
                        for i, idx in enumerate(idxs_pre):
                            layer_out[idx, :] += delta_pre

                    for i, idx in enumerate(lookup_idxs):
                        layer_out[idx, :] += delta

                else:
                    raise RuntimeError(f"Unsupported layer output dim: {layer_out.dim()}")

                # Write back if original output was a container
                if not is_tensor:
                    tmp = list(cur_out)
                    tmp[0] = layer_out
                    cur_out = type(cur_out)(tmp)

            return cur_out

        # Optimizer
        opt = torch.optim.Adam([delta], lr=hparams.v_lr)
        nethook.set_requires_grad(False, model)  

        # Execute optimization
        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()

            # Forward propagation
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
                logits = model(input_ids).logits
                

            # Compute loss on rewriting targets

            raw_output = tr[hparams.layer_module_tmp.format(loss_layer)].output
            # Normalize to Tensor
            layer_out = raw_output if torch.is_tensor(raw_output) else raw_output[0]
            # Ensure shape is (batch, seq, hidden)
            if layer_out.dim() == 3:
                full_repr = layer_out
                # If shape is (batch, hidden, seq), transpose to (batch, seq, hidden)
                if full_repr.shape[1] != rewriting_targets.shape[1] and full_repr.shape[2] == rewriting_targets.shape[1]:
                    full_repr = full_repr.transpose(1, 2)
            elif layer_out.dim() == 2:
                # (seq, hidden) -> add batch dim
                full_repr = layer_out.unsqueeze(0)
                # If mistakenly (hidden, seq), fix to (seq, hidden) first
                if full_repr.shape[1] != rewriting_targets.shape[1] and full_repr.shape[2] == rewriting_targets.shape[1]:
                    full_repr = full_repr.transpose(1, 2)
            else:
                raise RuntimeError(f"Unexpected layer_out dims: {layer_out.dim()}")

            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / current_target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
            loss = nll_loss + weight_decay.to(nll_loss.device)
            print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}  + {np.round(weight_decay.item(), 3)} "
            # f"avg prob of [{cur_sen}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 1e-2:
               break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backpropagate
            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()
        # cur_sen = ""
        target = target_init + delta  
        all_delta.append((lookup_idxs ,delta.clone()))
        all_target.append(target)
        all_idxs.append(lookup_idxs[0])
        print(
        f"Iteration {len(all_delta)}: Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return all_idxs, all_target




