import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import nethook
import torch.optim as optim

import argparse

import numpy as np
import os
# Removed import of AttentionMaskConverter and _prepare_4d_causal_attention_mask to support newer transformers
from .unke_ARE_hparams import unkeAREHyperParams
def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: unkeAREHyperParams,
    layer: int,
    idxs_dict:dict,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to("cuda")
    zs_out_dict = {}

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    for k, idxs in idxs_dict.items():
        zs_out_list = []
        for idx in idxs:
            zs_out_list.append(zs_out[k,idx])
        zs_out_dict[k] = zs_out_list
    return zs_out_dict

def get_optimizer_params(model, encoder_lr, weight_decay=0.01):
        param_optimizer = list(model.named_parameters())
        no_decay = ["input_layernorm.weight", "post_attention_layernorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], # and 'mlp' in n
            'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': encoder_lr, 'weight_decay': 0.0},
        ]
        return optimizer_parameters




def apply_unke_ARE_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams:unkeAREHyperParams,
    batch_data:list,
    ex_data:list):

    preserve_params = []
    for name, params in model.named_parameters():
        #print(name)
        splitted_name = name.split('.')
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[2]):
            if int(splitted_name[2]) in hparams.layers:
                preserve_params.append(name)
    weights = {
        param: nethook.get_parameter(
            model, param)
        for param in preserve_params
    }
    
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}




    z_layer = hparams.layers[-1]
    zs_dict = {}
    idxs_dict = {}
    for k, data in enumerate(batch_data):
        
        idxs_list, target_list = compute_z(   
            model,
            tok,
            data,
            z_layer,
            hparams,
        )
        idxs_dict[k] = idxs_list
        zs_dict[k] = target_list
    batch_question_ans = [
        i['question'] + i['answer'] for i in batch_data
    ]
    
    # Insert
    for i, layer in enumerate(hparams.layers):
        #print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question_ans, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        

        cur_zs_dict = compute_ks(model, tok,batch_question_ans, hparams, z_layer, idxs_dict)
        targets_dict = {}
        for k, cur_zs_list in cur_zs_dict.items():
            zs_list = zs_dict[k]
            targets_list = [(a - b)/(len(hparams.layers) - i) for a, b in zip(zs_list, cur_zs_list)]
            targets_dict[k] = targets_list

        if isinstance(ex_data, str):
            ex_data = [ex_data] # 如果是单个字符串，将其包装成列表
    
            # 检查数据类型是否符合预期
            if not isinstance(ex_data, list) or (ex_data and not all(isinstance(s, str) for s in ex_data)):
                raise TypeError(f"ex_data 必须是 str 或 list[str] 类型, 但收到了: {type(ex_data)}")

            ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )


        ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.layer_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**ex_tok)
                stat_in = tr.input
                stat_out = tr.output
        stat_out = stat_out[0] if type(stat_out) is tuple else stat_out



        #resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers(1,4096)

        
        criterion = nn.MSELoss()
        
        _layer = nethook.get_module(model, hparams.layer_module_tmp.format(layer))
        
        for n,m in _layer.named_parameters():
            
            m.requires_grad=True
            
        params = get_optimizer_params(_layer,hparams.lr)
        
        
        optimizer = optim.AdamW(params,lr=hparams.lr,eps=1e-8,betas = (0.9,0.999))
        
        for k, idxs_list in idxs_dict.items():
            for j, idx in enumerate(idxs_list):
                resid = targets_dict[k][j]
                layer_out_ks[k,idx]+=resid
        
        input_causal_mask = None
        input_position_ids = None
        input_cache_position = None
        ex_causal_mask = None
        ex_position_ids = None
        ex_cache_position = None

        # 2. 生成 Mask (判定逻辑保持不变)
        if 'llama' in hparams.model_name.lower():
            input_causal_mask, input_position_ids, input_cache_position = get_causal_mask(layer_in_ks, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids, ex_cache_position = get_causal_mask(stat_in, ex_tok['attention_mask'])
        elif 'qwen' in hparams.model_name.lower():
            input_causal_mask, input_position_ids = get_qwen2_causal_mask(layer_in_ks, contexts_tok['attention_mask'])
            ex_causal_mask, ex_position_ids = get_qwen2_causal_mask(stat_in, ex_tok['attention_mask'])

        # --- 核心修复：手动计算 Position Embeddings ---
        # 无论是 Qwen 还是新版 Llama，单独调用层时都需要手动传入 position_embeddings
        
        # 初始化 embeddings 变量
        ex_position_embeddings = None
        input_position_embeddings = None
        
        # 检查模型是否需要手动计算 RoPE (Llama 和 Qwen 均需要)
        need_manual_rope = False
        if 'Qwen' in hparams.model_name or 'Llama' in hparams.model_name or 'llama' in hparams.model_name.lower():
            need_manual_rope = True

        if need_manual_rope:
            # 获取 rotary_emb 模块
            # 通常在 model.model.rotary_emb 下
            if hasattr(model.model, 'rotary_emb'):
                rotary_emb = model.model.rotary_emb
                
                # 1. 为保留集 (ex_data) 计算 cos, sin
                # rotary_emb 通常接受 (x, position_ids)
                # 注意：传入 stat_in 主要是为了获取 device 和 dtype
                ex_cos, ex_sin = rotary_emb(stat_in, position_ids=ex_position_ids)
                ex_position_embeddings = (ex_cos, ex_sin)
                
                # 2. 为编辑集 (input_data) 计算 cos, sin
                input_cos, input_sin = rotary_emb(layer_in_ks, position_ids=input_position_ids)
                input_position_embeddings = (input_cos, input_sin)
            else:
                print("Warning: Could not find 'rotary_emb' in model.model. Skipping manual RoPE computation.")

        # --- 优化循环 ---
        for step in range(hparams.optim_num_step):
            # scheduler.step()
            optimizer.zero_grad()
            
            # 根据模型类型调用层
            if 'Qwen3' in hparams.model_name or 'Qwen2' in hparams.model_name:
                # Qwen 逻辑
                loss = criterion(_layer(stat_in, attention_mask=ex_causal_mask, position_embeddings=ex_position_embeddings)[0], stat_out) \
                    + criterion(_layer(layer_in_ks, attention_mask=input_causal_mask, position_embeddings=input_position_embeddings)[0], layer_out_ks)

            elif hparams.model_name == 'Llama3-8B-Instruct' or 'llama' in hparams.model_name.lower():
                # Llama 逻辑修复：传入 position_embeddings
                
                # 安全检查
                if ex_position_embeddings is None:
                    raise ValueError("Error: position_embeddings is None. Llama layer execution failed.")

                loss = criterion(
                    _layer(
                        stat_in, 
                        attention_mask=ex_causal_mask, 
                        position_ids=ex_position_ids, 
                        cache_position=ex_cache_position,
                        position_embeddings=ex_position_embeddings  # <--- 必须传入这个!
                    )[0], 
                    stat_out
                ) + criterion(
                    _layer(
                        layer_in_ks, 
                        attention_mask=input_causal_mask, 
                        position_ids=input_position_ids, 
                        cache_position=input_cache_position,
                        position_embeddings=input_position_embeddings # <--- 必须传入这个!
                    )[0], 
                    layer_out_ks
                )
                
            loss.backward(retain_graph=True)
            optimizer.step()
                
            # print('Step [{}/{}], Loss: {:.4f}, Layer:{}'.format(step+1, hparams.optim_num_step, loss.item(),layer))
            # if loss.item() < 5e-5:
            #     break

        for x in [layer_in_ks, layer_out_ks,stat_in,stat_out]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
        
    return weights_copy
def get_qwen2_causal_mask(input_tensor, attention_mask, past_key_values_length=0):
    """
    为 Qwen2 / Qwen3 模型创建兼容新版 transformers 的 4D 因果注意力掩码。
    """
    device = input_tensor.device
    batch_size, seq_length = input_tensor.shape[0], input_tensor.shape[1]

    # 创建 position_ids
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    # 创建 4D 因果注意力掩码
    # 1. 创建一个下三角矩阵，值为0，上三角矩阵值为-inf
    causal_mask = torch.full(
        (seq_length, seq_length),
        fill_value=torch.finfo(input_tensor.dtype).min,
        dtype=input_tensor.dtype,
        device=device,
    )
    mask_cond = torch.arange(causal_mask.size(-1), device=device)
    causal_mask.masked_fill_(mask_cond < (mask_cond + 1).view(causal_mask.size(-1), 1), 0)
    
    # 2. 扩展到 4D: [batch_size, 1, seq_len, seq_len]
    expanded_causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length)

    # 3. 将 padding mask (attention_mask) 合并进来
    # attention_mask 是 [batch_size, seq_len]，需要扩展维度
    # padding 的位置 (值为0) 在最终的 mask 中也应该是 -inf
    if attention_mask is not None:
        padding_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length)
        # 如果一个位置在 padding_mask 中是 0，就在最终的 mask 中设置为 -inf
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
    causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit

    if attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    elif attention_mask.dim() == 4:
        # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
        # cache. In that case, the 4D attention mask attends to the newest tokens only.
        if attention_mask.shape[-2] < cache_position[0] + sequence_length:
            offset = cache_position[0]
        else:
            offset = 0
        mask_shape = attention_mask.shape
        mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
        causal_mask[
            : mask_shape[0], : mask_shape[1], offset : mask_shape[2] + offset, : mask_shape[3]
        ] = mask_slice

    #causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))
    return causal_mask,position_ids,cache_position