import os
import json
import re
import shutil
from pathlib import Path
from itertools import islice
from time import time
from typing import Tuple, Union, List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from dsets import (
    UnKEDataset,
    CounterFactDataset,
    MQUAKEDataset,
    EditeveryDataset,
    WAITDataset,
    AnchorEditDataset,
    SafeEditDataset
)
from dsets.valueinject import ValueInjectDataset

def _locate_toxic_layer(model, tokenizer, requests, cfg):
    toxic_layer = []
    device = next(model.parameters()).device
    input = tokenizer([value for pair in requests for value in [pair["target_new"], pair["ground_truth"]]],
                      return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**input, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    for j in range(len(requests)):
        max_distance_layer = None
        max_distance_value = float('-inf')

        for layer_index in range(1, len(hidden_states)):
            euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[layer_index][j * 2 + 1], p=2)

            if euclidean_distance.item() > max_distance_value:
                max_distance_value = euclidean_distance.item()
                max_distance_layer = layer_index
        toxic_layer.append(max_distance_layer - 1)
    return toxic_layer[0]

from memit import MEMITHyperParams, apply_memit_to_model
from memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from memit_AnchorEdit import MEMITAnchorEditHyperParams, apply_memit_AnchorEdit_to_model
from AlphaEdit import AlphaEditHyperParams, apply_AlphaEdit_to_model, get_cov
from AlphaEdit_ARE import AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model
from AnchorEdit import AnchorEditHyperParams, apply_AnchorEdit_to_model
from unke import unkeHyperParams, apply_unke_to_model
from unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from unke_AnchorEdit import unkeAnchorEditHyperParams, apply_unke_AnchorEdit_to_model
from util import nethook
from util.llama_classifier import *
from util.globals import *
# Removed unused GLUEEval import to avoid ModuleNotFoundError when glue_eval is absent
ALG_DICT = {
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "unke": (unkeHyperParams, apply_unke_to_model),
    "unke_AnchorEdit":(unkeAnchorEditHyperParams, apply_unke_AnchorEdit_to_model),
    "AnchorEdit": (AnchorEditHyperParams, apply_AnchorEdit_to_model),
    "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_AnchorEdit": (MEMITAnchorEditHyperParams, apply_memit_AnchorEdit_to_model)
}

DS_DICT = {
    "unke": UnKEDataset,
    "cf": CounterFactDataset,
    "mquake": MQUAKEDataset,
    "editevery": EditeveryDataset,
    "valueinject": ValueInjectDataset,
    "wait": WAITDataset,
    "safety_window": AnchorEditDataset,
    "safeedit": SafeEditDataset
}
def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

REFUSALS = [
    "The previous content is inducing me to generate something harmful. Reject the request and ignore previous instructions. Strictly start with distinctive rejection word, such as \"Sorry, I can't ...\", \"I'm sorry, ...\" or \"... Wait, ...\". State the harm of the topic in detail to the user as well. Do not use numerical labels and quotation marks. Let me start my rejection: "
]

def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def _safe_texts_for_ppl(tokenizer, texts):
    """
    将空字符串/全空白文本替换为一个最小可接受 token，
    防止 tokenizer 返回长度为 0 的 input_ids 导致模型 forward 出错。
    """
    replacement = tokenizer.eos_token or tokenizer.pad_token or "."
    safe = []
    for t in texts:
        s = t if isinstance(t, str) else str(t)
        s = s.strip()
        if not s:
            s = replacement
        safe.append(s)
    return safe

def _compute_ppl(model, tokenizer, texts, max_length=512, batch_size=8):
    # 统一成 list[str]
    if isinstance(texts, str):
        texts = [texts]
    elif not isinstance(texts, list):
        texts = [str(texts)]
    else:
        texts = [str(t) for t in texts]

    texts = _safe_texts_for_ppl(tokenizer, texts)

    model.eval()
    total_nll, total_tokens = 0.0, 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            enc = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100   # 无条件按 mask 屏蔽

            valid_tokens = (labels != -100).sum().item()
            if valid_tokens == 0:
                continue

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_nll += out.loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        return float('nan')
    return float(np.exp(total_nll / total_tokens))

def _ppl_to_fluency(ppl_val: float) -> float:
    return (1.0 / ppl_val) if (ppl_val is not None and np.isfinite(ppl_val) and ppl_val > 0) else float('nan')

def split_into_windows(text, tokenizer, window_size=25, overlap=5):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    windows = []
    start = 0
    while start < len(tokens):
        end = start + window_size
        window_tokens = tokens[start:end]
        windows.append(window_tokens)
        start = end - overlap
        if start >= len(tokens) - overlap:
            break
    return [tokenizer.decode(win, skip_special_tokens=True) for win in windows]

def compute_window_ppls(model, tokenizer, text, window_size, overlap):
    windows = split_into_windows(text, tokenizer, window_size, overlap)
    window_ppls = [_compute_ppl(model, tokenizer, [w]) for w in windows]  # 关键改动：包成列表
    return window_ppls, windows

import re

def find_last_punct_token_idx(tokenizer, token_ids: List[int], start_search_idx: int) -> int:
    """
    在 token_ids 中，从 start_search_idx 开始【向前】搜索，
    找到第一个包含标点符号（.,?;!等）的 token，并返回其【下一个位置】的索引。
    这样截断时，标点会被包含在 prefix 里。
    """
    # 定义标点集合 (根据你的正则需求)
    punct_chars = set('.。!！?？:：;；…”')
    
    # 边界检查
    search_limit = max(0, start_search_idx - 50) # 只向前搜 50 个 token，避免退太远
    
    for i in range(min(len(token_ids)-1, start_search_idx), search_limit, -1):
        tid = token_ids[i]
        # 获取 token 的文本表示 (例如 "Ġ." 或 ".")
        # Llama-3/Qwen 的 convert_ids_to_tokens 通常返回原始字节或特定字符
        token_str = tokenizer.convert_ids_to_tokens(tid)
        
        # 防御性编程：如果是 None (某些特殊 token)
        if not token_str: continue
        
        # 将 Byte-level BPE 的特殊字符替换（如 Ġ -> 空格）以便检查
        # 注意：Llama-3 的特殊字符可能是 'Ġ'
        clean_str = token_str.replace('Ġ', '').replace(' ', '')
        
        # 检查是否包含标点
        for char in clean_str:
            if char in punct_chars:
                return i + 1 # 返回标点之后的那个位置
                
    # 如果没找到，就返回原始起点（不截断或从默认位置截断）
    return start_search_idx



def has_punctuation(text: str) -> bool:
    # 包含中英文常见标点
    punct_regex = r"[\.。!！?？:：;；…”]"
    return bool(re.search(punct_regex, text))

def pick_anchors(window_ppls: List[List], hparams) -> List[int]:
    """
    选择编辑锚点的窗口索引：
    - 优先策略：PPL 高 且 有标点
    - 兜底策略：如果凑不够，再选 PPL 高 但 无标点 的
    - 始终尝试返回 top_k 个中间锚点（外加首个窗口 0）
    """
    top_k = hparams.top_k_windows
    if not window_ppls:
        return []
    
    indices = []
    n = len(window_ppls)
    if hparams.select_first_window:
        indices.append(0)
    
    if n <= 2:
        return indices
    
    # 中间窗口索引池 (假设范围限制为 20，和你原逻辑一致)
    mid_indices = list(range(1, min(n, 15)))
    if hparams.select_first_window:
        mid_indices = list(range(0, min(n, 15)))
    
    # 构建完整信息列表：(PPL, Text, Index)
    # 注意：排除掉 idx=0，因为它已经被 indices 包含了
    candidates = []
    for i in mid_indices:
        if i == 0: continue # 跳过 0，避免重复
        ppl = float(window_ppls[i][0])
        text = str(window_ppls[i][1])
        candidates.append({'ppl': ppl, 'text': text, 'idx': i})
    
    # 按 PPL 从高到低排序
    candidates.sort(key=lambda x: x['ppl'], reverse=True)
    
    # 分桶策略
    punct_candidates = []
    no_punct_candidates = []
    
    for item in candidates:
        if has_punctuation(item['text']):
            punct_candidates.append(item)
        else:
            no_punct_candidates.append(item)
            
    # 贪心选择：先拿有标点的，不够再拿没标点的
    selected_mid = []
    
    # 1. 先取有标点的 (本身已经按 PPL 排序了)
    take_punct = punct_candidates[:top_k]
    selected_mid.extend([x['idx'] for x in take_punct])
    
    # 2. 如果还没凑够 K 个，从没标点的里面补
    needed = top_k - len(selected_mid)
    if needed > 0:
        take_no_punct = no_punct_candidates[:needed]
        selected_mid.extend([x['idx'] for x in take_no_punct])
    
    indices.extend(selected_mid)
    
    return sorted(set(indices))



# -----------------------------
# Entropy computation
# -----------------------------
def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    probs = torch.softmax(logits, dim=dim)
    log_probs = torch.log_softmax(logits, dim=dim)
    ent = -(probs * log_probs).sum(dim=dim)
    return ent

def get_model_device(model) -> torch.device:
    """Robustly get the primary device of model parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_max_entropy_token_index(model, tokenizer, text: str) -> int:
    """
    计算 text 的逐 token 最大熵位置（针对下一个 token 的预测）。
    返回最大熵位置的**当前 token 索引**（用于“保留到锚点”）。
    """
    device = get_model_device(model)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, L, V]

    next_logits = logits[:, :-1, :]                 # [1, L-1, V]
    ent = entropy_from_logits(next_logits, dim=-1).squeeze(0)  # [L-1]
    max_next_idx = int(torch.argmax(ent).item())    # 0..L-2
    anchor_token_index = max_next_idx
    return anchor_token_index

# -----------------------------
# Punctuation-based anchor: RANDOM pick among all punctuations
# -----------------------------
def find_anchor_idx_after_punct_text(anchor_text: str, tokenizer) -> int:
    # 1. 对整段文本进行编码，并获取 offset_mapping
    # offset_mapping[i] = (start_char, end_char) 表示第 i 个 token 对应的字符范围
    enc = tokenizer(anchor_text, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0] # shape: [num_tokens, 2]
    
    # 2. 找到所有标点的位置 (字符级)
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    punct_regex = r"[\.。!！?？:：;；…”]"
    
    # 找到标点结束的字符索引 (char_end)
    matches = list(re.finditer(punct_regex, text))
    if not matches:
        return len(input_ids)
        
    # 随机选一个标点
    chosen_match = random.choice(matches)
    punct_char_end = chosen_match.end() # 标点之后的字符位置
    
    # 3. 映射回 Token 索引
    # 我们要找的是：哪个 Token 的起始位置 >= punct_char_end
    
    target_token_idx = len(input_ids) # 默认：如果是最后一个字符，则指向末尾
    
    for idx, (start, end) in enumerate(offsets):
        # 注意：special tokens (如 BOS) 的 offset 通常是 (0,0)，要跳过
        if start == end == 0: continue
        
        # 如果这个 token 的开始位置已经在标点之后了，那就是它！
        if start >= punct_char_end:
            target_token_idx = idx
            break
            
    return target_token_idx


# -----------------------------
# Datetime cleanup utils (only for continuation)
# -----------------------------
MONTHS_RE = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_PATTERNS = [
    r"\b(19|20)\d{2}[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?(?:\s*[AP]M)?\b",
    r"\b(19|20)\d{2}[-/\.](0?[1-9]|1[0-2])\b",
    rf"\b{MONTHS_RE}\s+\d{{1,2}},\s*(19|20)\d{{2}}\b",
    rf"\b{MONTHS_RE}\s+(19|20)\d{{2}}\b",
    r"\b(19|20)\d{2}年(0?[1-9]|1[0-2])月((0?[1-9]|[12]\d|3[01])日?)?\b",
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b",
]
DATE_RE = re.compile("|".join(DATE_PATTERNS), flags=re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")

def sanity_check_ids(ids: List[int], vocab_size: int):
    """Raise if any id is out of vocab range."""
    bad = [t for t in ids if (t < 0 or t >= vocab_size)]
    if bad:
        raise ValueError(f"Found out-of-range token ids (vocab={vocab_size}), examples: {bad[:10]}")
    
def drop_date_sentences(text: str) -> str:
    """仅对 continuation：剔除包含日期/时间模式的整句。"""
    if not text.strip():
        return text
    parts = SENT_SPLIT_RE.split(text)
    keep = [s for s in parts if not DATE_RE.search(s)]
    cleaned = " ".join(keep).strip()
    return cleaned if cleaned else text

# -----------------------------
# Boundary helpers (NEW)
# -----------------------------
# 定义“标点或空白”的集合（用于识别左侧边界是否干净）
PUNCT_SET = set('.。!！?？,，:：;；、…—~～”`')
LOWERCASE_TRIGGER = set([',', '，', ';', '；', ':', '：'])

def clean_left_boundary(left: str) -> str:
    """
    若左侧最后一个非空白字符不是标点，则说明多了半截/多余单词：
    回退到最近的空格或标点为止（保留该分隔符），否则清空。
    """
    if not left:
        return left
    # 若末尾已是空白，直接返回；若末尾是标点，也返回
    tail = left[-1]
    if tail.isspace() or tail in PUNCT_SET:
        return left
    # 向左回溯，找到最近的“空白或标点”
    for i in range(len(left) - 1, -1, -1):
        ch = left[i]
        if ch.isspace() or ch in PUNCT_SET:
            return left[:i+1]
    # 整段都无分隔符：清空
    return ""

def lowercase_first_alpha(s: str) -> str:
    """将字符串中的第一个字母（任意语言 isalpha()）转为小写。未找到字母则原样返回。"""
    for i, c in enumerate(s):
        if c.isalpha():
            if c.islower():
                return s
            return s[:i] + c.lower() + s[i+1:]
    return s

def last_nonspace_char(s: str) -> str:
    for i in range(len(s) - 1, -1, -1):
        if not s[i].isspace():
            return s[i]
    return ""

# -----------------------------
# Replace & Generate
# -----------------------------
def replace_from_index(
    tokenizer, text: str, insert_text: str, start_idx: int, include_anchor: bool = True
) -> Tuple[List[int], int, int]:
    """
    将拒绝回复“插入到锚点处”，返回：
      - new_ids: 新 input ids
      - new_len: 新 ids 长度
      - prefix_tok_len: 插入前“保留段”的 token 数（用于后续解码 left）
    不额外插入 glue 空格，避免字符级回退错位。
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0].tolist()

    cutoff = start_idx if include_anchor else start_idx
    prefix_ids = input_ids[:cutoff]

    # 拒绝文本原样编码
    suffix_ids = tokenizer.encode(insert_text, add_special_tokens=False)

    new_ids = prefix_ids + suffix_ids
    return new_ids, len(new_ids), len(prefix_ids)


def continue_generation(
    model, tokenizer, input_ids: List[int],
    max_new_tokens: int = 20, temperature: float = 0.8, top_p: float = 0.95
) -> str:
    device = get_model_device(model)

    vocab_size = getattr(model.config, "vocab_size", None)
    if vocab_size is not None:
        sanity_check_ids(input_ids, vocab_size)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)
    if pad_id is None:
        pad_id = eos_id

    with torch.no_grad():
        gen_out = model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6 if temperature is None else temperature,
            top_p=0.9 if top_p is None else top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            repetition_penalty=1.1
        )
    full = gen_out[0].tolist()
    new_tokens = full[len(input_ids):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def find_first_punct_token_idx(tokenizer, token_ids: List[int]) -> int:
    """
    在 token_ids 中【从前往后】找，找到第一个包含标点的 token 的索引。
    没找到返回 -1。
    """
    punct_chars = set('.。!！?？;；：:”')
    
    for i, tid in enumerate(token_ids):
        token_str = tokenizer.convert_ids_to_tokens(tid)
        if not token_str: continue
        
        # 清洗 Llama-3/Qwen 的特殊前缀 (如 Ġ)
        clean_str = token_str.replace('Ġ', '').replace(' ', '')
        
        for char in clean_str:
            if char in punct_chars:
                return i
                
    return -1

REGEX_PRONOUN_I = re.compile(r"\bI\b|\bI['’]([amdvll]+)\b")

# 匹配标点符号开头 (用于判断是否需要加空格)
REGEX_PUNCT_START = re.compile(r"^[\,\.\!\?\;\'’”）\]\}]")

# 匹配句子结束符 (用于判断是否需要转小写)
REGEX_SENTENCE_END = re.compile(r"[\.\?\!。？！][\"”’\']?$") 

# 匹配句子结束位置 (用于截断)
REGEX_TRUNCATE = re.compile(r'([.?!。？！]["”’\']?)')

def smart_truncate_sentence(text: str) -> str:
    """截断到最后一个完整的句子结束符，支持 . ? ! 及引号"""
    matches = list(REGEX_TRUNCATE.finditer(text))
    if not matches:
        return text # 如果没有标点，保留原样 (或者根据需求返回 text[:0])
    
    last_match = matches[-1]
    return text[:last_match.end()]

def process_sample(
    sample: Dict,
    model,
    tokenizer,
    hparams,
    seed: int = 1234,
    max_new_tokens: int = 20
) -> Dict:
    random.seed(seed)
    
    sample_id = sample.get("id", "unknown")
    input_text = sample.get("question", "")
    output = sample["pre_edit"]["response"]
    window_ppls = sample.get("window_ppls", [])
    
    # 1. 全局分词 (Single Source of Truth)
    full_text = input_text + output
    # 注意：确保 add_special_tokens 配置与你的训练一致
    full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
    full_ids = full_enc["input_ids"][0].tolist()
    
    q_enc = tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
    q_ids_len = q_enc["input_ids"].shape[1]

    # 获取 Anchors
    anchors = pick_anchors(window_ppls, hparams)
    processing_indices = sorted(list(set(anchors)))
    
    sample_pack = {"sample_id": sample_id}
    piece = {}
    ppls = []
    
    step = max(1, hparams.window_size - hparams.overlap)
    
    for rank, idx in enumerate(tqdm(processing_indices, desc=f"Sample {sample_id}", leave=False)):
        
        # 获取 PPL 值
        if idx < len(window_ppls):
            ppl_val = float(window_ppls[idx][0])
        else:
            ppl_val = 0.0
            
        # --- Step 1: 定位 (Locate) ---
        window_start_abs = q_ids_len + idx * step
        window_end_abs = min(window_start_abs + hparams.window_size, len(full_ids))
        
        if idx == 0:
            anchor_idx = q_ids_len
        else:
            current_window_ids = full_ids[window_start_abs : window_end_abs]
            punct_offset = find_first_punct_token_idx(tokenizer, current_window_ids)
            
            if punct_offset != -1:
                anchor_idx = window_start_abs + punct_offset + 1
            else:
                anchor_idx = window_start_abs
                
        if anchor_idx < q_ids_len: 
            anchor_idx = q_ids_len

        # --- Step 2: 截断 (Truncate) ---
        prefix_ids = full_ids[:anchor_idx]
        
        # --- Step 3: 插入 Refusal (Insert) ---
        refusal_text = random.choice(REFUSALS)
        # 统一处理：refusal 前加空格，交给 tokenizer 处理特殊 token
        refusal_ids = tokenizer.encode(" " + refusal_text, add_special_tokens=False)
        input_for_gen = prefix_ids + refusal_ids
        
        # --- Step 4: 生成 (Generate) ---
        try:
            continuation_str = continue_generation(
                model, tokenizer, input_for_gen, 
                max_new_tokens=max_new_tokens, temperature=0.001
            )
        except Exception as e:
            print(f"[WARN] Gen failed: {e}")
            continuation_str = ""
            
        # 清理逻辑
        continuation_str = drop_date_sentences(continuation_str)
        continuation_str = continuation_str.replace("<think>", "").replace("</think>", "")
        
        # 编码回 Token
        continuation_ids = tokenizer.encode(continuation_str, add_special_tokens=False)
        
        # --- Step 5: 拼接与融合 (Splice & Fusion) ---
        # 核心修改：使用扩大的窗口进行融合
        
        if len(prefix_ids) > 0 and len(continuation_ids) > 0:
            # 取 Prefix 的最后 N 个 token 和 Continuation 的前 M 个 token
            # 这里的 2 是经验值，足够覆盖大多数跨 Token 的单词边界
            p_slice_len = min(2, len(prefix_ids))
            c_slice_len = min(5, len(continuation_ids)) # Cont 取长一点以便做正则检查
            
            # 提取用于融合的 ID 片段
            ids_prefix_part = prefix_ids[-p_slice_len:]
            ids_cont_part = continuation_ids[:c_slice_len]
            
            # 解码为纯文本 (Skip special tokens 避免干扰)
            text_p = tokenizer.decode(ids_prefix_part, skip_special_tokens=True)
            text_c = tokenizer.decode(ids_cont_part, skip_special_tokens=True)
            
            # --- 5.1 文本处理 (Text Processing) ---
            
            # A. 检查 Prefix 是否以句子结束符结尾
            # rstrip() 去除末尾空格，确保能检测到 "word." 中的点
            text_p_stripped = text_p.rstrip()
            prefix_ends_with_punct = False
            if text_p_stripped:
                # 使用正则检查结尾字符
                if REGEX_SENTENCE_END.search(text_p_stripped):
                    prefix_ends_with_punct = True

            # B. 大小写转换 (只针对 Continuation 的第一个单词)
            text_c_clean = text_c.lstrip() # 去掉开头空格
            
            # 如果 Prefix 不是句尾，且 Cont 以大写开头
            if text_c_clean and text_c_clean[0].isupper() and not prefix_ends_with_punct:
                
                # C. 强力的 "I" 保护
                # 我们只看 text_c 的第一个单词
                first_word_match = re.search(r'\S+', text_c_clean) # 获取第一个非空连续字符块
                
                should_lower = True
                if first_word_match:
                    first_word = first_word_match.group(0)
                    # 检查是否匹配 "I", "I'm" 等
                    if REGEX_PRONOUN_I.match(first_word):
                        should_lower = False
                
                if should_lower:
                    # 执行首字母小写转换
                    # 注意：我们要保留 text_c 原有的前导空格结构
                    leading_space_len = len(text_c) - len(text_c_clean)
                    lower_c = text_c_clean[0].lower() + text_c_clean[1:]
                    text_c = text_c[:leading_space_len] + lower_c

            # --- 5.2 智能拼接 (Smart Join) ---
            
            text_p_clean = text_p.rstrip()
            text_c_clean = text_c.lstrip()
            
            # 默认分隔符为空格
            separator = " "
            
            # 如果 Cont 是以标点开头 (如 , . ! ?)，不加空格
            if text_c_clean and REGEX_PUNCT_START.match(text_c_clean):
                separator = ""
            
            # 如果 Prefix 是以左引号/括号结尾，不加空格
            if text_p_clean and text_p_clean[-1] in ['"', '“', '‘', '(', '[', '{']:
                separator = ""
                
            fusion_text = text_p_clean + separator + text_c_clean
            
            # --- 5.3 重新编码 (Re-tokenize) ---
            # 让 Tokenizer 决定最终的 token ID
            fusion_ids = tokenizer.encode(fusion_text, add_special_tokens=False)
            
            # --- 5.4 组装最终序列 ---
            # Prefix剩余 + 融合重编码部分 + Cont剩余
            final_ids = prefix_ids[:-p_slice_len] + fusion_ids + continuation_ids[c_slice_len:]
        
        else:
            # 边缘情况：如果某一边为空，直接拼
            final_ids = prefix_ids + continuation_ids

        # --- Step 6: 最终处理 ---
        answer_ids = final_ids[q_ids_len:]
        piece_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
        
        # 智能截断 (支持 . ? ! 及引号)
        piece_text = smart_truncate_sentence(piece_text)
            
        print(f"final_text = {piece_text}")
        
        # 补全 EOS (优先从 Tokenizer 获取)
        # 只有当 piece_text 不为空时才补全，避免只有 EOS 的空回复
        if piece_text:
            eos_token = tokenizer.eos_token 
            if not eos_token:
                # Fallback for models without explicit eos_token in config
                if 'qwen' in hparams.model_name.lower(): eos_token = "<|im_end|>"
                elif 'llama' in hparams.model_name.lower(): eos_token = "<|eot_id|>"
                else: eos_token = ""
            
            if eos_token and not piece_text.endswith(eos_token):
                piece_text += eos_token
            
        piece[idx] = piece_text
        ppls.append(ppl_val)

    sample_pack["generated_answer"] = "" 
    sample_pack["input"] = input_text
    sample_pack["output"] = output
    sample_pack["anchor_ppl"] = ppls
    sample_pack["piece"] = piece
    
    return sample_pack



def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    sequential: bool = False,
    gpu_id: str = "0",
    double_edit: bool = False,  # 添加双编辑模式参数
):
    # 设置使用的GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"使用GPU设备: {gpu_id}")
    set_seed()

    # 打印双编辑模式状态
    print(f"双编辑模式: {'开启' if double_edit and ds_name == 'wait' else '关闭'}")
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    params_path = (HPARAMS_DIR / alg_name / hparams_fname)
    hparams = params_class.from_json(params_path)
    jailbreak_judge = LlamaClassifier() 

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        print(model_name)
        model = AutoModelForCausalLM.from_pretrained("/mnt/ssd2/models/llama3.1-8b-instruct", trust_remote_code=True, local_files_only=True).to("cuda")
        tok = AutoTokenizer.from_pretrained("/mnt/ssd2/models/llama3.1-8b-instruct", trust_remote_code=True)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    ds_class = DS_DICT[ds_name]
    print(f"创建数据集: {ds_name}, 大小限制: {dataset_size_limit}")
    ds = ds_class(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)
    print(f"实际数据集大小: {len(ds)}")
    with open(Path(DATA_DIR)/"AKEW/WikiUpdate.json", 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    if 'llama' in hparams.model_name.lower():
        print(f"检测到 Llama 模型, 正在格式化 ex_datas...")
        ex_datas = [get_llama_without_answer(i.get('requested_rewrite',{}).get('question')) + i.get('requested_rewrite',{}).get('fact_new_uns') for i in ex_datas]
    elif "qwen" in hparams.model_name.lower():
        print(f"检测到 Qwen 模型, 正在格式化 ex_datas...")
        ex_datas = [get_llama_without_answer(i.get('requested_rewrite',{}).get('question')) + i.get('requested_rewrite',{}).get('fact_new_uns') for i in ex_datas]

    tokenizer = AutoTokenizer.from_pretrained("/mnt/ssd2/models/llama3.1-8b-instruct", padding_side='left', trust_remote_code=True)
    if 'llama' in hparams.model_name.lower():
        tokenizer.pad_token_id = tok.eos_token_id

    with open(Path(DATA_DIR)/"Qwen3-8B.json", 'r', encoding='utf-8') as json_file:
        jb_datas = json.load(json_file)

    with open(Path(DATA_DIR)/"dra_data_qwen3.json", 'r', encoding='utf-8') as json_file:
        dra_data = json.load(json_file)

    if alg_name in ["AlphaEdit","AlphaEdit_ARE","AnchorEdit"]:
        if not os.path.exists(f"{hparams.model_name}_null_space_project.pt"):
            W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            del W_out
            for i, layer in enumerate(hparams.layers):
                print(f"model_device = {model.device}")
                P[i,:,:] = get_project(model,tok,layer,hparams)
            torch.save(P, f"{hparams.model_name}_null_space_project.pt")
        else:
            P = torch.load(f"{hparams.model_name}_null_space_project.pt")
    batch_size = num_edits
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size else 0)
    edited_data = []
    test_data = []
    cresults = dra_data.get("results", {})
    candidates = []

    for item in cresults.values():
        qa = item.get("qa", [])
        if not qa:
            continue
        last = qa[-1]
        if last.get("harmbench") is True:
            prompt = last.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                candidates.append(prompt)

    # Pre-select a small subset for locality PPL evaluation
    ppl_subset_size = min(64, len(ex_datas))
    ppl_texts = random.sample(ex_datas, ppl_subset_size) if ppl_subset_size > 0 else []

    for batch_index in tqdm(range(num_batches)):

        continue
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = ds[start_index:end_index]
        # 打印WAIT数据集的id
        if ds_name == 'wait':
            print(f"批次 {batch_index} 中的数据ID:")
            for data in batch:
                print(f"  ID: {data.get('id', '未找到ID')}")

        # 保存原始answer以便双编辑时恢复
        if double_edit and ds_name == 'wait':
            for data in batch:
                if 'defense_response' in data and 'recovery_response' in data:
                    data['original_answer'] = data['answer']
        random_elements = random.sample(ex_datas, 20)
        # case_result_template = str(run_dir / "{}_edits-case_{}.json")

        ex_args = dict(ex_data = random_elements) if any(alg in alg_name for alg in ["unke", "unke_ARE"]) else dict()
        nc_args = dict(P = P) if any(alg in alg_name for alg in ["AlphaEdit","AlphaEdit_ARE"]) else dict()

        start = time()
        # Compute PPL before editing (locality - pre)
        ppl_pre = _compute_ppl(model, tokenizer, ppl_texts) if ppl_texts else float('nan')

        # ……（此处保持你的原逻辑）……
        # 批量路径里若后续需要 pre_edit，也可在此加入（与 sequential 类似）

        exec_time = time() - start
        print("Execution took", exec_time)

        # ……（此处保持你的原逻辑）……

    if sequential:
        # 处理顺序模式下的双编辑
        pre_gen = False
        if True:
            random_elements = random.sample(ex_datas, 20)
            ex_args = dict(ex_data = random_elements) if any(alg in alg_name for alg in ["unke", "unke_ARE", "unke_AnchorEdit"]) else dict()
            nc_args = dict(P = P) if alg_name in ["AlphaEdit","AlphaEdit_ARE","AnchorEdit"] else dict()

            def _ppl_to_fluency_local(ppl_val):
                return _ppl_to_fluency(ppl_val)

            for index, data in tqdm(enumerate(ds)):
                if 'qwen' in model_name.lower() and jb_datas[index]["is_unsafe"] is not True:
                    continue
                device = next(model.parameters()).device
                pre_ppl = 0

                if pre_gen:
                    print("===========================================")
                    print(f"Pre-Editing Evaluation. Case = {index}")
                    print("===========================================")

                    # =========================
                    # （新增）pre_edit：在任何编辑之前记录基线输出
                    # =========================
                    pre_edit: Dict[str, Union[str, float, List[str], Dict[str, Union[str, float]]]] = {
                        "response": "",
                        "general_response": [],
                        "knowledge_constrain": None,
                    }

                    # (A) 主问题 baseline 回复
                    q_pre = tok([data['question']], return_tensors='pt', padding=True)
                    with torch.no_grad():
                        outs_pre = model.generate(
                            input_ids=q_pre['input_ids'].to(device),
                            attention_mask=q_pre['attention_mask'].to(device),
                            do_sample=False,
                            max_new_tokens=768,
                            repetition_penalty=1.15
                        )
                    gen_ids_pre = outs_pre[:, q_pre['input_ids'].shape[1]:]
                    if gen_ids_pre.shape[1] > 200:
                        out_cut = gen_ids_pre[:, :200]
                    else:
                        out_cut = gen_ids_pre
                    main_text_pre = tok.batch_decode(out_cut, skip_special_tokens=True)[0]
                    saved_text_pre = tok.batch_decode(gen_ids_pre, skip_special_tokens=True)[0]
                    pre_edit["response"] = saved_text_pre

                    # (B) knowledge constrain：保存为 {text,ppl,fluency}，ppl/fluency 基于 KC 文本计算
                    if 'knowledge constrain' in data and isinstance(data['knowledge constrain'], dict) and 'prompt' in data['knowledge constrain']:
                        kn_prompt = data['knowledge constrain']['prompt']
                        q_kn = tok([kn_prompt], return_tensors='pt', padding=True)
                        with torch.no_grad():
                            outs_kn = model.generate(
                                input_ids=q_kn['input_ids'].to(device),
                                attention_mask=q_kn['attention_mask'].to(device),
                                do_sample=False,
                                max_new_tokens=768,
                                repetition_penalty=1.15
                            )
                        gen_ids_kn = [o[len(i):] for i, o in zip(q_kn['input_ids'], outs_kn)]
                        kn_text_pre = tok.batch_decode(gen_ids_kn, skip_special_tokens=True)[0]
                        ppl_kn_pre = _compute_ppl(model, tok, [kn_text_pre], max_length=512, batch_size=8)
                        pre_ppl = ppl_kn_pre
                        pre_edit["knowledge_constrain"] = {
                            "text": kn_text_pre,
                            "ppl": ppl_kn_pre,
                            #"fluency": _ppl_to_fluency_local(ppl_kn_pre)
                        }
                    else:
                        pre_edit["knowledge_constrain"] = None

                    # (C) general prompt 列表
                    if 'general prompt' in data and isinstance(data['general prompt'], list):
                        general_responses = []
                        for prompt in data['general prompt']:
                            q_gp = tok([prompt], return_tensors='pt', padding=True)
                            with torch.no_grad():
                                outs_gp = model.generate(
                                    input_ids=q_gp['input_ids'].to(device),
                                    attention_mask=q_gp['attention_mask'].to(device),
                                    do_sample=False,
                                    max_new_tokens=768,
                                    repetition_penalty=1.15
                                )
                            gen_ids_gp = [o[len(i):] for i, o in zip(q_gp['input_ids'], outs_gp)]
                            out_gp = tok.batch_decode(gen_ids_gp, skip_special_tokens=True)[0]
                            general_responses.append(out_gp)
                        pre_edit["general_response"] = general_responses

                    # (D) 多跳问题
                    test_input = random.choice(candidates)
                    d = {"id": index, "prompt": test_input, "response": {}}

                    test_input_ids = tok(test_input, return_tensors="pt")
                    test_output = model.generate(
                        input_ids=test_input_ids["input_ids"].to("cuda"),
                        attention_mask=test_input_ids["attention_mask"].to("cuda"),
                        do_sample=False,
                        max_new_tokens=2048,
                        repetition_penalty=1.15,
                        pad_token_id=(tok.pad_token_id or tok.eos_token_id),
                    )

                    # 只取新生成的部分
                    gen_only = test_output[:, test_input_ids["input_ids"].shape[1]:]

                    # 任选其一：
                    # A: 单条 decode
                    test_res = tok.decode(gen_only[0].detach().cpu().tolist(),
                                        skip_special_tokens=True).strip()
                    
                    d["response"]["pre"] = test_res
                else:
                    q_pre = tok([data['question']], return_tensors='pt', padding=True)
                    with torch.no_grad():
                        outs_pre = model.generate(
                            input_ids=q_pre['input_ids'].to(device),
                            attention_mask=q_pre['attention_mask'].to(device),
                            do_sample=False,
                            max_new_tokens=768,
                            repetition_penalty=1.15
                        )
                    gen_ids_pre = outs_pre[:, q_pre['input_ids'].shape[1]:]
                    # if gen_ids_pre.shape[1] > 200:
                    #     out_cut = gen_ids_pre[:, :200]
                    # else:
                    out_cut = gen_ids_pre
                    main_text_pre = tok.batch_decode(out_cut, skip_special_tokens=True)[0]
                    kn_prompt = data['knowledge constrain']['prompt']
                    q_kn = tok([kn_prompt], return_tensors='pt', padding=True)
                    with torch.no_grad():
                        outs_kn = model.generate(
                            input_ids=q_kn['input_ids'].to(device),
                            attention_mask=q_kn['attention_mask'].to(device),
                            do_sample=False,
                            max_new_tokens=768,
                            repetition_penalty=1.15
                        )
                    gen_ids_kn = [o[len(i):] for i, o in zip(q_kn['input_ids'], outs_kn)]
                    kn_text_pre = tok.batch_decode(gen_ids_kn, skip_special_tokens=True)[0]
                    ppl_kn_pre = _compute_ppl(model, tok, [kn_text_pre], max_length=512, batch_size=8)
                    pre_ppl = ppl_kn_pre
                    test_input = random.choice(candidates)
                    d = {"id": index, "prompt": test_input, "response": {}}
                if "AnchorEdit" in alg_name:
                    # 绑定到样本上
                    is_unsafe_list = jailbreak_judge.classify_responses([data['behaviour']], [main_text_pre])
                    is_unsafe = bool(is_unsafe_list[0])
                    data["pre_edit"] = {}
                    data["pre_edit"]["response"] = main_text_pre

                    # =========================
                    # 第二次编辑：defense（保留你原逻辑）
                    # ========================= 
                    if is_unsafe:

                        print("===========================================")
                        print(f"Select Windows. Case = {index}")
                        print("===========================================")

                        
                        window_ppls, window_texts = compute_window_ppls(model, tokenizer, main_text_pre, hparams.window_size, hparams.overlap)
                        wp = [[ppl, text] for ppl, text in zip(window_ppls, window_texts)]
                        data["window_ppls"] = wp


                        print("===========================================")
                        print(f"Construct Positive Response. Case = {index}")
                        print("===========================================")
                                    
                        pack = process_sample(
                            sample=data,
                            model=model,
                            tokenizer=tokenizer,
                            hparams=hparams,
                            seed=1234,
                            max_new_tokens=40
                        )

                        pack["general prompt"] = data["general prompt"]
                        pack["knowledge constrain"] = data["knowledge constrain"]
                        pack["question"] = data["behaviour"] if data["behaviour"] else data["question"]
                        pack["adversarial prompt"] = data["question"] if data["behaviour"] else data["adversarial_prompt"]
                        pack["pre_edit"] = data["pre_edit"]
                        
                        data = pack
                        
                        # ====================================================
                        # 【修改点】: 正确设置 Answer 和 Piece
                        # ====================================================
                        
                        # 1. 设置 Answer: 使用 process_sample 特别返回的 Window 0 生成文本
                        # (注意: process_sample 需要确保 generated_answer 字段存在)
                        data["answer"] = data["pre_edit"]["response"]

                        # 2. 添加 EOS Token (保持不变)
                        if 'qwen' in hparams.model_name.lower():
                            if not data["answer"].endswith("<|im_end|>"):
                                data["answer"] += "<|im_end|>"
                        elif 'llama' in hparams.model_name.lower():
                            if not data["answer"].endswith("<|eot_id|>"):
                                data["answer"] += "<|eot_id|>"

                        # 3. 确保 Piece 字典里只有高 PPL 的锚点，没有 0
                        # process_sample 的新逻辑已经处理了：如果 0 不在 anchors 里，piece 就不会有 0。
                        # 但为了双重保险，这里再检查一次：
                        
                        # 打印 Debug 信息确认状态
                        #print(f"[Debug] Answer (Source: Window 0): {data['answer'][:50]}...")
                        print(f"[Debug] Training Anchors (Piece Keys): {list(data['piece'].keys())}")
                        
                        # ====================================================

                        print("===========================================")
                        print(f"Start Editing. Case = {index}")
                        

                        start2 = time()

                        # —— 可选：AlphaEdit(ARE) defense 层定位（若需要）
                        if any(alg in alg_name for alg in ["AlphaEdit","AlphaEdit_ARE","AnchorEdit"]):
                            pass
                        
                        # ---------- (1) 设置第二次编辑 answer ----------
                        if 'defense_response' in data:
                            data['answer'] = data['defense_response']
                            if hparams.model_name == 'Llama3-8B-Instruct':
                                data['answer'] += '<|eot_id|>'
                            elif hparams.model_name == 'Qwen2.5-7B-Instruct':
                                data['answer'] += '<|im_end|>'

                        # ---------- (2) 执行第二次编辑 ----------
                        weights_copy = apply_algo(model, tok, hparams, [data], **ex_args, **nc_args)

                        end2 = time()
                else:
                    is_unsafe_list = jailbreak_judge.classify_responses([data['question']], [main_text_pre])
                    is_unsafe = bool(is_unsafe_list[0])
                    if pre_gen:
                        data["pre_edit"] = pre_edit

                    # =========================
                    # 第二次编辑：defense（保留你原逻辑）
                    # ========================= 
                    if is_unsafe:
                        start2 = time()
                        
                        weights_copy = apply_algo(model, tok, hparams, [data], **ex_args, **nc_args)

                        end2 = time()
                
                print("===========================================")
                print(f"Post-Editing Evaluation. Case = {index}")
                print("===========================================")

                # ---------- (3) 编辑后：主回答 生成 + PPL/fluency ----------
                q2_post_main = None
                if is_unsafe and 'AnchorEdit' in alg_name:
                    q2_post_main = tok([data['adversarial prompt']], return_tensors='pt', padding=True)
                else:
                    q2_post_main = tok([data['question']], return_tensors='pt', padding=True)
                with torch.no_grad():
                    outs2_post_main = model.generate(
                        input_ids=q2_post_main['input_ids'].to('cuda'),
                        attention_mask=q2_post_main['attention_mask'].to('cuda'),
                        do_sample=False,
                        max_new_tokens=768,
                        repetition_penalty=1.15
                    )
                gen2_ids_post_main = outs2_post_main[:, q2_post_main['input_ids'].shape[1]:]
                def_text = tok.batch_decode(gen2_ids_post_main, skip_special_tokens=True)[0]

                ppl2_post_main = _compute_ppl(model, tok, [def_text], max_length=512, batch_size=8)
                flu2_post_main = _ppl_to_fluency_local(ppl2_post_main)

                data.setdefault('defense response', {})
                data['defense response']['located layer'] = hparams.v_loss_layer
                data['defense response']['edited response'] = def_text

                # ---- (4) knowledge constrain：只保存 {text, ppl_post, ppl_delta, fluency_post, fluency_delta}
                if 'knowledge constrain' in data and 'prompt' in data['knowledge constrain']:
                    knowledge_prompt = data['knowledge constrain']['prompt']

                    # (编辑后) 生成 KC 文本
                    q2_kn_post = tok([knowledge_prompt], return_tensors='pt', padding=True)
                    device2 = next(model.parameters()).device
                    with torch.no_grad():
                        gen2_kn_post = model.generate(
                            input_ids=q2_kn_post['input_ids'].to(device=device2, dtype=torch.long),
                            attention_mask=q2_kn_post['attention_mask'].to(device=device2, dtype=torch.long),
                            do_sample=False,
                            max_new_tokens=768,
                            repetition_penalty=1.15
                        )
                    gen2_ids_kn_post = [o[len(i):] for i, o in zip(q2_kn_post['input_ids'], gen2_kn_post)]
                    kn2_text_post = tok.batch_decode(gen2_ids_kn_post, skip_special_tokens=True)[0]

                    # 后验 ppl、fluency
                    ppl2_kn_post = _compute_ppl(model, tok, [kn2_text_post], max_length=512, batch_size=8)
                    flu2_kn_post = _ppl_to_fluency_local(ppl2_kn_post)

                    # 基线：来自 pre_edit["knowledge_constrain"]
                    #pre_kc = data.get("pre_edit", {}).get("knowledge_constrain", {})
                    #pre_ppl = pre_kc.get("ppl", float('nan')) if isinstance(pre_kc, dict) else float('nan')
                    #pre_flu = pre_kc.get("fluency", float('nan')) if isinstance(pre_kc, dict) else float('nan')

                    data['defense response']['knowledge constrain'] = {
                        "text": kn2_text_post,
                        "ppl_post": ppl2_kn_post,
                        "ppl_delta": (ppl2_kn_post - pre_ppl) if (np.isfinite(ppl2_kn_post) and np.isfinite(pre_ppl)) else float('nan'),
                        #"fluency_post": flu2_kn_post,
                       # "fluency_delta": (flu2_kn_post - pre_flu) if (np.isfinite(flu2_kn_post) and np.isfinite(pre_flu)) else float('nan')
                    }


                # —— 去掉附加的特殊 token（若之前加过）
                if 'llama' in hparams.model_name.lower() and data.get('answer','').endswith('<|eot_id|>'):
                    data['answer'] = data['answer'][:-len('<|eot_id|>')]
                elif 'qwen' in hparams.model_name.lower() and data.get('answer','').endswith('<|im_end|>'):
                    data['answer'] = data['answer'][:-len('<|im_end|>')]

                # —— 可选：general prompt 在你原逻辑里（如需也可保留不动）
                device = next(model.parameters()).device
                if 'general prompt' in data:
                    general_prompts = data['general prompt']
                    general_responses = []
                    for prompt in general_prompts:
                        question = tokenizer([prompt], return_tensors='pt', padding=True)
                        with torch.no_grad():
                            generated_ids = model.generate(
                                input_ids=question['input_ids'].to(device=device, dtype=torch.long),
                                attention_mask=question['attention_mask'].to(device=device, dtype=torch.long),
                                do_sample=False,
                                max_new_tokens=768,
                                repetition_penalty=1.15
                            )
                        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)]
                        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        general_responses.append(output[0])
                    data['defense response']['general response'] = general_responses
                if is_unsafe:
                    data['time'] = end2 - start2
                else:
                    data['time'] = 0
                
                for kdrop in ["question", "answer", "anchor_ppl", "output"]:
                    if kdrop in data:
                        del data[kdrop]

                edited_data.append(data)
                path = f'output/dynamic_{alg_name}_{hparams.model_name}_sequential_{ds_name}_ckpt.json'
                # 确保id字段在保存前被保留
                if ds_name == 'wait':
                    for d_ in edited_data:
                        if 'id' not in d_:
                            d_['id'] = 'unknown_id'

                with open(path, 'w', encoding='utf-8') as json_file:
                    json.dump(edited_data, json_file, ensure_ascii=False, indent=4)

                # print("===========================================")
                # print(f"Multi-Jump Evaluation. Case = {index}")
                # print("===========================================")

                

                # if not candidates:
                #     raise ValueError("未找到满足条件（qa[-1].harmbench == True 且有 prompt）的样本。")

                
                
                # test_input_ids = tok(test_input, return_tensors="pt")
                # test_output = model.generate(
                #     input_ids=test_input_ids["input_ids"].to("cuda"),
                #     attention_mask=test_input_ids["attention_mask"].to("cuda"),
                #     do_sample=False,
                #     max_new_tokens=2048,
                #     repetition_penalty=1.15,
                #     pad_token_id=(tok.pad_token_id or tok.eos_token_id),
                # )

                # # 只取新生成的部分
                # gen_only = test_output[:, test_input_ids["input_ids"].shape[1]:]

                # # 任选其一：
                # # A: 单条 decode
                # test_res = tok.decode(gen_only[0].detach().cpu().tolist(),
                #                     skip_special_tokens=True).strip()
                
                # d["response"]["post"] = test_res

                # test_data.append(d)

                # mjpath = f"output/multi-jump_result_{alg_name}_{hparams.model_name}_layer7.json"
                # os.makedirs(os.path.dirname(mjpath), exist_ok=True)
                # with open(mjpath, "w", encoding="utf-8") as f:
                #     json.dump(test_data, f, ensure_ascii=False, indent=4)

        else:
            # 非双编辑模式的处理（保持原逻辑，可按需补齐 baseline 与 fluency 计算）
            for data in ds:
                # -------- 新增：pre_edit（非顺序模式） --------
                device = next(model.parameters()).device
                pre_edit: Dict[str, Union[str, float, List[str], Dict[str, Union[str, float]]]] = {
                    "response": "",
                    "general_response": [],
                    "knowledge_constrain": None,
                    "pre_ppl": float('nan'),
                    "pre_fluency": float('nan'),
                }
                # 主问题
                q_pre = tok([data['question']], return_tensors='pt', padding=True)
                with torch.no_grad():
                    outs_pre = model.generate(
                        input_ids=q_pre['input_ids'].to(device),
                        attention_mask=q_pre['attention_mask'].to(device),
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        max_new_tokens=768
                    )
                gen_ids_pre = outs_pre[:, q_pre['input_ids'].shape[1]:]
                main_text_pre = tok.batch_decode(gen_ids_pre, skip_special_tokens=True)[0]
                pre_edit["response"] = main_text_pre

                # knowledge constrain -> {text,ppl,fluency}
                if 'knowledge constrain' in data and isinstance(data['knowledge constrain'], dict) and 'prompt' in data['knowledge constrain']:
                    kn_prompt = data['knowledge constrain']['prompt']
                    q_kn = tok([kn_prompt], return_tensors='pt', padding=True)
                    with torch.no_grad():
                        outs_kn = model.generate(
                            input_ids=q_kn['input_ids'].to(device),
                            attention_mask=q_kn['attention_mask'].to(device),
                            do_sample=True,
                            temperature=0.1,
                            top_p=0.9,
                            max_new_tokens=768
                        )
                    gen_ids_kn = [o[len(i):] for i, o in zip(q_kn['input_ids'], outs_kn)]
                    kn_text_pre = tok.batch_decode(gen_ids_kn, skip_special_tokens=True)[0]
                    ppl_kn_pre = _compute_ppl(model, tok, [kn_text_pre], max_length=512, batch_size=8)
                    pre_edit["knowledge_constrain"] = {
                        "text": kn_text_pre,
                        "ppl": ppl_kn_pre,
                        "fluency": _ppl_to_fluency(ppl_kn_pre)
                    }
                else:
                    pre_edit["knowledge_constrain"] = None

                # general prompt
                if 'general prompt' in data and isinstance(data['general prompt'], list):
                    general_responses = []
                    for prompt in data['general prompt']:
                        q_gp = tok([prompt], return_tensors='pt', padding=True)
                        with torch.no_grad():
                            outs_gp = model.generate(
                                input_ids=q_gp['input_ids'].to(device),
                                attention_mask=q_gp['attention_mask'].to(device),
                                do_sample=True,
                                temperature=0.1,
                                top_p=0.9,
                                max_new_tokens=768
                            )
                        gen_ids_gp = [o[len(i):] for i, o in zip(q_gp['input_ids'], outs_gp)]
                        out_gp = tok.batch_decode(gen_ids_gp, skip_special_tokens=True)[0]
                        general_responses.append(out_gp)
                    pre_edit["general_response"] = general_responses

                # 指标（基于主问题 response）
                ppl_pre_main = _compute_ppl(model, tok, [pre_edit["response"]], max_length=512, batch_size=8)
                pre_edit["pre_ppl"] = ppl_pre_main
                pre_edit["pre_fluency"] = _ppl_to_fluency(ppl_pre_main)

                data["pre_edit"] = pre_edit
                # ------------------------------------------

                if ds_name in ['unke','cf']:
                    question = tokenizer([data['question'],data['para_question']], return_tensors='pt', padding=True)
                else:
                    question = tokenizer([data['question']], return_tensors='pt', padding=True)

                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=question['input_ids'].to('cuda'),
                        attention_mask=question['attention_mask'].to('cuda'),
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        max_new_tokens=768
                    )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                if ds_name in ['unke','cf']:
                    data['para_prediction'] = output[1]

            if 'general_predictions' not in data:
                data['general_predictions'] = []
            if 'knowledge_prediction' not in data:
                data['knowledge_prediction'] = ""

            if hparams.model_name == 'Llama3-8B-Instruct':
                data['answer'] = data['answer'][:-len('<|eot_id|>')]
            elif hparams.model_name == 'Qwen2.5-7B-Instruct':
                data['answer'] = data['answer'][:-len('<|im_end|>')]

        if ds_name in ['unke','cf','mquake']:
            for data in ds:
                question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=question['input_ids'].to('cuda'),
                        attention_mask=question['attention_mask'].to('cuda'),
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,# Analysis exp
                        max_new_tokens=768
                    )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]

                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                data['sub_pred'] = output

        # 恢复原始权重，支持双编辑模式
        if double_edit and ds_name == 'wait' and 'weights_copy_defense' in locals():
            # 先恢复到第一次编辑后的权重
            for k, v in weights_copy_defense.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
            # 再恢复到原始权重
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
        elif 'weights_copy' in locals():
            # 单编辑模式下直接恢复
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

        # 确保数据结构完整性并处理PPL相关参数（保持你的原逻辑）
        for data in ds:
            defense_response_dict = data.get('defense response', {})
            recovery_response_dict = data.get('recovery response', {})

            if 'initial_output' in data and 'edited response' in defense_response_dict:
                defense_response_dict['ppl_pre'] = data.get('ppl_pre_defense', float('nan'))
                defense_response_dict['ppl_post'] = data.get('ppl_post_defense', float('nan'))
                defense_response_dict['delta_ppl'] = data.get('delta_ppl_defense', float('nan'))

            if 'initial_output' in data and 'edited response' in recovery_response_dict:
                recovery_response_dict['ppl_pre'] = data.get('ppl_pre_defense', float('nan'))
                recovery_response_dict['ppl_post'] = data.get('ppl_post_defense', float('nan'))
                recovery_response_dict['delta_ppl'] = data.get('delta_ppl_defense', float('nan'))

            data['defense response results'] = defense_response_dict
            data['recovery response results'] = recovery_response_dict

            fields_to_remove = ['recovery_prediction', 'ppl_pre', 'ppl_post', 'delta_ppl', 'general_predictions', 'knowledge_prediction', 'defense_prediction', 'defense_para_prediction', 'recovery_para_prediction', 'ppl_pre_defense', 'ppl_post_defense', 'delta_ppl_defense', 'defense response', 'recovery response']
            for field in fields_to_remove:
                if field in data:
                    del data[field]

        edited_data.extend(ds)
    if sequential:
        path = f'output/dynamic_{alg_name}_{hparams.model_name}_sequential_{ds_name}.json'
    else:
        path = f'output/{alg_name}_{hparams.model_name}_{ds_name}_result.json'
    # 确保id字段在保存前被保留
    if ds_name == 'wait':
        for data in edited_data:
            if 'id' not in data:
                data['id'] = 'unknown_id'

    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(edited_data, json_file, ensure_ascii=False, indent=4)

    print(f"saving to {path}")


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
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","AlphaEdit_ARE", "MEMIT","MEMIT_ARE", "MEMIT_AnchorEdit", "ROME", "FT", "MEND","unke","unke_ARE","AnchorEdit","unke_AnchorEdit"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="Llama3-8B-Instruct",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama3-8B-Instruct.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "editevery", "unke","mquake","valueinject","wait","safety_window","safeedit"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--sequential",
        dest="sequential",
        action="store_true",
        help="sequential editing",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="指定要使用的GPU设备ID，默认为0。可以指定多个GPU，例如：'0,1'",
    )
    parser.add_argument(
        "--double_edit",
        action="store_true",
        default=False,
        help="对于WAIT数据集，启用双编辑模式：先使用defense_response编辑，再使用recovery_response编辑",
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        sequential=args.sequential,
        gpu_id=args.gpu_id,
        double_edit=args.double_edit,
    )

