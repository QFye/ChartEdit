import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 尝试导入 JailbreakJudge
# ==========================================
try:
    # 根据你的项目结构调整，假设在 classifiers 文件夹下
    from classifiers import LlamaClassifier
except ImportError:
    try:
        # 或者在 util 下
        from util import LlamaClassifier
    except ImportError:
        print("[Error] 无法导入 LlamaClassifier。请检查 sys.path 或路径设置。")
        sys.exit(1)

def main(gpu_id):
    # ---------------------------------------------------------
    # 0. 设置 GPU
    # ---------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Using GPU: {gpu_id}")
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Using CPU (this will be slow).")
        device = "cpu"
    else:
        device = "cuda"

    # ---------------------------------------------------------
    # 1. 配置路径与参数
    # ---------------------------------------------------------
    model_path = "/mnt/ssd2/models/llama3.1-8b-instruct"
    input_file = "/home/ubuntu/ye/AnyEdit/data/SafeEdit_test.json"
    output_file = "/home/ubuntu/ye/AnyEdit/data/SafeEdit_test_llama.json"
    
    # 显存优化：Batch Size (Llama-3.1 8B 在 24G 显存上通常可设 8-16)
    BATCH_SIZE = 8
    
    # ---------------------------------------------------------
    # 2. 加载模型与分词器
    # ---------------------------------------------------------
    print(f"Loading Model from {model_path} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            local_files_only=True,
            torch_dtype=torch.bfloat16 # 建议使用 bf16 以节省显存并匹配 Llama3 原生精度
        ).to(device)
        
        tok = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}")
        return

    # Llama 3 必须设置 pad_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        
    model.eval()

    # ---------------------------------------------------------
    # 3. 初始化判别器
    # ---------------------------------------------------------
    print("Initializing JailbreakJudge (LlamaClassifier)...")
    try:
        # JailbreakJudge 内部也会加载模型，它通常会自动使用 device
        # 如果它内部写死了 "cuda"，设置了 CUDA_VISIBLE_DEVICES 后也是安全的
        jailbreak_judge = LlamaClassifier()
    except Exception as e:
        print(f"[Error] JailbreakJudge 初始化失败: {e}")
        return

    # ---------------------------------------------------------
    # 4. 加载数据
    # ---------------------------------------------------------
    if not os.path.exists(input_file):
        print(f"[Error] 输入文件不存在: {input_file}")
        return

    print(f"Loading data from {input_file} ...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Total samples: {len(raw_data)}")
    
    unsafe_data_list = []
    
    # ---------------------------------------------------------
    # 5. 批量生成与判别
    # ---------------------------------------------------------
    all_questions = [item.get('adversarial prompt', '') for item in raw_data]
    num_batches = (len(all_questions) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("Start generating and judging...")
    
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(all_questions))
        
        batch_questions = all_questions[start_idx:end_idx]
        batch_items = raw_data[start_idx:end_idx]
        
        # 5.1 构造输入 (无 Template，直接生成)
        # 直接使用原始 question 列表
        batch_inputs_text = batch_questions 
            
        # 5.2 Tokenize
        # 注意：不加 add_special_tokens=True 可能更好，或者视模型而定。
        # Llama-3 通常建议加 BOS。
        inputs = tok(
            batch_inputs_text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=768,
            add_special_tokens=False # 加上 BOS
        ).to(device)
        
        # 5.3 模型生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False, 
                pad_token_id=tok.pad_token_id
            )
            
        # 5.4 解码
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]
        batch_responses = tok.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # 5.5 判别 (Judge)
        judge_results = jailbreak_judge.classify_responses(batch_questions, batch_responses)
        
        # 5.6 筛选保存
        for j, is_unsafe_val in enumerate(judge_results):
            if bool(is_unsafe_val):
                item_to_save = batch_items[j].copy()
                
                # 【新增】保存模型生成的 Output
                generated_response = batch_responses[j]
                
                # 方式 A：直接存为顶层字段 (简单明了)
                item_to_save['unsafe_response_llama3'] = generated_response
                
                
                # 如果你也想覆盖顶层 answer 字段 (视你的 main.py 读取逻辑而定)
                # item_to_save['answer'] = generated_response 
                
                unsafe_data_list.append(item_to_save)

    # ---------------------------------------------------------
    # 6. 保存结果
    # ---------------------------------------------------------
    print(f"\nProcessing complete.")
    print(f"Original size: {len(raw_data)}")
    print(f"Unsafe samples found: {len(unsafe_data_list)}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unsafe_data_list, f, ensure_ascii=False, indent=4)
        
    print(f"Filtered dataset saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter unsafe samples using Llama-3 and JailbreakJudge")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use (e.g., '0' or '0,1')")
    args = parser.parse_args()
    
    main(args.gpu_id)
