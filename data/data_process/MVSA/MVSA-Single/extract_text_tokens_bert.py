import torch 
import transformers
import numpy as np
import os
import json
from tqdm import tqdm
import shutil
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def process_batch(args,max_length=197):
    """
    处理一个批次的样本
    """
    batch_data, datasub, json_dir, text_target_dir, img_target_dir, tokenizer = args
    
    # 批量处理文本
    batch_texts = [item["text"] for item in batch_data]
    encoded_batch = tokenizer.batch_encode_plus(
        batch_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=False  # 单句子任务不需要
    )
    
    # 创建必要的目录（每个批次只创建一次）
    token_dir = os.path.join(text_target_dir, f"{datasub}_token")
    img_dir = os.path.join(img_target_dir, f"{datasub}_imgs")
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    # print(f"input_id shape is {encoded_batch['input_ids'][0].shape}")
    # print(f"attention_mask shape is {encoded_batch['attention_mask'][0].shape}")
    results = []
    for j, item in enumerate(batch_data):
        # 处理文本token
        tokenized_caption = encoded_batch["input_ids"][j]
        padding_mask = encoded_batch["attention_mask"][j]
        padding_mask = padding_mask
        # print(f"tokenized_caption shape is {tokenized_caption.shape}") 
        # print(f"padding_mask  is {padding_mask.shape}") 
        # 处理图片路径
        img_path = item["img"].replace("\\", "/")
        img_filename = os.path.basename(img_path)
        spec_name = img_filename.split(".jpg")[0]
        
        # 构建保存路径
        token_save_path = os.path.join(token_dir, f"{spec_name}_token.npy")
        pm_save_path = os.path.join(token_dir, f"{spec_name}_pm.npy")
        
        # 源图片和目标图片路径
        img_source = os.path.join(json_dir, img_path)
        img_target = os.path.join(img_dir, img_filename)
        
        # 保存token和mask
        np.save(token_save_path, tokenized_caption.astype(np.int64))
        np.save(pm_save_path, padding_mask.astype(np.float32))
        
        # 复制图片
        shutil.copy2(img_source, img_target)
        
        results.append(spec_name)
    
    return results

if __name__ == "__main__":
    # 初始化tokenizer
    tokenizer = transformers.BertTokenizer.from_pretrained("/root/autodl-tmp/AME/model/bert-base-uncased")
    
    # 设置路径
    # data_set = 'food101'
    # data_save_dir = "/root/autodl-tmp/AVDataModel/data/dataset/Food101/food101"
    data_save_dir = "/root/autodl-tmp/AME/data/dataset/MVSA/MVSA_Single"
    # data_dir = os.path.join(data_save_dir,"data")
    text_target_dir = os.path.join(data_save_dir, "text_token")
    img_target_dir = os.path.join(data_save_dir, "visual")
    
    # 确保目标目录存在
    os.makedirs(text_target_dir, exist_ok=True)
    os.makedirs(img_target_dir, exist_ok=True)
    
    # 处理所有JSONL文件
    all_jsonls = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    # 获取CPU核心数用于并行处理
    # num_workers = multiprocessing.cpu_count()
    num_workers = 1
    
    for filename in all_jsonls:
        json_path = os.path.join(data_save_dir, filename)
        
        # 读取数据
        data = []
        with open(json_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"{filename} has {len(data)} files")
        datasub = filename.split(".jsonl")[0]
        
        # 设置批量大小
        batch_size = 64  # 根据内存调整
        num_batches = (len(data) + batch_size - 1) // batch_size
        
        # 准备批次数据
        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batches.append((batch_data, datasub, data_save_dir, text_target_dir, img_target_dir, tokenizer))
        
        # 使用多进程并行处理批次
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 使用tqdm显示进度
            results = list(tqdm(executor.map(process_batch, batches), 
                                total=num_batches,
                                desc=f"Processing {datasub} batches",
                                unit="batch"))
        
        # 统计处理结果
        total_processed = sum(len(batch_result) for batch_result in results)
        print(f"Processed {total_processed} samples for {datasub}")
    
    print("Done!")