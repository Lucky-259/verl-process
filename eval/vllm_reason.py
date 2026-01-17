import sys
import json
import argparse
import threading
from openai import OpenAI, APIConnectionError, RateLimitError, APIError
from concurrent.futures import ThreadPoolExecutor, as_completed
from deepscaler.rewards.math_reward import deepscaler_reward_fn
import traceback
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import time
import random

suffix_1 = (
    " Let's think step by step and output the final answer within \\boxed{}."
)
suffix_2 = (
    "\nLet's reason step by step. Enclose the reasoning process within <think>...</think>, then summarize it and present the final answer within \\boxed{} — for example: <think>reasoning process here</think> \\boxed{answer here}."
)
suffix_3 = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}." # DeepSeek
)

def request_model(client, prompt, seed, model, tokenizer=None):
    """
    封装一次推理请求，内置重试逻辑。
    如果遇到连接错误或限流，会进行指数退避重试。
    """
    max_retries = 20
    base_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                extra_body={
                    "add_generation_prompt": True,
                    "seed": seed,
                },
                temperature=0.6,
                top_p=0.95,
                timeout=300, # 单次请求超时时间
            )
            
            # 请求成功，处理结果
            model_response = response.choices[0].message.content.strip()
            
            # === Token 计数逻辑 ===
            completion_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                completion_tokens = response.usage.completion_tokens
            
            if completion_tokens == 0 and tokenizer is not None:
                try:
                    completion_tokens = len(tokenizer.encode(model_response))
                except Exception:
                    completion_tokens = 0

            return model_response, completion_tokens

        except (APIConnectionError, RateLimitError, APIError) as e:
            # 遇到网络层面的错误，打印日志并等待重试
            # 如果是最后一次尝试，则抛出异常
            if attempt == max_retries - 1:
                print(f"[Error] Max retries reached. Last error: {e}")
                raise e
            
            # 指数退避 + 随机抖动
            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
            # 限制最大等待时间为 60 秒
            sleep_time = min(sleep_time, 60)
            
            # print(f"[Warning] Connection failed ({e}), retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_time)
            
        except Exception as e:
            # 其他未知错误（如参数错误），直接抛出不重试
            raise e

    return "", 0

def main(seed):
    parser = argparse.ArgumentParser(description="vLLM AIME 推断并统计 pass@1 与 pass@16 (支持断点续传 & 平均长度)")
    parser.add_argument('--model', type=str, required=True, help="vLLM 模型名称或路径")
    parser.add_argument('--file', type=str, required=True, help="测试集 JSON 文件路径")
    parser.add_argument('--ports', type=str, required=True, help="vLLM server 端口号列表")
    parser.add_argument('--repeat', type=int, required=True, help="每个问题重复的次数")
    parser.add_argument('--concurrency', type=int, required=True, help="每个端口最大并发请求数")
    parser.add_argument('--output_dir', type=str, required=True, help="输出文件夹")
    args = parser.parse_args()
    suffix = suffix_3

    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 定义中间结果保存路径 (JSONL 格式)
    cache_file = os.path.join(args.output_dir, "intermediate_results.jsonl")

    # === 加载 Tokenizer (用于 Fallback) ===
    print(f"Loading tokenizer from {args.model} for fallback counting...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}. Token counts will be 0 if API fails.")
        tokenizer = None

    # === 初始化 OpenAI 客户端 ===
    ports = args.ports.split(',')
    client_list = []
    for port in ports:
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        sem = threading.Semaphore(args.concurrency)
        client_list.append((client, sem))
    total_clients = len(client_list)
    total_concurrency = total_clients * args.concurrency
    print(f"共构造 {total_clients} 个客户端，总并发数为 {total_concurrency}")

    # === 读取数据 ===
    with open(args.file, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    if not original_data:
        print("测试集为空。")
        return

    # -------------------------------------------------------------------------
    # 1. 初始化结果容器
    # -------------------------------------------------------------------------
    detailed_results = {}
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        difficulty = item.get("difficulty", "")
        detailed_results[q_idx] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "difficulty": difficulty,
            "responses": [None] * args.repeat,
            "token_counts": [0] * args.repeat, # 存储长度
            "correct_flags": [False] * args.repeat
        }

    # -------------------------------------------------------------------------
    # 2. 加载断点 (Checkpoint Loading)
    # -------------------------------------------------------------------------
    finished_tasks = set() # 存储已完成的 (q_idx, rep_index)
    total_token_count = 0  # 全局 Token 总数累加器
    
    if os.path.exists(cache_file):
        print(f"发现中间结果文件 {cache_file}，正在加载断点...")
        loaded_count = 0
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: continue
                    record = json.loads(line)
                    q_idx = record["q_idx"]
                    rep = record["rep_idx"]
                    resp = record["response"]
                    is_corr = record["is_correct"]
                    t_len = record.get("token_len", 0) 
                    
                    if q_idx in detailed_results and rep < args.repeat:
                        detailed_results[q_idx]["responses"][rep] = resp
                        detailed_results[q_idx]["correct_flags"][rep] = is_corr
                        detailed_results[q_idx]["token_counts"][rep] = t_len
                        
                        finished_tasks.add((q_idx, rep))
                        total_token_count += t_len
                        loaded_count += 1
                except Exception as e:
                    pass # 忽略坏行
        print(f"成功加载 {loaded_count} 条已完成的推理记录。当前总 Token 数: {total_token_count}")

    # -------------------------------------------------------------------------
    # 3. 构造任务
    # -------------------------------------------------------------------------
    tasks = []
    for q_idx, item in enumerate(original_data):
        prompt = detailed_results[q_idx]["prompt"]
        ground_truth = detailed_results[q_idx]["ground_truth"]
        
        for rep in range(args.repeat):
            if (q_idx, rep) in finished_tasks:
                continue
            tasks.append((q_idx, prompt, ground_truth, rep))

    print(f"剩余需推理任务数: {len(tasks)}")

    # -------------------------------------------------------------------------
    # 4. 执行推理并实时保存
    # -------------------------------------------------------------------------
    executor = ThreadPoolExecutor(max_workers=total_concurrency)
    global_index = 0

    def request_task(client, semaphore, prompt, seed, model, tokenizer):
        # 这里的 semaphore 限制的是并发数，但内部的 request_model 会负责重试
        with semaphore:
            return request_model(client, prompt, seed, model, tokenizer)

    future_to_meta = {}
    
    for task in tqdm(tasks, desc="提交剩余任务", total=len(tasks)):
        q_idx, prompt, ground_truth, rep = task
        client, sem = client_list[global_index % total_clients]
        global_index += 1
        
        fut = executor.submit(request_task, client, sem, prompt, seed, args.model, tokenizer)
        future_to_meta[fut] = (q_idx, rep, ground_truth)

    pbar = tqdm(total=len(tasks), desc="Processing & Saving", ncols=100)
    
    with open(cache_file, "a", encoding="utf-8") as f_cache:
        for fut in as_completed(future_to_meta):
            q_idx, rep, ground_truth = future_to_meta[fut]
            
            model_response = ""
            token_len = 0
            
            # 获取结果
            # 注意：因为重试逻辑已经在 request_model 里了，
            # 只要 Future 抛出异常，说明重试了多次依然失败，或者出现了不可恢复的错误。
            try:
                model_response, token_len = fut.result()
            except Exception as e:
                error_message = str(e) # 简化报错信息，避免存入过长 Traceback
                print(f"Task (q={q_idx}, r={rep}) FATAL ERROR after retries: {error_message}")
                model_response = f"Error: {error_message}"
                token_len = 0

            # 计算奖励
            try:
                # 如果 model_response 是错误信息，这里自然会 False
                is_correct = deepscaler_reward_fn(model_response, ground_truth)
            except Exception:
                is_correct = False

            # 实时保存
            record = {
                "q_idx": q_idx,
                "rep_idx": rep,
                "response": model_response,
                "is_correct": is_correct,
                "token_len": token_len
            }
            f_cache.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_cache.flush()

            # 更新内存
            detailed_results[q_idx]["responses"][rep] = model_response
            detailed_results[q_idx]["correct_flags"][rep] = is_correct
            detailed_results[q_idx]["token_counts"][rep] = token_len
            total_token_count += token_len
            
            pbar.update(1)
    
    pbar.close()
    executor.shutdown(wait=True)

    # -------------------------------------------------------------------------
    # 5. 统计与最终保存
    # -------------------------------------------------------------------------
    total_responses_count = len(original_data) * args.repeat
    correct_count = 0
    pass16_correct = 0

    for q_idx, result in detailed_results.items():
        correct_count += sum(result["correct_flags"])
        question_correct = any(result["correct_flags"])
        result["is_correct"] = question_correct
        result["question_accuracy"] = sum(result["correct_flags"]) / args.repeat
        
        valid_lens = [l for l in result["token_counts"] if l > 0]
        result["avg_tokens"] = sum(valid_lens) / len(valid_lens) if valid_lens else 0
        
        if question_correct:
            pass16_correct += 1

    pass_at_1 = correct_count / total_responses_count
    pass_at_16 = pass16_correct / len(original_data)
    average_token_len = total_token_count / total_responses_count if total_responses_count > 0 else 0

    summary_results = {
        "file": args.file,
        "model": args.model,
        "pass@1": pass_at_1,
        f"pass@{args.repeat}": pass_at_16,
        "average_token_len": average_token_len,
        "total_tokens": total_token_count,
        "total_responses": total_responses_count,
        "total_questions": len(original_data)
    }
    
    with open(os.path.join(args.output_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    detailed_results_list = []
    for q_idx in sorted(detailed_results.keys()):
        detailed_results_list.append(detailed_results[q_idx])
    
    details_file_path = os.path.join(args.output_dir, "results_details.json")
    with open(details_file_path, "w", encoding="utf-8") as f:
        json.dump(detailed_results_list, f, ensure_ascii=False, indent=2)

    print("\n######### Deepscaler 统计结果:")
    print(f"pass@1 = {pass_at_1:.4f}, pass@{args.repeat} = {pass_at_16:.4f}")
    print(f"Avg Tokens = {average_token_len:.2f}, Total Tokens = {total_token_count}")
    print(f"统计信息已保存至 {args.output_dir}/results_summary.json")

    # -------------------------------------------------------------------------
    # 6. 自动清理逻辑
    # -------------------------------------------------------------------------
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"【清理】中间文件已删除: {cache_file}")
        except Exception:
            pass

if __name__ == "__main__":
    main(None)