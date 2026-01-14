import sys
import json
import argparse
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from deepscaler.rewards.math_reward import deepscaler_reward_fn
import traceback
from tqdm import tqdm
from transformers import AutoTokenizer

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
    """封装一次推理请求，便于在线程池中调用"""
    def single_request():
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
            timeout=100000000,
        )
        return response
    
    response = single_request()
    model_response = response.choices[0].message.content.strip()
    
    # === Token 计数逻辑 ===
    completion_tokens = 0
    # 1. 尝试从 API 响应中获取 usage
    if hasattr(response, 'usage') and response.usage:
        completion_tokens = response.usage.completion_tokens
    
    # 2. 如果 API 获取失败，且有 tokenizer，则本地计算
    if completion_tokens == 0 and tokenizer is not None:
        try:
            # 仅对输出进行 encode 计数
            completion_tokens = len(tokenizer.encode(model_response))
        except Exception as e:
            print(f"Token counting failed: {e}")
            completion_tokens = 0

    return model_response, completion_tokens

def main(seed):
    parser = argparse.ArgumentParser(description="vLLM AIME 推断并统计 pass@1 与 pass@16")
    parser.add_argument('--model', type=str, required=True, help="vLLM 模型名称或路径")
    parser.add_argument('--file', type=str, required=True, help="测试集 JSON 文件路径")
    parser.add_argument('--ports', type=str, required=True, help="vLLM server 端口号列表")
    parser.add_argument('--repeat', type=int, required=True, help="每个问题重复的次数")
    parser.add_argument('--concurrency', type=int, required=True, help="每个端口最大并发请求数")
    parser.add_argument('--output_dir', type=str, required=True, help="输出文件夹")
    args = parser.parse_args()
    suffix = suffix_3

    # === 加载 Tokenizer (用于 Fallback) ===
    print(f"Loading tokenizer from {args.model} for fallback counting...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load tokenizer: {e}. Token counts will be 0 if API fails.")
        tokenizer = None

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

    with open(args.file, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    if not original_data:
        print("测试集为空。")
        return

    tasks = []   
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        for rep in range(args.repeat):
            tasks.append((q_idx, prompt, ground_truth, rep))

    detailed_results = {}
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        detailed_results[q_idx] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "responses": [None] * args.repeat,
            "token_counts": [0] * args.repeat, # 新增：记录每条回复的长度
            "correct_flags": [False] * args.repeat
        }

    total_responses = len(tasks)
    correct_count = 0
    total_token_count = 0 # 全局总 token 数

    executor = ThreadPoolExecutor(max_workers=total_concurrency)
    future_list = []
    global_index = 0

    def request_task(client, semaphore, prompt, seed, model, tokenizer):
        with semaphore:
            return request_model(client, prompt, seed, model, tokenizer)

    for task in tqdm(tasks, desc="提交推理任务", total=len(tasks)):
        q_idx, prompt, ground_truth, rep = task
        client, sem = client_list[global_index % total_clients]
        global_index += 1
        fut = executor.submit(request_task, client, sem, prompt, seed, args.model, tokenizer)
        future_list.append((q_idx, rep, prompt, ground_truth, fut))

    for q_idx, rep, prompt, ground_truth, fut in tqdm(future_list, desc="Processing responses", ncols=100):
        try_count = 0
        model_response = ""
        token_len = 0
        
        while try_count <= 10:
            try:
                # 获取 (response, tokens)
                model_response, token_len = fut.result()
                break
            except Exception as e:
                try_count += 1
                error_message = traceback.format_exc()
                print("Error, retrying", error_message, flush=True)
                model_response = f"Error, {error_message}"
                token_len = 0

        try:
            is_correct = deepscaler_reward_fn(model_response, ground_truth)
        except Exception as e:
            is_correct = False

        # 更新详细结果
        detailed_results[q_idx]["responses"][rep] = model_response
        detailed_results[q_idx]["token_counts"][rep] = token_len
        detailed_results[q_idx]["correct_flags"][rep] = is_correct

        if is_correct:
            correct_count += 1
        
        total_token_count += token_len

    pass_at_1 = correct_count / total_responses

    pass16_correct = 0
    for q_idx, result in detailed_results.items():
        question_correct = any(result["correct_flags"])
        result["is_correct"] = question_correct
        result["question_accuracy"] = sum(result["correct_flags"]) / args.repeat
        # 记录该问题的平均输出长度
        valid_lens = [l for l in result["token_counts"] if l > 0]
        result["avg_tokens"] = sum(valid_lens) / len(valid_lens) if valid_lens else 0
        
        if question_correct:
            pass16_correct += 1
    
    pass_at_16 = pass16_correct / len(original_data)
    
    # 计算全局平均长度
    average_token_len = total_token_count / total_responses if total_responses > 0 else 0

    # Summary
    summary_results = {
        "file": args.file,
        "model": args.model,
        "pass@1": pass_at_1,
        f"pass@{args.repeat}": pass_at_16,
        "average_token_len": average_token_len, # 平均长度
        "total_tokens": total_token_count,      # 总长度
        "total_responses": total_responses,
        "total_questions": len(original_data)
    }
    
    with open(args.output_dir + "/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    detailed_results_list = []
    for q_idx in sorted(detailed_results.keys()):
        detailed_results_list.append(detailed_results[q_idx])
    with open(args.output_dir + "/results_details.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results_list, f, ensure_ascii=False, indent=2)

    print("######### Deepscaler 统计结果:")
    print(f"pass@1 = {pass_at_1:.4f}, pass@{args.repeat} = {pass_at_16:.4f}")
    print(f"Avg Tokens = {average_token_len:.2f}, Total Tokens = {total_token_count}")
    print(f"统计信息已保存至 {args.output_dir}")

if __name__ == "__main__":
    main(None)