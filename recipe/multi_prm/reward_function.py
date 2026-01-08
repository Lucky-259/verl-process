# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import re
import time
from typing import Any
from openai import OpenAI
from verl.utils.reward_score import math_verify
#from tenacity import retry, wait_exponential, stop_after_attempt

# -----------------------
# Configuration
# -----------------------
# vLLM API 配置
VLLM_API_KEY = "EMPTY"
VLLM_API_BASE = "http://localhost:8000/v1"  # 修改为你的 vLLM 服务地址

# API 调用配置
MAX_RETRIES = 5
BASE_DELAY = 2
MAX_WORKERS = 8
TIMEOUT = 5000

# Regex for parsing step scores
# STEP_LINE_PATTERN = re.compile(
#     r'<step_(\d+)>.*?boxed{([01])}.*?</step_\1>',
#     re.IGNORECASE
# )

STEP_BLOCK_PATTERN = re.compile(
    r'<step_(\d+)>\s*.*?\\?boxed\{(CORRECT|INCORRECT)\}.*?</step_\1>',
    re.IGNORECASE | re.DOTALL
)

ANSWER_VERIFICATION_PATTERN = re.compile(
    r"<answer_verification>\s*.*?\\?boxed\{(CORRECT|INCORRECT)\}.*?</answer_verification>",
    re.IGNORECASE | re.DOTALL,
)

# 生成式 PRM 提示词模板
PROCESS_SCORE_PROMPT = """\
You are a verification assistant specialized in mathematical reasoning. Your task is to carefully evaluate the CORRECTNESS of the Assistant's solution, which contains thinking steps and final answer.

Definition of step correctness:
- A step is CORRECT if all its key mathematical/logical claims are correct and supported by the problem context.
- If any key claim is incorrect, unsupported, or contradicts the previous context, the step is INCORRECT.


You will be given the original problem and the Assistant's solution, which contains {generator_step_count} <step> tags:

Problem: {problem}
Assistant's Solution:
{thinking}


IMPORTANT:
- You MUST produce exactly {generator_step_count} step evaluations, in order.
- Each <step> contains a brief verification in YOUR OWN words - Do NOT copy the original step text.
- Each <step> must end with EXACTLY ONE label: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}
Your output MUST follow this exact format:

<step_1>Step 1 Analysis: Your detailed verification reasoning goes here. Conclude with only one judgement: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}</step_1>
<step_2>Step 2 Analysis: Your detailed verification reasoning goes here. Conclude with only one judgement: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}</step_2>
... [CONTINUE for ALL {generator_step_count} <step> blocks in the Assistant's Solution] ...
<answer_verification>Answer Analysis: Verify whether the content inside <answer>...</answer> is correct. Conclude with only one judgement: \\boxed{{CORRECT}} or \\boxed{{INCORRECT}}</answer_verification>

Your Verification:
"""

# ============================================================================
# vLLM API 调用函数
# ============================================================================

def call_generative_prm_api(client, messages, vllm_api_base=None) -> str:
    """
    调用生成式 PRM API 获取验证结果
    
    Args:
        question: 问题文本
        tagged_solution: 带 <step> 标签的解答
        step_count: 步骤数量
        vllm_api_base: vLLM API 基础 URL
    
    Returns:
        verification_response: PRM 生成的验证文本
    """
    if vllm_api_base is None:
        vllm_api_base = VLLM_API_BASE
    
    for attempt in range(MAX_RETRIES):
        try:
            # 获取模型列表
            models = client.models.list()
            model = models.data[0].id
            
            # 调用 Chat Completion API
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            
            # 提取响应文本
            # reasoning_content = chat_completion.choices[0].message.reasoning
            verification_response = chat_completion.choices[0].message.content
            return verification_response
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Generative PRM API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                delay = BASE_DELAY * (2 ** attempt)
                sleep(delay)
            else:
                print(f"Generative PRM API call failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None

def compute_outcome_reward(response: str, ground_truth: Any) -> float:
    """
    Compute the outcome reward based on the final answer correctness.
    
    Extracts the boxed result and checks if it equals the ground truth.
    
    Args:
        response: Model response string
        
    Returns:
        1.0 if correct, 0.0 otherwise
    """
    try:
        res = math_verify.compute_score(response, ground_truth)
        return float(res)
        
    except Exception as e:
        print(f"[Outcome Reward] Error: {e}")
        return 0.0


def parse_step_scores(text: str, step_count: int) -> list[float]:
    scores = [0.0] * step_count
    seen = [False] * step_count

    if not text:
        print("[Parse Scores] Empty response text")
        return scores

    parsed_count = 0
    for m in STEP_BLOCK_PATTERN.finditer(text):
        step_idx = int(m.group(1)) - 1
        label = m.group(2).upper()

        if 0 <= step_idx < step_count:
            scores[step_idx] = 1.0 if label == "CORRECT" else 0.0
            seen[step_idx] = True
            parsed_count += 1

    print(f"[Parse Scores] Parsed {parsed_count}/{step_count} step labels")

    missing_indices = [i for i, ok in enumerate(seen) if not ok]
    if missing_indices and parsed_count > 0:
        print(f"[Parse Scores] Warning: Missing step labels for steps: {missing_indices}")

    return scores

def parse_answer_verification(text: str) -> int | None:
    """Parse answer judgement from <answer_verification> block.

    Returns:
        1 for CORRECT, 0 for INCORRECT, None if missing/unparseable.
    """
    if not text:
        return None
    m = ANSWER_VERIFICATION_PATTERN.search(text)
    if not m:
        return None
    return 1 if m.group(1).upper() == "CORRECT" else 0


def compute_process_scores(
    problem: str, 
    tagged_solution: str, 
    step_count: int,
    ground_truth: Any
) -> list[float]:
    """
    Compute process-level scores for each step using generative reward model.
    
    Args:
        problem: The mathematical problem text
        tagged_solution: Solution with <step_i> tags
        step_count: Number of steps to score
        
    Returns:
        List of step scores (length = step_count)
    """
    if step_count <= 0:
        print("[Process Scores] step_count <= 0, returning empty list")
        return ([], None)
    
    print(f"[Process Scores] Computing scores for {step_count} steps")
    
    try:
        # Build prompt
        prompt = PROCESS_SCORE_PROMPT.format(
            generator_step_count=step_count,
            problem=problem, 
            thinking=tagged_solution
        )
        
        # Call API
        messages = [
            {"role": "user", "content": prompt}
        ]
        client = OpenAI(
            api_key=api_key,                       
            base_url=base_url,
        )
        response_text = _post_chat_completion(client, messages)
        print(f"[Verfication] {response_text}")

        ###################################################################
        data = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "verification": response_text
        }
        with open("prompt_verfication.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        ###################################################################
        
        # Parse scores
        step_scores = parse_step_scores(response_text, step_count)
        answer_verification = parse_answer_verification(response_text)
        
        print(f"[Process Scores] Final scores: {step_scores}")
        print(f"[Process Scores] Answer verification: {answer_verification}")
        return step_scores, answer_verification
        
    except Exception as e:
        print(f"[Process Scores] Error computing scores: {e}")
        # Return zero scores on error
        return ([], None)


def compute_score(
    data_source: str, 
    solution_str: str, 
    ground_truth: Any, 
    extra_info: dict
) -> dict[str, Any]:
    """
    Main entry point for computing both outcome and process rewards.
    
    This function is called by ProcessNaiveRewardManager for each sample.
    
    Args:
        data_source: Identifier for the data source
        solution_str: Model's solution string
        ground_truth: Ground truth answer (may be None)
        extra_info: Dict containing:
            - tagged_solution: Solution with step tags
            - step_count: Number of identified steps
            - question: The problem text (optional)
            
    Returns:
        Dict with keys:
            - score: Final outcome reward (float)
            - step_scores: List of process rewards (list[float])
    """
    print(f"\n[Compute Score] Data source: {data_source}")
    
    # Compute outcome reward
    outcome_score = compute_outcome_reward(solution_str, ground_truth)
    answer_verification = None
    process_used = False
    # Extract parameters for process scoring
    tagged_solution = extra_info.get("tagged_solution", solution_str)
    step_count = int(extra_info.get("step_count", 0))
    problem = extra_info.get("question", "")
    
    # Compute process rewards if steps exist
    if step_count > 0 and problem:
        print(f"[Compute Score] Computing process scores for {step_count} steps")
        step_scores, answer_verification = compute_process_scores(problem, tagged_solution, step_count, ground_truth)
        # Gate process rewards: only use step_scores if verifier's answer judgement matches outcome_score
        process_used = (answer_verification is not None) and (int(outcome_score) == int(answer_verification))
        if not process_used:
            step_scores = []
        
    else:
        if step_count > 0 and not problem:
            print("[Compute Score] Warning: step_count > 0 but no problem text provided")
        step_scores = []
    
    result = {
        "score": outcome_score,
        "step_scores": step_scores,
        "process_used": process_used,
        "answer_verification": answer_verification
    }
    
    print(f"[Compute Score] Result: outcome={outcome_score}, process={len(step_scores)} scores")
    return result