# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from openai import OpenAI
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig
import sys
sys.path.append('/mnt/luoyingfeng/changkaiyan/verl-process')
# sys.path.append('/opt/tiger/hqz_debug/cky/verl-process')
from deepscaler.rewards.math_utils.utils import grade_answer_verl

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

THINK_END_TAG = "</think>"
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.])\s+|\n+")

# vLLM API 配置
VLLM_API_KEY = "EMPTY"
VLLM_API_BASE = "http://localhost:8080/v1"  # 修改为你的 vLLM 服务地址

# API 调用配置
MAX_RETRIES = 5
BASE_DELAY = 2
MAX_WORKERS = 8
TIMEOUT = 5000

EXTRACTION_PROMPT = """\
You are a reasoning trace analyst. Your role is to identify the first sentence where the model gets the answer of a problem. The goal is to identify the redundant overthinking process after the model has actually solved the problem.

You will be given a reasoning trace, which ends with the `</think>`tag; you will also be given the final answer. You must: 

1. Identify only the **first** sentence where the model gets the final answer.
2. The sentence you return should be **exactly the same** as the one in the original reasoning trace.
3. If the sentence containing the final answer is not found, please return **NULL**.

Return only the sentence you identify, no extra commentary or explanation.

---

Final Answer:
{final_answer}

Reasoning trace:
{thinking}
"""

TIGHT_CONCLUSION_CUES = [
    "answer", "therefore", "final", "conclude", "result",
    "equals", "solution", 
]
LOOSE_CONCLUSION_CUES = [
    "thus", "hence", "we get", "we have", "it is", "maybe", "seem", "maximum possible", "would be", "should be", "correct option", "value of", "indeed", "so", "perhaps", "i get", "that's", "it's", "lead to", "the only", "valid", "set",
]
VERIFICATION_CUES = [
    "check",
    "verify",
    "confirm",
    "make sure",
    "double-check",
    "wait",
    "let me",
    "let's",
    "straightforward",
    "miss anything",
    "is that right",
    "is that correct",
    "is that all"
]

def split_think_answer(text: str) -> Tuple[Optional[str], str]:
    """Return (think_part or None if missing, answer_part). Split at the LAST </think>."""
    if THINK_END_TAG not in text:
        return None, text
    idx = text.rfind(THINK_END_TAG)
    return text[:idx], text[idx + len(THINK_END_TAG):]

def find_boxed_spans_bracematch(text: str) -> List[str]:
    """Extract contents inside \\boxed{...} using brace matching."""
    out = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        i = m.end()
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[m.end(): i - 1].strip())
    return out

def extract_final_boxed_answer(answer_text: str) -> Optional[str]:
    boxed = find_boxed_spans_bracematch(answer_text)
    return boxed[-1].strip() if boxed else None

def count_tokens_from_char(tokenizer, text: str, start_char: Optional[int]) -> Optional[int]:
    """Token count for text[start_char:] using a HF tokenizer."""
    if start_char is None:
        return None
    sub = text[start_char:]
    ids = tokenizer.encode(sub, add_special_tokens=False)
    return len(ids)

def split_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
    """
    Split text into sentences but keep (start,end) char spans in the original text.
    Returns: [{"start": int, "end": int, "text": str}, ...]
    """
    spans: List[Dict[str, Any]] = []
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.start()
        raw = text[start:end]
        chunk = raw.strip()
        if chunk:
            lstrip_len = len(raw) - len(raw.lstrip())
            rstrip_len = len(raw) - len(raw.rstrip())
            real_start = start + lstrip_len
            real_end = end - rstrip_len
            spans.append({"start": real_start, "end": real_end, "text": text[real_start:real_end]})
        start = m.end()

    tail_raw = text[start:]
    tail = tail_raw.strip()
    if tail:
        lstrip_len = len(tail_raw) - len(tail_raw.lstrip())
        real_start = start + lstrip_len
        real_end = len(text)
        spans.append({"start": real_start, "end": real_end, "text": text[real_start:real_end]})

    return spans

def call_vllm(client, prompt, temperature):
    
    messages = [
        {"role": "user", "content": prompt},
    ]
    
    for attempt in range(MAX_RETRIES):
        try:
            # 获取模型列表
            models = client.models.list()
            model = models.data[0].id
            
            # 调用 Chat Completion API
            # 设置 stop="</final_verification>" 和 include_stop_str_in_output=True
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=150,
                temperature=temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )
            
            # 提取响应文本
            verification_response = chat_completion.choices[0].message.content
            
            return verification_response
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"VLLM API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                delay = BASE_DELAY * (2 ** attempt)
                sleep(delay)
            else:
                print(f"VLLM API call failed after {MAX_RETRIES} attempts: {e}")
                return None
    
    return None

def find_first_conclusion_mention(
    think_text: str,
    boxed_answer: str,
    client: OpenAI
) -> Dict[str, Any]:
    """
    Find the first sentence that matches the answer (regex) and looks conclusion-like.
    Returns: found, reason, first_idx/score, first_char_start, prev/sent/next, all_matches (idx, score).
    """
    if not boxed_answer.strip():
        return {
            "found": False,
            "reason": "EMPTY_BOXED_ANSWER",
            "first_char_start": None,
            "sent": ""
        }
    
    prompt = EXTRACTION_PROMPT.format(thinking=think_text + "</think>", final_answer=boxed_answer)
    verification_sentence = call_vllm(client, prompt, temperature=0.6)
    pattern = re.compile(f"({re.escape(verification_sentence)})", re.DOTALL)
    match = pattern.search(think_text)
    
    if match:
        return {
            "found": True,
            "reason": "llm",
            "first_char_start": match.end(),
            "sent": verification_sentence
        }
    
    sents = split_sentences_with_spans(think_text)

    first_i = None
    for i, obj in enumerate(sents):
        sent = obj["text"].lower()
        has_conclusion = any(cue in sent for cue in TIGHT_CONCLUSION_CUES) or any(cue in sent for cue in LOOSE_CONCLUSION_CUES) or "\\boxed" in sent
        has_verification = any(cue in (sents[i+1]["text"].lower() if i+1 < len(sents) else "") for cue in VERIFICATION_CUES)
        if has_conclusion or has_verification:
            pred = parse(sent, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            if pred and grade_answer_verl("\\boxed{" + str(pred[-1]) + "}", boxed_answer):
                first_i = i
                break
    if not first_i:
        return {
            "found": False,
            "reason": "ANSWER_NOT_FOUND_IN_THINK",
            "first_char_start": None,
            "sent": ""
        }
    prev_sent = sents[first_i - 1]["text"] if first_i - 1 >= 0 else ""
    next_sent = sents[first_i + 1]["text"] if first_i + 1 < len(sents) else ""

    return {
        "found": True,
        "reason": "rule",
        "first_char_start": sents[first_i]["end"],
        "sent": sents[first_i]["text"]
    }

@register("redundancy_llm")
class RedundancyLlmRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", alpha=1, beta=0.0001, extraction="self", way="correct") -> None:
        """
        Initialize the RedundancyRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.alpha = alpha
        self.beta = beta
        self.extraction = extraction
        self.way = way
        print(f"==ALPHA==: {self.alpha}, ==BETA==: {self.beta}, ==EXTRACT==: {self.extraction}, ==WAY==: {self.way}")

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # 创建 OpenAI 客户端
        client = OpenAI(
            api_key=VLLM_API_KEY,
            base_url=VLLM_API_BASE,
            timeout=TIMEOUT
        )

        metrics = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"] # prompt token ids

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] # valid prompt token ids (without padding)

            response_ids = data_item.batch["responses"] # response token ids
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length] # valid response token ids (without padding)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True) # prompt text str
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True) # response text str

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            # Calculate Verification Length
            think, answer = split_think_answer(response_str)
            has_think = think is not None
            boxed = extract_final_boxed_answer(answer) if think else None
            
            if self.extraction == "self":
                boxed_answer = boxed
            else:
                boxed_answer = boxed if think else ground_truth
                think = think if think else response_str
            
            first = None
            verification_length, think_length = 0, 0
            if think and boxed_answer and not (self.way != "full" and reward == 0):
                think_ids = self.tokenizer.encode(think, add_special_tokens=False)
                think_length = len(think_ids)
                first = find_first_conclusion_mention(think, boxed_answer, client)
                if first["found"]:
                    first_char_start = first["first_char_start"]
                    verification_length = count_tokens_from_char(self.tokenizer, think, first_char_start)
            
            verification_ratio = verification_length / think_length if think_length > 0 else 0
            
            # Compute Rollout Reward
            split = extra_info.get("split", "train")
            no_think_reward = 0
            if split == "train":
                if self.way == "ratio_1": # 方式三用减去比率惩罚，正确和错误都惩罚
                    reward = self.alpha * reward - self.beta * verification_ratio
                elif self.way == "ratio_2": # 方式三用乘以比率惩罚，正确和错误都惩罚
                    reward = self.alpha * reward * (1 - self.beta * verification_ratio)
                elif self.way == "full": # 方式四用长度惩罚，正确的和错误的都惩罚
                    no_think_reward = (0.5 if verification_length else -0.5) if not has_think else 0
                    reward = self.alpha * (reward + no_think_reward) - self.beta * verification_length
                else: # 方式一和方式二用长度惩罚，只惩罚正确的
                    reward = self.alpha * reward - self.beta * verification_length
            
            metrics.append({
                "split": split,
                "outcome": score["score"] if isinstance(score, dict) else score,
                "ground_truth": ground_truth,
                "v_l": verification_length,
                "v_p": verification_ratio,
                "reward": reward,
                "no_think_reward": no_think_reward,
                "first_sent": first["sent"] if first else "",
                "use_llm": first["reason"] if first else "",
                "response": response_str,
            })

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
        
        print(f"==METRICS==\n {metrics}")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
