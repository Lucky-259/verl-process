#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import json
import re
from pathlib import Path
from openai import OpenAI
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
from math_verify import parse, ExprExtractionConfig, LatexExtractionConfig
from tqdm import tqdm
import sys
sys.path.append('/opt/tiger/hqz_debug/cky/verl-process')
# sys.path.append('/mnt/luoyingfeng/changkaiyan/verl-process')
from deepscaler.rewards.math_utils.utils import grade_answer_verl

THINK_END_TAG = "</think>"

# ---------- Tokenizer ----------
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


# ---------- IO ----------
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- Think/Answer split ----------
def split_think_answer(text: str) -> Tuple[Optional[str], str]:
    """Return (think_part or None if missing, answer_part). Split at the LAST </think>."""
    if THINK_END_TAG not in text:
        return None, text
    idx = text.rfind(THINK_END_TAG)
    return text[:idx], text[idx + len(THINK_END_TAG):]


# ---------- Boxed extraction ----------
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


# ---------- Sentence splitting with spans ----------
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.])\s+|\n+")

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


# ---------- Answer matching (fix: avoid 9 matching 196) ----------
def build_answer_regex(boxed_answer: str) -> re.Pattern:
    """
    Build a regex that matches the answer as an "independent" occurrence to avoid
    cases like answer='9' matching '196'.

    Rules:
    - Pure integer: match not preceded/followed by a digit.
    - Pure decimal: match not preceded/followed by digit or dot.
    - Otherwise: fallback to literal substring (escaped).
    """
    ans = boxed_answer.strip().lower()

    if re.fullmatch(r"[+-]?\d+", ans): # 是整数
        if ans.isdigit() and len(ans) == 1: # 是0-9的整数，前面必须是空格，后面必须无数字
            return re.compile(rf"(?<=\s){re.escape(ans)}(?!\d)")
        return re.compile(rf"(?<!\d){re.escape(ans)}(?!\d)") # 是大于9的整数，前后必须无数字

    if re.fullmatch(r"[+-]?\d+\.\d+", ans): # 是小数
        return re.compile(rf"(?<![\d.]){re.escape(ans)}(?![\d.])")

    return re.compile(re.escape(ans))

def prompt_contains_answer(prompt: str, boxed_answer: str, way: str) -> bool:
    """Mark samples where the final answer already appears in the prompt."""
    if not prompt or not boxed_answer:
        return False
    if way == "base":
        pat = build_answer_regex(boxed_answer)
        return pat.search(prompt) is not None
    elif way == "llm":
        return False
    else:
        #gold = parse("\\boxed{" + boxed_answer + "}", extraction_config=[LatexExtractionConfig()])
        pred = parse(prompt, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return pred and grade_answer_verl("\\boxed{" + str(pred[-1]) + "}", boxed_answer)
    
# ---------- Conclusion scoring ----------
#CONCLUSION_CUES = [
#    "answer", "therefore", "thus", "hence", "so", "final", "conclude", "result",
#    "we get", "we have", "it is", "equals"
#]
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

# ---------- Find first conclusion mention ----------
def find_first_conclusion_mention(
    think_text: str,
    boxed_answer: str,
) -> Dict[str, Any]:
    """
    Find the first sentence that matches the answer (regex) and looks conclusion-like.
    Returns: found, reason, first_idx/score, first_char_start, prev/sent/next, all_matches (idx, score).
    """
    sents = split_sentences_with_spans(think_text)
    if not boxed_answer.strip():
        return {
            "found": False,
            "reason": "EMPTY_BOXED_ANSWER",
            "first_idx": None,
            "first_score": None,
            "first_char_start": None,
            "prev": "",
            "sent": "",
            "next": "",
            "all_matches": []
        }
    
    ans_pat = build_answer_regex(boxed_answer)

    first_i = None
    for i, obj in enumerate(sents):
        sent = obj["text"].lower()
        if ans_pat.search(sent) is not None:
            has_conclusion = any(cue in sent for cue in TIGHT_CONCLUSION_CUES) or any(cue in sent for cue in LOOSE_CONCLUSION_CUES) or "\\boxed" in sent
            has_verification = any(cue in (sents[i+1]["text"].lower() if i+1 < len(sents) else "") for cue in VERIFICATION_CUES)
            if has_conclusion or has_verification:
                first_i = i
                break
    if not first_i:
        return {
            "found": False,
            "reason": "ANSWER_NOT_FOUND_IN_THINK",
            "first_idx": None,
            "first_score": None,
            "first_char_start": None,
            "prev": "",
            "sent": "",
            "next": "",
            "all_matches": []
        }

    prev_sent = sents[first_i - 1]["text"] if first_i - 1 >= 0 else ""
    next_sent = sents[first_i + 1]["text"] if first_i + 1 < len(sents) else ""

    return {
        "found": True,
        "reason": "",
        "first_idx": first_i,
        "first_score": None,
        "first_char_start": sents[first_i]["end"],
        "prev": prev_sent,
        "sent": sents[first_i]["text"],
        "next": next_sent,
        "all_matches": []
    }

def find_first_conclusion_mention_2(
    think_text: str,
    boxed_answer: str,
) -> Dict[str, Any]:
    """
    Find the first sentence that matches the answer (regex) and looks conclusion-like.
    Returns: found, reason, first_idx/score, first_char_start, prev/sent/next, all_matches (idx, score).
    """
    sents = split_sentences_with_spans(think_text)
    if not boxed_answer.strip():
        return {
            "found": False,
            "reason": "EMPTY_BOXED_ANSWER",
            "first_idx": None,
            "first_score": None,
            "first_char_start": None,
            "prev": "",
            "sent": "",
            "next": "",
            "all_matches": []
        }
    #gold = parse("\\boxed{" + boxed_answer + "}", extraction_config=[LatexExtractionConfig()])

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
            "first_idx": None,
            "first_score": None,
            "first_char_start": None,
            "prev": "",
            "sent": "",
            "next": "",
            "all_matches": []
        }

    prev_sent = sents[first_i - 1]["text"] if first_i - 1 >= 0 else ""
    next_sent = sents[first_i + 1]["text"] if first_i + 1 < len(sents) else ""

    return {
        "found": True,
        "reason": "",
        "first_idx": first_i,
        "first_score": None,
        "first_char_start": sents[first_i]["end"],
        "prev": prev_sent,
        "sent": sents[first_i]["text"],
        "next": next_sent,
        "all_matches": []
    }

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

# vLLM API 配置
VLLM_API_KEY = "EMPTY"
VLLM_API_BASE = "http://localhost:8000/v1"  # 修改为你的 vLLM 服务地址

# API 调用配置
MAX_RETRIES = 5
BASE_DELAY = 2
MAX_WORKERS = 8
TIMEOUT = 5000

def call_vllm(client, prompt, temperature):
    """
    调用VLLM API 获取句子
    
    Args:
        prompt: 模型输入
        vllm_api_base: vLLM API 基础 URL
    
    Returns:
        verification_response: PRM 生成的验证文本
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
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

def find_first_conclusion_mention_llm(
    think_text: str,
    boxed_answer: str,
) -> Dict[str, Any]:
    """
    Find the first sentence that matches the answer (regex) and looks conclusion-like.
    Returns: found, reason, first_idx/score, first_char_start, prev/sent/next, all_matches (idx, score).
    """
    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_API_BASE,
        timeout=TIMEOUT
    )

    if not boxed_answer.strip():
        return {
            "found": False,
            "reason": "EMPTY_BOXED_ANSWER",
            "first_idx": None,
            "first_score": None,
            "first_char_start": None,
            "prev": "",
            "sent": "",
            "next": "",
            "all_matches": []
        }
    prompt = EXTRACTION_PROMPT.format(thinking=think_text + "</think>", final_answer=boxed_answer)
    verification_sentence = call_vllm(client, prompt, temperature=1)
    pattern = re.compile(f"({re.escape(verification_sentence)})", re.DOTALL)
    match = pattern.search(think_text)

    if not match:
        return {
            "found": False,
            "reason": "ANSWER_NOT_FOUND_IN_THINK",
            "first_idx": None,
            "first_score": None,
            "first_char_start": None,
            "prev": "",
            "sent": "",
            "next": "",
            "all_matches": []
        }

    prev_text = think_text[:match.start()]
    next_text = think_text[match.end():]
    prev_sents = split_sentences_with_spans(prev_text)
    next_sents = split_sentences_with_spans(next_text)
    prev_sent = prev_sents[-1] if prev_sents else ""
    next_sent = next_sents[0] if next_sents else ""

    return {
        "found": True,
        "reason": "",
        "first_idx": len(prev_sents) + 1,
        "first_score": None,
        "first_char_start": match.start(),
        "prev": prev_sent,
        "sent": verification_sentence,
        "next": next_sent,
        "all_matches": []
    }


# ---------- Count answer sentences (regex-based) ----------
def count_answer_sentences(think_text: str, boxed_answer: str, way: str) -> int:
    """Count how many sentences in think match the answer (regex-based)."""
    sents = split_sentences_with_spans(think_text)
    if not boxed_answer.strip():
        return 0
    if way == "base":
        ans_pat = build_answer_regex(boxed_answer)
        return sum(1 for obj in sents if ans_pat.search(obj["text"].lower()) is not None)
    elif way == "llm":
        return 1
    else:
        #boxed_answer = "\\boxed{" + boxed_answer + "}"
        #gold = parse(boxed_answer, extraction_config=[LatexExtractionConfig()])
        count = 0
        for obj in sents:
            sent = obj["text"].lower()
            pred = parse(sent, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            if pred and grade_answer_verl("\\boxed{" + str(pred[-1]) + "}", boxed_answer):
                count += 1
        return count

# ---------- Token count ----------
def count_tokens_from_char(tokenizer, text: str, start_char: Optional[int]) -> Optional[int]:
    """Token count for text[start_char:] using a HF tokenizer."""
    if start_char is None:
        return None
    sub = text[start_char:]
    ids = tokenizer.encode(sub, add_special_tokens=False)
    return len(ids)

def draw_length_heatmap(length_dict, args):
    # 计算每个键的平均值，并按键升序排序
    length_dict = dict(sorted(length_dict.items()))

    # 创建空列表存储横纵坐标的数据
    x_values = []
    y_values = []

    # 为每个键对应的浮动数生成数据
    for i, (key, values) in enumerate(length_dict.items()):
        # x坐标是键的索引，y坐标是该键的浮动数列表
        x_values.extend([i] * len(values))  # 每个点的横坐标
        y_values.extend(values)  # 每个点的纵坐标

    # 绘制热图：使用 hist2d 来表示点的密度，采用正方形网格
    plt.figure(figsize=(8, 6))
    plt.hist2d(x_values, y_values, bins=[len(length_dict), 120], cmap='Reds')  # bins=30代表纵坐标的分组数

    # 添加颜色条（表示密度）
    plt.colorbar(label='Density')

    # 设置横坐标的标签（字典的键）
    plt.xticks(range(len(length_dict)), list(length_dict.keys()))

    # 设置图形标签
    plt.xlabel('Difficulty')
    plt.ylabel('Length')
    plt.title('Heatmap for Verification Length')

    # 保存为 PNG 文件
    plt.savefig(args.length_heatmap, format='png')

def draw_proportion_heatmap(proportion_dict, args):
    proportion_dict = dict(sorted(proportion_dict.items()))

    # 创建空列表存储横纵坐标的数据
    x_values = []
    y_values = []

    # 为每个键对应的浮动数生成数据
    for i, (key, values) in enumerate(proportion_dict.items()):
        # x坐标是键的索引，y坐标是该键的浮动数列表
        x_values.extend([i] * len(values))  # 每个点的横坐标
        y_values.extend(values)  # 每个点的纵坐标

    # 绘制热图：使用 hist2d 来表示点的密度，采用正方形网格
    plt.figure(figsize=(8, 6))
    plt.hist2d(x_values, y_values, bins=[len(proportion_dict), 120], cmap='Blues')  # bins=30代表纵坐标的分组数

    # 添加颜色条（表示密度）
    plt.colorbar(label='Density')

    # 设置横坐标的标签（字典的键）
    plt.xticks(range(len(proportion_dict)), list(proportion_dict.keys()))

    # 设置图形标签
    plt.xlabel('Difficulty')
    plt.ylabel('Proportion')
    plt.title('Heatmap for Verification Proportion')

    # 保存为 PNG 文件
    plt.savefig(args.proportion_heatmap, format='png')

def draw_multi_dataset_proportion_bar(dataset_stats, args):
    """
    绘制多个数据集的比例柱状图，横坐标为不同数据集
    
    Args:
        dataset_stats: 字典，key为数据集名称，value为(avg_think_length, avg_proportion)
        args: 命令行参数
    """
    if not dataset_stats:
        return
    
    # 按数据集名称排序
    sorted_datasets = sorted(dataset_stats.keys())
    
    # 提取数据
    think_lengths = [dataset_stats[ds][0] for ds in sorted_datasets]
    proportions = [dataset_stats[ds][1] for ds in sorted_datasets]
    
    # 计算验证长度
    verification_lengths = [think_lengths[i] * proportions[i] for i in range(len(sorted_datasets))]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(max(8, len(sorted_datasets) * 1.5), 8))
    
    x = range(len(sorted_datasets))
    
    # 绘制红色部分（平均思考长度）
    ax.bar(x, think_lengths, color='red', edgecolor='black', label='Avg Think Length', alpha=0.7)
    
    # 绘制紫色部分（验证长度）
    ax.bar(x, verification_lengths, bottom=0, color='purple', edgecolor='black', 
           label='Verification Length', alpha=0.7)
    
    # 在每个柱子上方显示比例百分比
    for i, (think_len, prop) in enumerate(zip(think_lengths, proportions)):
        verif_len = think_len * prop
        # 在紫色部分中间显示比例
        ax.text(i, verif_len / 2, f'{prop*100:.1f}%', 
                ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        # 在红色部分顶部显示思考长度
        ax.text(i, think_len + think_len * 0.02, f'{think_len:.0f}', 
                ha='center', va='bottom', fontsize=8)
    
    # 设置图形标签
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_datasets, rotation=45, ha='right')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Token Length')
    ax.set_title('Think Length and Verification Proportion Across Datasets')
    ax.legend()
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 保存为 PNG 文件
    plt.tight_layout()
    plt.savefig(str(Path(args.output_dir)/ "multi_proportion_bar.png"), format='png')
    plt.close()
    print(f"Multi-dataset proportion bar chart saved to: {args.output_dir}")

# ---------- Main ----------
def process_single_dataset(args, dataset_name, input_path):
    """处理单个数据集"""
    # 获取目录部分
    directory = os.path.dirname(args.output)
    
    # 如果目录不存在，则创建（包括所有父目录）
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)  # exist_ok=True 避免已存在时报错

    if AutoTokenizer is None:
        raise ImportError("transformers is not installed. Please `pip install transformers`.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        use_fast=True,
        trust_remote_code=True
    )

    in_path = Path(input_path)
    data = load_json(in_path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items, as in your example.")

    total_responses = 0
    no_think_tag = 0
    no_boxed = 0
    not_found_in_think = 0
    
    length_dict = defaultdict(list)
    proportion_dict = defaultdict(list)
    think_length_dict = defaultdict(list)
    think_length_lst = []
    proportion_lst = []

    out_path = Path(args.output)
    #with out_path.open("w", encoding="utf-8") as fout, open(f"{out_path.parent}/high_samples.json", "w", encoding="utf-8") as high_fout:
    with out_path.open("w", encoding="utf-8") as fout:
        for item_idx, item in tqdm(enumerate(data), total=len(data), desc="Processing items"):
            prompt = item.get("prompt", "")
            gt = item.get("ground_truth", "")
            difficulty = item.get("difficulty", "")

            responses = item.get("responses", [])
            correct_flags = item.get("correct_flags", [])
            if not isinstance(responses, list):
                continue

            for resp_idx, resp in enumerate(responses):
                total_responses += 1

                # Defaults
                status = "OK"
                boxed_answer = ""
                answer_sentence_count = 0
                first_idx = None
                first_score = None
                prev_sent = ""
                first_sent = ""
                next_sent = ""
                all_matches = []
                tokens_from_first_to_end = None
                first_char_start = None
                answer_in_prompt = False

                flag = (
                    correct_flags[resp_idx]
                    if isinstance(correct_flags, list) and resp_idx < len(correct_flags)
                    else None
                )

                think, answer = split_think_answer(resp)
                boxed = extract_final_boxed_answer(answer) if think else None

                if args.extraction == "self":
                    boxed_answer = boxed
                else:
                    think = think if think else resp
                    boxed_answer = gt
                
                has_think = think is not None
                if not has_think:
                    no_think_tag += 1
                    status = "NO_THINK_END_TAG"
                    think = ""
                if boxed_answer is None:
                    no_boxed += 1
                    status = "NO_BOXED_IN_ANSWER"
                else:
                    answer_in_prompt = prompt_contains_answer(prompt, boxed_answer, args.way)
                    if answer_in_prompt:
                        status = "ANSWER_IN_PROMPT"

                if status == "OK":
                    think_ids = tokenizer.encode(think, add_special_tokens=False)
                    think_length = len(think_ids)
                    if args.way == "base":
                        first = find_first_conclusion_mention(think, boxed_answer)
                    elif args.way == "llm":
                        first = find_first_conclusion_mention_llm(think, boxed_answer)
                    else:
                        first = find_first_conclusion_mention_2(think, boxed_answer)
                    if not first["found"]:
                        not_found_in_think += 1
                        status = first["reason"]
                        all_matches = first["all_matches"]
                    else:
                        first_idx = first["first_idx"]
                        first_score = first["first_score"]
                        first_char_start = first["first_char_start"]
                        prev_sent = first["prev"]
                        first_sent = first["sent"]
                        next_sent = first["next"]
                        all_matches = first["all_matches"]

                        do_tokens = True
                        if args.tokens_correct_only:
                            do_tokens = (flag is True)

                        if do_tokens:
                            tokens_from_first_to_end = count_tokens_from_char(tokenizer, think, first_char_start)
                            length_dict[difficulty].append(tokens_from_first_to_end)
                            proportion_dict[difficulty].append(tokens_from_first_to_end / think_length)
                            think_length_dict[difficulty].append(think_length)
                            think_length_lst.append(think_length)
                            proportion_lst.append(tokens_from_first_to_end / think_length)

                    answer_sentence_count = count_answer_sentences(think, boxed_answer, args.way)

                record = {
                    "dataset": dataset_name,
                    "difficulty": difficulty,

                    "item_idx": item_idx,
                    "resp_idx": resp_idx,

                    "status": status,
                    "has_think_end_tag": has_think,
                    "boxed_answer": boxed_answer,
                    "answer_in_prompt": answer_in_prompt,

                    "first_answer_sentence_idx": first_idx,
                    "first_answer_score": first_score,
                    "first_prev_sentence": prev_sent,
                    "first_sentence": first_sent,
                    "first_next_sentence": next_sent,

                    "correct_flag": flag,
                    "answer_sentence_count": answer_sentence_count,
                    "tokens_from_first_answer_to_think_end": tokens_from_first_to_end,
                    "proportion": (tokens_from_first_to_end / think_length) if tokens_from_first_to_end is not None else None,

                    "all_matches": all_matches,
                    "prompt": prompt,
                    "ground_truth": gt,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                #if tokens_from_first_to_end is not None and (tokens_from_first_to_end / think_length) >= 0.7:
                #        high_fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("==== Summary ====")
    print(f"Dataset: {dataset_name}")
    print(f"Total responses: {total_responses}")
    print(f"Responses without </think>: {no_think_tag}")
    print(f"Responses with no boxed answer: {no_boxed}")
    print(f"Responses where boxed answer not found in think: {not_found_in_think}")
    print(f"Output written to: {out_path}")
    
    # 绘制单个数据集的比例柱状图
    draw_length_heatmap(length_dict, args)
    draw_proportion_heatmap(proportion_dict, args)
    
    return think_length_lst, proportion_lst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs='+', required=True, help="Paths to input JSON files (space-separated)")
    ap.add_argument("--datasets", nargs='+', required=True, help="Dataset names corresponding to input files")
    ap.add_argument("--output_dir", required=True, help="Directory to save all outputs")
    ap.add_argument("--extraction", required=True, help="Place to extract boxed answers")
    ap.add_argument("--way", required=True, help="The Way to find first sentence")
    ap.add_argument("--min_score", type=int, default=4, help="Min conclusion score threshold (default 4)")
    ap.add_argument(
        "--tokenizer_path",
        default="/mnt/hdfs/if_au/models/DeepSeek-R1-Distill-Qwen-1.5B",
        # default="/mnt/luoyingfeng/model_card/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Tokenizer path for token counting",
    )
    ap.add_argument(
        "--tokens_correct_only",
        action="store_true",
        help="If set, only compute tokens_from_first_answer_to_think_end for correct_flag==True.",
    )
    args = ap.parse_args()

    if len(args.inputs) != len(args.datasets):
        raise ValueError("Number of inputs must match number of dataset names")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储所有数据集的统计信息
    dataset_stats = {}  # key: dataset_name, value: (avg_think_length, avg_proportion)
    
    # 处理所有数据集
    for input_path, dataset_name in zip(args.inputs, args.datasets):
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"Input file: {input_path}")
        print(f"{'='*50}")
        
        # 为每个数据集设置输出路径
        args.output = str(output_dir / f"{dataset_name}" / "details.jsonl")
        args.proportion_heatmap = str(output_dir / f"{dataset_name}" / "proportion_heatmap.png")
        args.length_heatmap = str(output_dir / f"{dataset_name}" / "length_heatmap.png")
        
        # 处理单个数据集
        think_length_lst, proportion_lst = process_single_dataset(args, dataset_name, input_path)
        
        if think_length_lst and proportion_lst:
            avg_think = sum(think_length_lst) / len(think_length_lst)
            avg_prop = sum(proportion_lst) / len(proportion_lst)
            dataset_stats[dataset_name] = (avg_think, avg_prop)
    
    # 绘制多个数据集的比例柱状图
    if dataset_stats:
        draw_multi_dataset_proportion_bar(dataset_stats, args)
        print(f"\nMulti-dataset analysis completed!")
        print("Dataset statistics:")
        for ds_name, (think_len, prop) in dataset_stats.items():
            print(f"  {ds_name}: Think length = {think_len:.1f}, Proportion = {prop:.3f} ({prop*100:.1f}%)")
    else:
        print("\nNo valid data found for any dataset.")
    
    if dataset_stats:
        # 将统计信息保存到CSV文件
        csv_path = output_dir / "AST_statistics.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # 写入标题行
            headers = ['Metric'] + list(dataset_stats.keys())
            writer.writerow(headers)

            # 写入think长度行
            think_row = ['Think Length'] + [dataset_stats[ds][0] for ds in dataset_stats.keys()]
            writer.writerow(think_row)

            # 写入冗余占比行
            proportion_row = ['Proportion (%)'] + [dataset_stats[ds][1] * 100 for ds in dataset_stats.keys()]
            writer.writerow(proportion_row)

            # 写入冗余长度行
            redundant_length_row = ['Redundant Length'] + [dataset_stats[ds][0] * dataset_stats[ds][1] for ds in dataset_stats.keys()]
            writer.writerow(redundant_length_row)

        print(f"\nDataset statistics saved to: {csv_path}")

if __name__ == "__main__":
    main()