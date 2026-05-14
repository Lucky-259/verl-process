# python verify_analysis.py \
#   --input ../DS-1.5B/AIME25/results_details.json \
#   --output DS-1.5B-AIME25.jsonl \
#   --min_score 4


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
sys.path.append('/mnt/luoyingfeng/changkaiyan/verl-process')
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

occ = defaultdict(int)
has_tight_cnt, has_loose_cnt, has_veri_cnt, valid_responses = 0, 0, 0, 0

def conclusion_score(sentence: str, next_sentence: str, answer_pat: re.Pattern) -> int:
    """Heuristic scoring (regex-based answer match)."""
    s = sentence.lower()
    score = 0

    for cue in TIGHT_CONCLUSION_CUES:
        if cue in s:
            score += 2

    for cue in LOOSE_CONCLUSION_CUES:
        if cue in s:
            score += 1
            break
    
    n_s = next_sentence.lower()
    for v_cue in VERIFICATION_CUES:
        if v_cue in n_s:
            score += 1

    if "\\boxed" in sentence:
        score += 4
    #if "=" in sentence:
    #    score += 1
    
    #if answer_pat.search(s) is not None:
    #    score += 2
    #if answer_pat.search(s) is None:
    #    score = 0

    #if re.search(r"\b(minutes?|hours?|secs?|seconds?|cm|mm|km|m)\b", s):
    #    score += 1

    return score


## ---------- Find first conclusion mention ----------
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

    global has_tight_cnt, has_loose_cnt, has_veri_cnt, occ

    for cue in TIGHT_CONCLUSION_CUES:
        if cue in sents[first_i]["text"]:
            occ[cue] += 1
    for cue in LOOSE_CONCLUSION_CUES:
        if cue in sents[first_i]["text"]:
            occ[cue] += 1
    for cue in VERIFICATION_CUES:
        if cue in next_sent:
            occ[cue] += 1
    has_tight_cnt += any(cue in sents[first_i]["text"] for cue in TIGHT_CONCLUSION_CUES)
    has_loose_cnt += any(cue in sents[first_i]["text"] for cue in LOOSE_CONCLUSION_CUES)
    has_veri_cnt += any(cue in next_sent for cue in VERIFICATION_CUES)

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

    global has_tight_cnt, has_loose_cnt, has_veri_cnt, occ

    for cue in TIGHT_CONCLUSION_CUES:
        if cue in sents[first_i]["text"]:
            occ[cue] += 1
    for cue in LOOSE_CONCLUSION_CUES:
        if cue in sents[first_i]["text"]:
            occ[cue] += 1
    for cue in VERIFICATION_CUES:
        if cue in next_sent:
            occ[cue] += 1
    has_tight_cnt += any(cue in sents[first_i]["text"] for cue in TIGHT_CONCLUSION_CUES)
    has_loose_cnt += any(cue in sents[first_i]["text"] for cue in LOOSE_CONCLUSION_CUES)
    has_veri_cnt += any(cue in next_sent for cue in VERIFICATION_CUES)

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

    global has_tight_cnt, has_loose_cnt, has_veri_cnt, occ

    for cue in TIGHT_CONCLUSION_CUES:
        if cue in verification_sentence:
            occ[cue] += 1
    for cue in LOOSE_CONCLUSION_CUES:
        if cue in verification_sentence:
            occ[cue] += 1
    for cue in VERIFICATION_CUES:
        if cue in next_sent:
            occ[cue] += 1
    has_tight_cnt += any(cue in verification_sentence for cue in TIGHT_CONCLUSION_CUES)
    has_loose_cnt += any(cue in verification_sentence for cue in LOOSE_CONCLUSION_CUES)
    has_veri_cnt += any(cue in next_sent for cue in VERIFICATION_CUES)

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

def draw_cue_occurence(occ, args):
    # 提取字典中的键和值
    keys = list(occ.keys())
    values = list(occ.values())

    # 创建条形图
    plt.bar(keys, values)

    # 添加标题和标签
    plt.title("Cues Occurrence")
    plt.xlabel("Cues")
    plt.ylabel("Occ Freq")

    # 保存为 PNG 文件
    plt.savefig(args.occ, format='png')

def draw_proportion_bar(think_length_dict, proportion_dict, args):
    # 计算每个键的平均值，并按键升序排序
    think_length_dict = dict(sorted({k: sum(v) / len(v) for k, v in think_length_dict.items()}.items()))
    proportion_dict = dict(sorted({k: sum(v) / len(v) for k, v in proportion_dict.items()}.items()))

    # 创建一个条形图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 为每个键计算红色和紫色的部分
    for i, key in enumerate(proportion_dict.keys()):
        # 获取红色部分的高度（来自 think_length_dict）
        red_height = think_length_dict.get(key, 0)  # 默认值为0，如果key不存在
        # 获取紫色部分的高度占比（来自 proportion_dict）
        purple_ratio = proportion_dict.get(key, 0)  # 默认值为0，如果key不存在
        # 计算紫色部分的高度
        purple_height = red_height * purple_ratio

        # 横坐标的位置
        x_pos = i

        # 绘制红色部分（从0到红色的高度）
        ax.bar(x_pos, red_height, color='red', edgecolor='black')

        # 绘制紫色部分（从红色的顶部开始）
        ax.bar(x_pos, purple_height, bottom=0, color='purple', edgecolor='black')
        
        # 显示purple_ratio的百分比
        ax.text(x_pos, purple_height / 2, f'{purple_ratio*100:.1f}%', ha='center', va='center', color='white')

    # 设置图形标签
    ax.set_xticks(range(len(proportion_dict)))  # 设置横坐标的位置
    ax.set_xticklabels(proportion_dict.keys())  # 设置横坐标的标签
    ax.set_xlabel('Difficulty')  # 设置 x 轴标签
    ax.set_ylabel('Proportion')  # 设置 y 轴标签
    ax.set_title('Stacked Bar Chart with Red and Purple Sections')  # 设置图表标题

    # 保存为 PNG 文件
    plt.savefig(args.proportion_bar, format='png')


def draw_length_scatter(length_dict, args):

    length_dict = dict(sorted(length_dict.items()))

    # 创建空列表存储横纵坐标的数据
    x_values = []
    y_values = []

    # 为每个键对应的浮动数生成散点图的数据
    for i, (key, values) in enumerate(length_dict.items()):
        # x坐标是键的索引，y坐标是该键的浮动数列表
        x_values.extend([i] * len(values))  # 每个点的横坐标
        y_values.extend(values)  # 每个点的纵坐标

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values)

    # 设置横坐标的标签（字典的键）
    plt.xticks(range(len(length_dict)), list(length_dict.keys()))

    # 设置图形标签
    plt.xlabel('Difficulty')
    plt.ylabel('Length')
    plt.title('Scatter Plot for Verification Length')
    # 保存为 PNG 文件
    plt.savefig(args.length_scatter, format='png')

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
    plt.savefig(args.length_scatter, format='png')

def draw_proportion_scatter(proportion_dict, args):

    proportion_dict = dict(sorted(proportion_dict.items()))

    # 创建空列表存储横纵坐标的数据
    x_values = []
    y_values = []

    # 为每个键对应的浮动数生成散点图的数据
    for i, (key, values) in enumerate(proportion_dict.items()):
        # x坐标是键的索引，y坐标是该键的浮动数列表
        x_values.extend([i] * len(values))  # 每个点的横坐标
        y_values.extend(values)  # 每个点的纵坐标

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values)

    # 设置横坐标的标签（字典的键）
    plt.xticks(range(len(proportion_dict)), list(proportion_dict.keys()))

    # 设置图形标签
    plt.xlabel('Difficulty')
    plt.ylabel('Proportion')
    plt.title('Scatter Plot for Verification Proportion')
    # 保存为 PNG 文件
    plt.savefig(args.proportion_scatter, format='png')

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
    plt.savefig(args.proportion_scatter, format='png')


# ---------- Main ----------
def main():
    global has_tight_cnt, has_loose_cnt, has_veri_cnt, occ, valid_responses
    length_dict = defaultdict(list)
    proportion_dict = defaultdict(list)
    think_length_dict = defaultdict(list)
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSON file")
    ap.add_argument("--output", required=True, help="Path to output JSONL file")
    ap.add_argument("--occ", required=True, help="Path to save PNG picture")
    ap.add_argument("--proportion_bar", required=True, help="Path to save Proportion Bar PNG picture")
    ap.add_argument("--length_scatter", required=True, help="Path to save Length Scatter PNG picture")
    ap.add_argument("--proportion_scatter", required=True, help="Path to save Proportion Scatter PNG picture")
    ap.add_argument("--extraction", required=True, help="Place to extract boxed answers")
    ap.add_argument("--way", required=True, help="The Way to find first sentence")
    ap.add_argument("--min_score", type=int, default=4, help="Min conclusion score threshold (default 4)")
    ap.add_argument(
        "--tokenizer_path",
        default="/mnt/nvme1/luoyingfeng/lucky/model_card/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Tokenizer path for token counting",
    )
    ap.add_argument(
        "--tokens_correct_only",
        action="store_true",
        help="If set, only compute tokens_from_first_answer_to_think_end for correct_flag==True.",
    )
    args = ap.parse_args()

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

    in_path = Path(args.input)
    dataset_name = in_path.parent.name
    data = load_json(in_path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of items, as in your example.")

    total_responses = 0
    no_think_tag = 0
    no_boxed = 0
    not_found_in_think = 0

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as fout, open(f"{out_path.parent}/high_samples.json", "w", encoding="utf-8") as high_fout:
        for item_idx, item in tqdm(enumerate(data), total=len(data), desc="Processing items"):
            prompt = item.get("prompt", "")
            gt = item.get("ground_truth", "")
            difficulty = item.get("difficulty", "")

            responses = item.get("responses", [])
            correct_flags = item.get("correct_flags", [])
            if not isinstance(responses, list):
                continue

            #for resp_idx, resp in enumerate([responses[0]]):
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
                    valid_responses += 1
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
                if tokens_from_first_to_end is not None and (tokens_from_first_to_end / think_length) >= 0.7:
                        high_fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    for k in occ.keys():
        occ[k] /= valid_responses
    has_tight_cnt /= valid_responses
    has_loose_cnt /= valid_responses
    has_veri_cnt /= valid_responses

    print("==== Summary ====")
    print(f"Dataset: {dataset_name}")
    print(f"Total responses: {total_responses}")
    print(f"Responses without </think>: {no_think_tag}")
    print(f"Responses with no boxed answer: {no_boxed}")
    print(f"Responses where boxed answer not found in think: {not_found_in_think}")
    print(f"Output written to: {out_path}")
    print(f"Has Tight Cnt: {has_tight_cnt}, Has Loose Cnt: {has_loose_cnt}, Has Verification Cnt: {has_veri_cnt}")

    #for v in proportion_dict.values():
    #    if type(v) is not list:
    #        print(type(v))
    draw_cue_occurence(occ, args)
    draw_proportion_bar(think_length_dict, proportion_dict, args)
    draw_length_heatmap(length_dict, args)
    draw_proportion_heatmap(proportion_dict, args)

if __name__ == "__main__":
    main()
