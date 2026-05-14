import json
import re
from typing import Any, Dict, List, Optional, Tuple
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

THINK_END_TAG = "</think>"
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

def read_responses_from_json(file_path):
    """
    读取JSON文件，提取每个元素的response键值
    
    Args:
        file_path (str): JSON文件路径
        
    Returns:
        list: 包含所有response列表的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    has_only_think_cnt, think_not_contain_boxed_cnt, tneqa_cnt = 0, 0, 0
    for i, item in enumerate(data):
        for response in item['responses']:
            think, answer = split_think_answer(response)
            if think is None:
                has_only_think_cnt += 1
                continue
            boxed_answer = extract_final_boxed_answer(answer)
            boxed_think = extract_final_boxed_answer(think)
            if boxed_answer is None or boxed_think is None:
                think_not_contain_boxed_cnt += 1
                continue
            gold = parse("\\boxed{" + boxed_answer + "}", extraction_config=[LatexExtractionConfig()])
            pred = parse("\\boxed{" + boxed_think + "}", extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            if not verify(gold, pred, timeout_seconds=50):
                tneqa_cnt += 1
    
    print(f"Total Samples: {len(data) * 2}")
    print(f"Has Only Think Count: {has_only_think_cnt}")
    print(f"Think Not Contain Boxed Count: {think_not_contain_boxed_cnt}")
    print(f"Think boxed not equal to answer boxed Count: {tneqa_cnt}")

# 使用示例
if __name__ == "__main__":
    
    file_path = "/mnt/nvme1/luoyingfeng/lucky/verl-process/pre_exp/eval_results/DS-1.5B/Deepscaler-480_FP-8K/results_details.json"
    read_responses_from_json(file_path)