import json
import re
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

THINK_END_TAG = "</think>"
# ---------- Sentence splitting with spans ----------
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.])\s+|\n+")

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

def append_dict_to_jsonl(file_path, data_dict, ensure_ascii=False, indent=None):
    """
    将字典追加到JSONL文件中
    
    Args:
        file_path: JSONL文件路径
        data_dict: 要追加的字典
        ensure_ascii: 是否确保ASCII编码（False支持中文）
        indent: 缩进，None表示紧凑格式
    """
    try:
        with open(file_path, 'a', encoding='utf-8') as f:
            # 将字典转换为JSON字符串
            json_line = json.dumps(data_dict, ensure_ascii=ensure_ascii, indent=indent)
            # 写入文件，末尾添加换行符
            f.write(json_line + '\n')
        #print(f"成功追加数据到 {file_path}")
    except Exception as e:
        print(f"写入文件失败: {e}")

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
    
    has_only_think_cnt, has_verification_cnt, modified_cnt = 0, 0, 0
    for item in tqdm(data, total=len(data), desc="Processing responses..."):
        for response in item['responses']:
            think, answer = split_think_answer(response)
            if think is None:
                has_only_think_cnt += 1
                continue
            boxed_answer = extract_final_boxed_answer(answer)
            boxed_think = extract_final_boxed_answer(think)
            if boxed_answer is None or boxed_think is None:
                continue
            gold = parse("\\boxed{" + boxed_answer + "}", extraction_config=[LatexExtractionConfig()])
            sents = split_sentences_with_spans(think)
            for i, obj in enumerate(sents):
                sent = obj["text"].lower()
                has_conclusion = any(cue in sent for cue in TIGHT_CONCLUSION_CUES) or "\\boxed" in sent #or any(cue in sent for cue in LOOSE_CONCLUSION_CUES) 
                has_verification = any(cue in (sents[i+1]["text"].lower() if i+1 < len(sents) else "") for cue in VERIFICATION_CUES)
                if has_conclusion and has_verification:
                    has_verification_cnt += 1
                    pred = parse(sent, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
                    if not verify(gold, pred, timeout_seconds=10):
                        modified_cnt += 1
                        sample_dict = {"prompt": item['prompt'], "ground_truth": item['ground_truth'], "boxed_answer": boxed_answer, "conclusion": obj["text"], "verification": sents[i+1]['text'] if i+1 < len(sents) else ""}
                        append_dict_to_jsonl("/mnt/luoyingfeng/changkaiyan/verl-process/pre_exp/eval_results/analysis/v_modified_statistics.jsonl", sample_dict, ensure_ascii=False, indent=4)  # 请在此处填写目标JSONL文件路径
                    break
    
    print(f"Total Samples: {len(data) * 2}")
    print(f"Has Only Think Count: {has_only_think_cnt}")
    print(f"Has Verification Count: {has_verification_cnt}")
    print(f"Think boxed not equal to answer boxed Count: {modified_cnt}")
    print(f"Verification Modified Answer Rate: {modified_cnt / has_verification_cnt if has_verification_cnt > 0 else 0}")

# 使用示例
if __name__ == "__main__":
    
    file_path = "/mnt/luoyingfeng/changkaiyan/verl-process/pre_exp/eval_results/DS-1.5B/Deepscaler-480_FP-8K/results_details.json"
    read_responses_from_json(file_path)