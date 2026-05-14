import re
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoTokenizer
from math_verify import parse, ExprExtractionConfig, LatexExtractionConfig
import sys
sys.path.append('/mnt/luoyingfeng/changkaiyan/verl-process')
from deepscaler.rewards.math_utils.utils import grade_answer_verl

MODEL_PATH = "/mnt/luoyingfeng/model_card/DeepSeek-R1-Distill-Qwen-1.5B"

THINK_END_TAG = "</think>"
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

def prompt_contains_answer(prompt: str, boxed_answer: str) -> bool:
    """Mark samples where the final answer already appears in the prompt."""
    if not prompt or not boxed_answer:
        return False
    pat = build_answer_regex(boxed_answer)
    return pat.search(prompt) is not None

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

def find_first_conclusion_mention_1(
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
                has_conclusion = any(cue in sent for cue in TIGHT_CONCLUSION_CUES) or any(cue in sent for cue in LOOSE_CONCLUSION_CUES) or  "\\boxed" in sent
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
            print(f"sent: {sent}, pred: {pred}")
            if pred and grade_answer_verl(str(pred[-1]), boxed_answer):
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

response_str = """First, I need to determine the percentage of students who own cats. There are 75 students who own cats out of a total of 500 students.\n\nTo find the percentage, I will divide the number of cat owners by the total number of students and then multiply by 100.\n\nSo, 75 divided by 500 equals 0.15, and multiplying by 100 gives 15%. This means 15% of the students own cats.\n\nNext, I will calculate the percentage of students who own dogs. There are 125 students who own dogs out of the same 500 students.\n\nAgain, I will divide the number of dog owners by the total number of students and multiply by 100.\n\n125 divided by 500 equals 0.25, and multiplying by 100 results in 25%. Therefore, 25% of the students own dogs.\n</think>\n\nTo determine the percentage of students who own cats and dogs, follow these steps:\n\n1. **Find the percentage of cat owners:**\n   \\[\n   \\text{Percentage of cat owners} = \\left( \\frac{\\text{Number of cat owners}}{\\text{Total number of students}} \\right) \\times 100\n   \\]\n   \\[\n   \\text{Percentage of cat owners} = \\left( \\frac{75}{500} \\right) \\times 100 = 15\\%\n   \\]\n\n2. **Find the percentage of dog owners:**\n   \\[\n   \\text{Percentage of dog owners} = \\left( \\frac{\\text{Number of dog owners}}{\\text{Total number of students}} \\right) \\times 100\n   \\]\n   \\[\n   \\text{Percentage of dog owners} = \\left( \\frac{125}{500} \\right) \\times 100 = 25\\%\n   \\]\n\n**Final Answer:**\n\\[\n\\boxed{15\\% \\text{ own cats and } 25\\% \\text{ own dogs}}\n\\]"""


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
extraction = "ground"
way = "correct"

ground_truth = "25\\%"
# Calculate Verification Length
think, answer = split_think_answer(response_str)
boxed = extract_final_boxed_answer(answer) if think else None

if extraction == "self":
    boxed_answer = boxed
else:
    think = think if think else response_str
    boxed_answer = ground_truth

reward = 1

verification_length, think_length = 0, 0
if think and boxed_answer and not (way != "full" and reward == 0):
    print("Calculating verification length...")
    think_ids = tokenizer.encode(think, add_special_tokens=False)
    think_length = len(think_ids)
    answer_in_prompt = False # prompt_contains_answer(prompt_str, boxed_answer)
    if not answer_in_prompt:
        print("Finding first conclustion")
        if way == "base": # 方式一不用math_verify，只用正则找
            first = find_first_conclusion_mention_1(think, boxed_answer)
        else: # 方式二、三、四都用math_verify
            first = find_first_conclusion_mention_2(think, boxed_answer)
        if first["found"]:
            first_char_start = first["first_char_start"]
            verification_length = count_tokens_from_char(tokenizer, think, first_char_start)
            print(first)

verification_ratio = verification_length / think_length if think_length > 0 else 0

print(f"Verification Length: {verification_length}")
print(f"Verification Ratio: {verification_ratio:.2f}")