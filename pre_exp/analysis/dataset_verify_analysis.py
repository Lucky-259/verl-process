#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
import json
import re
from pathlib import Path
from openai import OpenAI
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
from math_verify import parse, ExprExtractionConfig, LatexExtractionConfig
from tqdm import tqdm
import sys

sys.path.append('/opt/tiger/hqz_debug/cky/verl-process')
# sys.path.append('/mnt/luoyingfeng/changkaiyan/verl-process')
from deepscaler.rewards.math_utils.utils import grade_answer_verl

THINK_END_TAG = "</think>"

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def split_think_answer(text: str) -> Tuple[Optional[str], str]:
    if THINK_END_TAG not in text:
        return None, text
    idx = text.rfind(THINK_END_TAG)
    return text[:idx], text[idx + len(THINK_END_TAG):]


def find_boxed_spans_bracematch(text: str) -> List[str]:
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


_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.])\s+|\n+")

def split_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
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
    ans = boxed_answer.strip().lower()

    if re.fullmatch(r"[+-]?\d+", ans):
        if ans.isdigit() and len(ans) == 1:
            return re.compile(rf"(?<=\s){re.escape(ans)}(?!\d)")
        return re.compile(rf"(?<!\d){re.escape(ans)}(?!\d)")

    if re.fullmatch(r"[+-]?\d+\.\d+", ans):
        return re.compile(rf"(?<![\d.]){re.escape(ans)}(?![\d.])")

    return re.compile(re.escape(ans))


def prompt_contains_answer(prompt: str, boxed_answer: str, way: str) -> bool:
    if not prompt or not boxed_answer:
        return False
    if way == "base":
        pat = build_answer_regex(boxed_answer)
        return pat.search(prompt) is not None
    elif way == "llm":
        return False
    else:
        pred = parse(prompt, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return pred and grade_answer_verl("\\boxed{" + str(pred[-1]) + "}", boxed_answer)


TIGHT_CONCLUSION_CUES = [
    "answer", "therefore", "final", "conclude", "result",
    "equals", "solution",
]
LOOSE_CONCLUSION_CUES = [
    "thus", "hence", "we get", "we have", "it is", "maybe", "seem",
    "maximum possible", "would be", "should be", "correct option",
    "value of", "indeed", "so", "perhaps", "i get", "that's", "it's",
    "lead to", "the only", "valid", "set",
]
VERIFICATION_CUES = [
    "check", "verify", "confirm", "make sure", "double-check",
    "wait", "let me", "let's", "straightforward", "miss anything",
    "is that right", "is that correct", "is that all"
]


def find_first_conclusion_mention(think_text: str, boxed_answer: str) -> Dict[str, Any]:
    sents = split_sentences_with_spans(think_text)
    if not boxed_answer.strip():
        return {
            "found": False, "reason": "EMPTY_BOXED_ANSWER",
            "first_idx": None, "first_score": None, "first_char_start": None,
            "prev": "", "sent": "", "next": "", "all_matches": []
        }

    ans_pat = build_answer_regex(boxed_answer)

    first_i = None
    for i, obj in enumerate(sents):
        sent = obj["text"].lower()
        if ans_pat.search(sent) is not None:
            has_conclusion = (
                any(cue in sent for cue in TIGHT_CONCLUSION_CUES)
                or any(cue in sent for cue in LOOSE_CONCLUSION_CUES)
                or "\\boxed" in sent
            )
            has_verification = any(
                cue in (sents[i + 1]["text"].lower() if i + 1 < len(sents) else "")
                for cue in VERIFICATION_CUES
            )
            if has_conclusion or has_verification:
                first_i = i
                break

    if first_i is None:
        return {
            "found": False, "reason": "ANSWER_NOT_FOUND_IN_THINK",
            "first_idx": None, "first_score": None, "first_char_start": None,
            "prev": "", "sent": "", "next": "", "all_matches": []
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


def find_first_conclusion_mention_2(think_text: str, boxed_answer: str) -> Dict[str, Any]:
    sents = split_sentences_with_spans(think_text)
    if not boxed_answer.strip():
        return {
            "found": False, "reason": "EMPTY_BOXED_ANSWER",
            "first_idx": None, "first_score": None, "first_char_start": None,
            "prev": "", "sent": "", "next": "", "all_matches": []
        }

    first_i = None
    for i, obj in enumerate(sents):
        sent = obj["text"].lower()
        has_conclusion = (
            any(cue in sent for cue in TIGHT_CONCLUSION_CUES)
            or any(cue in sent for cue in LOOSE_CONCLUSION_CUES)
            or "\\boxed" in sent
        )
        has_verification = any(
            cue in (sents[i + 1]["text"].lower() if i + 1 < len(sents) else "")
            for cue in VERIFICATION_CUES
        )
        if has_conclusion or has_verification:
            pred = parse(sent, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            if pred and grade_answer_verl("\\boxed{" + str(pred[-1]) + "}", boxed_answer):
                first_i = i
                break

    if first_i is None:
        return {
            "found": False, "reason": "ANSWER_NOT_FOUND_IN_THINK",
            "first_idx": None, "first_score": None, "first_char_start": None,
            "prev": "", "sent": "", "next": "", "all_matches": []
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

VLLM_API_KEY = "EMPTY"
VLLM_API_BASE = "http://localhost:8000/v1"
MAX_RETRIES = 5
TIMEOUT = 5000


def call_vllm(client, prompt, temperature):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            models = client.models.list()
            model = models.data[0].id
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=150,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"VLLM API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            else:
                print(f"VLLM API call failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def find_first_conclusion_mention_llm(think_text: str, boxed_answer: str) -> Dict[str, Any]:
    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_API_BASE,
        timeout=TIMEOUT
    )

    if not boxed_answer.strip():
        return {
            "found": False, "reason": "EMPTY_BOXED_ANSWER",
            "first_idx": None, "first_score": None, "first_char_start": None,
            "prev": "", "sent": "", "next": "", "all_matches": []
        }

    prompt = EXTRACTION_PROMPT.format(thinking=think_text + "</think>", final_answer=boxed_answer)
    verification_sentence = call_vllm(client, prompt, temperature=1)
    if not verification_sentence or verification_sentence == "NULL":
        return {
            "found": False, "reason": "ANSWER_NOT_FOUND_IN_THINK",
            "first_idx": None, "first_score": None, "first_char_start": None,
            "prev": "", "sent": "", "next": "", "all_matches": []
        }

    pattern = re.compile(f"({re.escape(verification_sentence)})", re.DOTALL)
    match = pattern.search(think_text)

    if not match:
        return {
            "found": False, "reason": "ANSWER_NOT_FOUND_IN_THINK",
            "first_idx": None, "first_score": None, "first_char_start": None,
            "prev": "", "sent": "", "next": "", "all_matches": []
        }

    prev_text = think_text[:match.start()]
    next_text = think_text[match.end():]
    prev_sents = split_sentences_with_spans(prev_text)
    next_sents = split_sentences_with_spans(next_text)
    prev_sent = prev_sents[-1]["text"] if prev_sents else ""
    next_sent = next_sents[0]["text"] if next_sents else ""

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


def count_answer_sentences(think_text: str, boxed_answer: str, way: str) -> int:
    sents = split_sentences_with_spans(think_text)
    if not boxed_answer.strip():
        return 0
    if way == "base":
        ans_pat = build_answer_regex(boxed_answer)
        return sum(1 for obj in sents if ans_pat.search(obj["text"].lower()) is not None)
    elif way == "llm":
        return 1
    else:
        count = 0
        for obj in sents:
            sent = obj["text"].lower()
            pred = parse(sent, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            if pred and grade_answer_verl("\\boxed{" + str(pred[-1]) + "}", boxed_answer):
                count += 1
        return count


def count_tokens_from_char(tokenizer, text: str, start_char: Optional[int]) -> Optional[int]:
    if start_char is None:
        return None
    sub = text[start_char:]
    ids = tokenizer.encode(sub, add_special_tokens=False)
    return len(ids)


def load_processed_pairs(details_path: Path) -> Set[Tuple[int, int]]:
    processed = set()
    if not details_path.exists():
        return processed

    with details_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                item_idx = obj.get("item_idx")
                resp_idx = obj.get("resp_idx")
                if item_idx is not None and resp_idx is not None:
                    processed.add((int(item_idx), int(resp_idx)))
            except Exception:
                continue
    return processed


def has_done_flag(done_flag_path: Path) -> bool:
    return done_flag_path.exists()


def write_done_flag(done_flag_path: Path, summary: Dict[str, Any]) -> None:
    with done_flag_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def summarize_details_jsonl(details_path: Path, step: str, dataset: str) -> Dict[str, Any]:
    total_responses = 0
    valid_samples = 0
    no_think_tag = 0
    no_boxed = 0
    not_found_in_think = 0

    think_sum = 0.0
    redundant_sum = 0.0
    proportion_sum = 0.0

    if not details_path.exists():
        return {
            "step": step,
            "dataset": dataset,
            "total_responses": 0,
            "valid_samples": 0,
            "no_think_tag": 0,
            "no_boxed": 0,
            "not_found_in_think": 0,
            "avg_think_length": None,
            "avg_redundant_length": None,
            "avg_proportion": None,
        }

    with details_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            total_responses += 1
            status = obj.get("status")
            if status == "NO_THINK_END_TAG":
                no_think_tag += 1
            elif status == "NO_BOXED_IN_ANSWER":
                no_boxed += 1
            elif status == "ANSWER_NOT_FOUND_IN_THINK":
                not_found_in_think += 1

            think_length = obj.get("think_length")
            redundant_length = obj.get("tokens_from_first_answer_to_think_end")
            proportion = obj.get("proportion")

            if think_length is not None and redundant_length is not None and proportion is not None:
                valid_samples += 1
                think_sum += float(think_length)
                redundant_sum += float(redundant_length)
                proportion_sum += float(proportion)

    return {
        "step": step,
        "dataset": dataset,
        "total_responses": total_responses,
        "valid_samples": valid_samples,
        "no_think_tag": no_think_tag,
        "no_boxed": no_boxed,
        "not_found_in_think": not_found_in_think,
        "avg_think_length": (think_sum / valid_samples) if valid_samples > 0 else None,
        "avg_redundant_length": (redundant_sum / valid_samples) if valid_samples > 0 else None,
        "avg_proportion": (proportion_sum / valid_samples) if valid_samples > 0 else None,
    }


def process_single_dataset(args, dataset_name, step, input_path):
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
        raise ValueError(f"Input JSON must be a list of items: {input_path}")

    dataset_dir = Path(args.output_dir) / f"step_{step}" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    detail_out_path = dataset_dir / "details.jsonl"
    done_flag_path = dataset_dir / "done.flag"

    if has_done_flag(done_flag_path):
        print(f"✅ step={step}, dataset={dataset_name} 已完成，直接跳过。")
        return summarize_details_jsonl(detail_out_path, step, dataset_name)

    processed_pairs = load_processed_pairs(detail_out_path)
    if processed_pairs:
        print(f"♻️  step={step}, dataset={dataset_name} 检测到已有 {len(processed_pairs)} 条已处理记录，将继续续跑。")

    total_candidates = 0
    for item in data:
        responses = item.get("responses", [])
        if isinstance(responses, list):
            total_candidates += len(responses)

    with detail_out_path.open("a", encoding="utf-8") as fout:
        for item_idx, item in tqdm(enumerate(data), total=len(data), desc=f"step={step} dataset={dataset_name}"):
            prompt = item.get("prompt", "")
            gt = item.get("ground_truth", "")
            difficulty = item.get("difficulty", "")

            responses = item.get("responses", [])
            correct_flags = item.get("correct_flags", [])
            if not isinstance(responses, list):
                continue

            for resp_idx, resp in enumerate(responses):
                if (item_idx, resp_idx) in processed_pairs:
                    continue

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
                think_length = None

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
                    status = "NO_THINK_END_TAG"
                    think = ""
                if boxed_answer is None:
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

                    answer_sentence_count = count_answer_sentences(think, boxed_answer, args.way)

                record = {
                    "step": step,
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
                    "think_length": think_length,
                    "proportion": (tokens_from_first_to_end / think_length)
                    if (tokens_from_first_to_end is not None and think_length and think_length > 0)
                    else None,
                    "all_matches": all_matches,
                    "prompt": prompt,
                    "ground_truth": gt,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    summary = summarize_details_jsonl(detail_out_path, step, dataset_name)

    if summary["total_responses"] >= total_candidates:
        write_done_flag(done_flag_path, summary)
        print(f"✅ step={step}, dataset={dataset_name} 处理完成，已写入 done.flag")
    else:
        print(f"⚠️  step={step}, dataset={dataset_name} 未完成：{summary['total_responses']}/{total_candidates}")

    return summary


def add_overall_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["step"]].append(row)

    out = list(rows)

    for step, step_rows in grouped.items():
        total_responses = sum(r["total_responses"] for r in step_rows)
        valid_samples = sum(r["valid_samples"] for r in step_rows)
        no_think_tag = sum(r["no_think_tag"] for r in step_rows)
        no_boxed = sum(r["no_boxed"] for r in step_rows)
        not_found_in_think = sum(r["not_found_in_think"] for r in step_rows)

        weighted_think_num = 0.0
        weighted_redundant_num = 0.0
        weighted_prop_num = 0.0

        for r in step_rows:
            if r["avg_think_length"] is not None:
                weighted_think_num += r["avg_think_length"] * r["valid_samples"]
            if r["avg_redundant_length"] is not None:
                weighted_redundant_num += r["avg_redundant_length"] * r["valid_samples"]
            if r["avg_proportion"] is not None:
                weighted_prop_num += r["avg_proportion"] * r["valid_samples"]

        out.append({
            "step": step,
            "dataset": "overall",
            "total_responses": total_responses,
            "valid_samples": valid_samples,
            "no_think_tag": no_think_tag,
            "no_boxed": no_boxed,
            "not_found_in_think": not_found_in_think,
            "avg_think_length": weighted_think_num / valid_samples if valid_samples > 0 else None,
            "avg_redundant_length": weighted_redundant_num / valid_samples if valid_samples > 0 else None,
            "avg_proportion": weighted_prop_num / valid_samples if valid_samples > 0 else None,
        })

    return sorted(out, key=lambda x: (int(x["step"]), x["dataset"]))


def write_summary_csv(rows: List[Dict[str, Any]], output_dir: Path):
    csv_path = output_dir / "AST_statistics_by_step.csv"
    fieldnames = [
        "step",
        "dataset",
        "total_responses",
        "valid_samples",
        "no_think_tag",
        "no_boxed",
        "not_found_in_think",
        "avg_think_length",
        "avg_redundant_length",
        "avg_proportion",
        "avg_proportion_percent",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            prop = out.get("avg_proportion")
            out["avg_proportion_percent"] = prop * 100 if prop is not None else None
            writer.writerow(out)

    print(f"📄 CSV saved to: {csv_path}")


def rebuild_summary_from_output(output_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for step_dir in sorted(output_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[-1])):
        if not step_dir.is_dir():
            continue
        step = step_dir.name.split("_")[-1]

        for dataset_dir in sorted(step_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            details_path = dataset_dir / "details.jsonl"
            if details_path.exists():
                rows.append(summarize_details_jsonl(details_path, step, dataset))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs='+', required=True, help="Paths to input JSON files")
    ap.add_argument("--datasets", nargs='+', required=True, help="Dataset names")
    ap.add_argument("--steps", nargs='+', required=True, help="Step id for each input")
    ap.add_argument("--output_dir", required=True, help="Directory to save outputs")
    ap.add_argument("--extraction", required=True, help="Place to extract boxed answers")
    ap.add_argument("--way", required=True, help="The way to find first sentence")
    ap.add_argument("--min_score", type=int, default=4, help="Min conclusion score threshold")
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

    if not (len(args.inputs) == len(args.datasets) == len(args.steps)):
        raise ValueError("The number of inputs, datasets, and steps must match.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path, dataset_name, step in zip(args.inputs, args.datasets, args.steps):
        print(f"\n{'='*60}")
        print(f"Processing step={step}, dataset={dataset_name}")
        print(f"Input file: {input_path}")
        print(f"{'='*60}")
        process_single_dataset(args, dataset_name, step, input_path)

    rows = rebuild_summary_from_output(output_dir)
    rows = add_overall_rows(rows)
    write_summary_csv(rows, output_dir)

    print("\nAll analysis completed.")


if __name__ == "__main__":
    main()