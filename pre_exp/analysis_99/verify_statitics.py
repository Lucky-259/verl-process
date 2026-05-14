# python verify_statitics.py \
#   --input DS-1.5B-Math_1.jsonl \
#   --output DS-1.5B-Math_1.png

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] JSON decode error at line {line_no}, skipped.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL (from verify_analysis.py)")
    ap.add_argument("--output", required=True, help="Path to output PNG figure")
    ap.add_argument(
        "--min_samples",
        type=int,
        default=1,
        help="Only plot difficulty bins with at least this many samples (default 1)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = load_jsonl(in_path)
    total = len(rows)

    # 1) filter: correct only
    correct_rows = [r for r in rows if r.get("correct_flag", False) is True]
    # 2) filter: answer_in_prompt must be False (missing treated as False)
    filtered_rows = [r for r in correct_rows if r.get("answer_in_prompt", False) is False]
    #filtered_rows = correct_rows
    # 3) filter: first next sentence exists
    filtered_rows = [r for r in filtered_rows if r.get("first_next_sentence", "") != ""]

    print("==== Summary ====")
    print(f"Total rows (responses): {total}")
    print(f"After correct_flag==True: {len(correct_rows)}")
    print(f"After answer_in_prompt==False: {len(filtered_rows)}")

    # group by difficulty
    sum_tokens: Dict[float, int] = defaultdict(int)
    sum_anscnt: Dict[float, int] = defaultdict(int)
    n_count: Dict[float, int] = defaultdict(int)

    skipped_no_diff = 0
    skipped_no_tokens = 0
    skipped_no_anscnt = 0

    max_length, min_length = 0, 8192

    for r in filtered_rows:
        d = safe_float(r.get("difficulty", None))
        if d is None:
            skipped_no_diff += 1
            continue

        tok = safe_int(r.get("tokens_from_first_answer_to_think_end", None))
        max_length = max(max_length, tok if tok is not None else 0)
        min_length = min(min_length, tok if tok is not None else 8192)
        if tok is None:
            skipped_no_tokens += 1
            continue

        cnt = safe_int(r.get("answer_sentence_count", None))
        if cnt is None:
            skipped_no_anscnt += 1
            continue

        sum_tokens[d] += tok
        sum_anscnt[d] += cnt
        n_count[d] += 1

    print(f"Skipped (no/invalid difficulty): {skipped_no_diff}")
    print(f"Skipped (no/invalid tokens_from_first_answer_to_think_end): {skipped_no_tokens}")
    print(f"Skipped (no/invalid answer_sentence_count): {skipped_no_anscnt}")

    points: List[Tuple[float, float, float, int]] = []
    for d in sorted(n_count.keys()):
        n = n_count[d]
        if n >= args.min_samples:
            mean_tok = sum_tokens[d] / n
            mean_cnt = sum_anscnt[d] / n
            points.append((d, mean_tok, mean_cnt, n))

    if not points:
        print("[ERROR] No valid points to plot (check filters/fields).")
        return

    xs = [p[0] for p in points]
    ys_tok = [p[1] for p in points]
    ys_cnt = [p[2] for p in points]
    ns = [p[3] for p in points]

    # Two subplots, shared x
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(8, 8)

    # subplot 1: tokens
    ax1.plot(xs, ys_tok, marker="o")
    ax1.set_ylabel("Avg tokens (first answer -> </think>)")
    ax1.grid(True)
    ax1.set_title("Correct only & answer_not_in_prompt: Avg length and answer mentions vs difficulty")

    for x, y, n in zip(xs, ys_tok, ns):
        ax1.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

    # subplot 2: answer sentence count
    ax2.plot(xs, ys_cnt, marker="o")
    ax2.set_xlabel("Difficulty")
    ax2.set_ylabel("Avg answer_sentence_count")
    ax2.grid(True)

    for x, y, n in zip(xs, ys_cnt, ns):
        ax2.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")
    print(f"Max Verification Length: {max_length}, Min Verification Length: {min_length}")


if __name__ == "__main__":
    main()
