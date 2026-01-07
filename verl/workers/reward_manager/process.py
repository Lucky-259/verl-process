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

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

# -----------------------
# Step split keywords
# -----------------------
SUMMARY_KEYWORDS = [
    "therefore,",
    "alright,",
    "hmm,",
    "actually,",
    "alternatively,",
    "first,",
    "second,",
    "then,",
    "next,",
    "finally,",
]

CONCLUSION_KEYWORDS = [
    "let me",
    "let's",
    "check",
    "double-check",
    "verify",
    "step by step",
    "we have",
]

# Regex patterns
THINK_PATTERN = re.compile(r"(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _extract_think_block(response_text: str) -> tuple[str | None, int | None, int | None, str, int | None]:
    """Split LRM output into think and answer by the </think> boundary.

    NOTE: LRM output does NOT include an explicit <think> start tag, only </think>.
    We treat everything before </think> as think_text and everything after as answer_text.

    Returns:
        think_text: substring before </think>
        think_start: char start index of think_text in response_text
        think_end: char end index (exclusive) of think_text in response_text
        answer_text: substring after </think> (stripped)
        think_close_end: char index right after </think> (end of the closing tag), or None
    """
    match = THINK_PATTERN.search(response_text)
    if not match:
        return None, None, None, response_text.strip(), None

    think_text = match.group(1)
    think_start = match.start(1)
    think_end = match.end(1)
    think_close_end = match.end(0)  # end index of the whole "(...)</think>"
    answer_text = response_text[think_close_end:].strip()
    return think_text, think_start, think_end, answer_text, think_close_end


def _is_sentence_boundary(char: str) -> bool:
    """
    Check if a character marks the end of a sentence.

    Args:
        char: Character to check

    Returns:
        True if the character is a sentence boundary marker
    """
    return char in ".!?\n"


def _find_step_boundaries(
    think_text: str,
    summary_keywords: list[str],
    conclusion_keywords: list[str],
    require_both: bool = True,
) -> list[int]:
    """
    Find STEP START positions in thinking text using keyword-based sentence analysis.

    We treat sentences that match keywords as the *start* of a step.
    A step is the paragraph between two step-start sentences:
        step_k: [start_k, start_{k+1})
    and the step_end will be (start_{k+1} - 1) in character index (inclusive).

    Args:
        think_text: The extracted thinking content
        summary_keywords: Keywords indicating summary/transition
        conclusion_keywords: Keywords indicating verification/conclusion
        require_both: Whether to require both types of keywords (AND logic)

    Returns:
        List of character indices (start-inclusive) marking step starts (relative to think_text)
    """
    if not think_text:
        return []

    print(f"[Think Text] {think_text}")
    summary_set = tuple(k.lower() for k in summary_keywords)
    conclusion_set = tuple(k.lower() for k in conclusion_keywords)

    step_starts: list[int] = []
    sentence_start = 0
    lower_text = think_text.lower()

    for i, char in enumerate(think_text):
        if not _is_sentence_boundary(char):
            continue

        sentence_end = i + 1  # end-exclusive index
        sentence = lower_text[sentence_start:sentence_end]

        has_summary = any(keyword in sentence for keyword in summary_set)
        has_conclusion = any(keyword in sentence for keyword in conclusion_set)

        # STEP START detection
        if require_both:
            if has_summary and has_conclusion:
                print(f"[Sentence Start] {sentence_start}")
                step_starts.append(sentence_start)
        else:
            if has_summary or has_conclusion:
                print(f"[Sentence Start] {sentence_start}")
                step_starts.append(sentence_start)

        sentence_start = sentence_end

    if step_starts:
        step_starts = sorted(set(step_starts))

    if not step_starts:
        return [0]

    # Ensure coverage from beginning
    if step_starts[0] != 0:
        step_starts = [0] + step_starts

    print(f"[Step Starts] {step_starts}")

    return step_starts


def _find_token_at_char_position(
    offsets: list[tuple[int, int]],
    char_position: int,
) -> int:
    """
    Find the token index that contains or immediately precedes a character position.

    This function handles the alignment between character positions and token indices
    using the offset mapping from the tokenizer.

    Args:
        offsets: List of (start_char, end_char) tuples for each token.
                 Assumes half-open intervals [start, end).
        char_position: Character position to locate

    Returns:
        Token index. Returns 0 if char_position is before all tokens,
        or len(offsets)-1 if beyond all tokens.
    """
    if not offsets:
        return 0

    # First pass: Find token that contains the character position
    # Using [start, end) convention where start <= char_pos < end
    for i, (start, end) in enumerate(offsets):
        if start <= char_position < end:
            return i

    # Second pass: Find the last token that ends at or before char_position
    # This handles cases where char_position falls between tokens
    for i in range(len(offsets) - 1, -1, -1):
        if offsets[i][1] <= char_position:
            return i

    # Edge case: char_position is before all tokens
    return 0


def _build_tagged_solution(
    response_text: str,
    think_start: int,
    think_end: int,
    step_boundaries: list[int],
    answer_text: str = "",
    include_answer: bool = False,
    think_close_end: int | None = None,
) -> str:
    """
    Wrap each step span in the thinking block with <step_i>...</step_i> tags.

    NOTE: step_boundaries are STEP START positions (relative to think_text).
    Steps are defined as:
        [start_k, start_{k+1}) ... and the last one [start_last, end)
    """
    if not step_boundaries:
        return response_text

    think_text = response_text[think_start:think_end]
    starts = step_boundaries  # step starts (relative to think_text)
    segments: list[str] = []

    for step_num, start in enumerate(starts, start=1):
        end = starts[step_num] if step_num < len(starts) else len(think_text)  # end-exclusive
        if end <= start:
            continue
        segment = think_text[start:end]
        tagged_segment = f"<step_{step_num}>{segment}</step_{step_num}>"
        segments.append(tagged_segment)

    wrapped_think = "".join(segments)

    # Default behavior: keep original suffix (including </think> and any raw answer text)
    suffix = response_text[think_end:]

    if include_answer:
        # Determine where </think> ends (if not provided)
        close_end = think_close_end
        if close_end is None:
            m = re.search(r"</think>", response_text[think_end:], flags=re.IGNORECASE)
            close_end = think_end + m.end() if m else think_end

        closing = response_text[think_end:close_end]
        if answer_text.strip():
            suffix = closing + "\n<answer>\n" + answer_text.strip() + "\n</answer>\n"
        else:
            suffix = closing

    return response_text[:think_start] + wrapped_think + suffix


@register("grpo_process")
class ProcessRewardManager:
    """
    Reward manager for mathematical reasoning with process-level rewards.

    This manager:
    1. Extracts <think> blocks from model responses
    2. Identifies step STARTs using keyword matching
    3. Converts step ENDs (next_start-1) to token indices via offset mapping
    4. Calls reward function to get both outcome and process scores
    5. Assigns scores to appropriate token positions

    Args:
        tokenizer: Tokenizer for encoding/decoding and offset mapping
        num_examine: Number of samples to print for debugging per data source
        compute_score: Function to compute rewards (outcome + process scores)
        reward_fn_key: Key in non_tensor_batch to identify data source
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        print(f"[ProcessNaiveRewardManager] Initialized with num_examine={num_examine}")

    def _extract_reward_from_rm_scores(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any] | None:
        """
        Extract reward from already-computed rm_scores if available.
        This is used when use_reward_loop=True and rewards are already computed during generate_sequences.

        Args:
            data: DataProto object containing the batch data
            return_dict: Whether to return a dictionary with reward_tensor and reward_extra_info

        Returns:
            If rm_scores exists:
                - If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
                - If return_dict=False: torch.Tensor of rm_scores
            If rm_scores doesn't exist: None
        """
        if "rm_scores" not in data.batch.keys():
            return None

        if return_dict:
            reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        else:
            return data.batch["rm_scores"]

    def __call__(
        self,
        data: DataProto,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """
        Compute rewards for a batch of data.

        Args:
            data: DataProto containing batch data
            return_dict: If True, return dict with reward_tensor and extra_info

        Returns:
            Reward tensor or dict containing reward_tensor and reward_extra_info
        """
        # Check if rewards are already provided
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            print("[ProcessNaiveRewardManager] Using pre-computed rm_scores")
            return reward_from_rm_scores

        batch_size = len(data)
        print(f"\n[ProcessNaiveRewardManager] Processing batch of size {batch_size}")

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Track printed samples per data source for debugging
        printed_count_by_source: dict[str, int] = {}

        for i in range(batch_size):
            item = data[i]

            # Extract prompt and response
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            attention_mask = item.batch["attention_mask"]

            valid_prompt_len = int(attention_mask[:prompt_len].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[:0]

            response_ids = item.batch["responses"]
            valid_response_len = int(attention_mask[prompt_len:].sum().item())
            print(f"[Response Length] {valid_response_len}")
            valid_response_ids = response_ids[:valid_response_len] if valid_response_len > 0 else response_ids[:0]

            # Decode to text
            prompt_text = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_text = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            # Extract metadata
            ntb = item.non_tensor_batch
            ground_truth = ntb.get("reward_model", {}).get("ground_truth")
            data_source = ntb.get(self.reward_fn_key, "unknown")
            extra_info = ntb.get("extra_info", {}).copy()

            # -------- Step boundary detection and token alignment --------
            step_mask = torch.zeros_like(data.batch["responses"][i], dtype=torch.bool, device=response_ids.device)
            think_mask = torch.zeros_like(data.batch["responses"][i], dtype=torch.bool, device=response_ids.device)
            step_end_token_indices: list[int] = []
            tagged_solution = response_text  # ?

            # Extract and process <think> block
            think_text, think_start, think_end, answer_text, think_close_end = _extract_think_block(response_text)
            print(f"[Think Start] {think_start}")
            print(f"[Think End] {think_end}")

            if think_text and think_start is not None and think_end is not None:
                if valid_response_len == 0:
                    print(f"[Sample {i}] Warning: valid_response_len is 0, skipping step detection")
                else:
                    # Find STEP STARTS (character-level, relative to think_text)
                    step_starts = _find_step_boundaries(
                        think_text,
                        SUMMARY_KEYWORDS,
                        CONCLUSION_KEYWORDS,
                        require_both=True,
                    )

                    if step_starts:
                        print(f"[Sample {i}] Found {len(step_starts)} step START positions")

                        # Build tagged solution for reward model
                        tagged_solution = _build_tagged_solution(
                            response_text, think_start, think_end, step_starts,
                            answer_text=answer_text,
                            include_answer=True,
                            think_close_end=think_close_end,
                        )

                        # Get token-level offset mapping
                        try:
                            encoding = self.tokenizer(
                                response_text,
                                add_special_tokens=False,
                                return_offsets_mapping=True,
                            )
                            offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
                            # Truncate to valid response length
                            offsets = offsets[:valid_response_len]

                            # Build think_mask: tokens whose char-span overlaps [think_start, think_end)
                            think_mask = torch.zeros_like(data.batch["responses"][i], dtype=torch.bool, device=response_ids.device)
                            for t, (s, e) in enumerate(offsets):
                                # overlap condition: [s,e) intersects [think_start,think_end)
                                if e > think_start and s < think_end:
                                    think_mask[t] = True

                            # Convert STEP ENDs (next_start-1; last ends at len(think_text)-1) to token indices
                            for k in range(len(step_starts)):
                                if k + 1 < len(step_starts):
                                    end_char_rel = step_starts[k + 1] - 1  # inclusive char index in think_text
                                else:
                                    end_char_rel = len(think_text) - 1  # last char in think_text

                                if end_char_rel < 0:
                                    continue

                                char_position_global = think_start + end_char_rel
                                token_idx = _find_token_at_char_position(offsets, char_position_global)

                                # Ensure token index is within valid range
                                token_idx = min(token_idx, valid_response_len - 1)
                                step_end_token_indices.append(token_idx)
                                step_mask[token_idx] = True

                            step_mask &= think_mask
                            print(f"[Sample {i}] Step end token indices: {step_end_token_indices}")

                        except Exception as e:
                            print(f"[Sample {i}] Error in token alignment: {e}")
                            step_end_token_indices = []
                    else:
                        print(f"[Sample {i}] No step START found in think block")
            else:
                print(f"[Sample {i}] No <think> block found in response")

            # -------- Call reward function --------
            extra_info.update(
                {
                    "tagged_solution": tagged_solution,
                    "step_count": len(step_end_token_indices),
                    "question": prompt_text,
                }
            )

            try:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            except Exception as e:
                print(f"[Sample {i}] Error in compute_score: {e}")
                score = 0.0

            # Parse score results
            if isinstance(score, dict):
                final_score = float(score.get("score", 0.0))
                step_scores = list(score.get("step_scores", []))
                process_used = bool(score.get("process_used", True))
            else:
                final_score = float(score)
                step_scores = []
                process_used = False

            print(f"[Sample {i}] Final score: {final_score}, Step scores: {step_scores}")

            # -------- Assign scores to tokens --------
            # Assign process scores to step-end tokens
            if step_scores and step_end_token_indices:
                num_steps = min(len(step_scores), len(step_end_token_indices))
                for j in range(num_steps):
                    token_idx = step_end_token_indices[j]
                    reward_tensor[i, token_idx] = float(step_scores[j])

                if len(step_scores) != len(step_end_token_indices):
                    print(
                        f"[Sample {i}] Warning: Mismatch between step_scores "
                        f"({len(step_scores)}) and step_end_indices "
                        f"({len(step_end_token_indices)})"
                    )

            # Assign final score to the last valid token
            if valid_response_len > 0:
                reward_tensor[i, valid_response_len - 1] += float(final_score)
            else:
                print(f"[Sample {i}] Warning: Cannot assign final score, valid_response_len=0")

            # Store step_masks / think_masks for process-level advantage computation
            reward_extra_info["step_masks"].append(step_mask.detach().cpu().numpy().astype(bool))
            reward_extra_info["think_masks"].append(think_mask.detach().cpu().numpy().astype(bool))
            reward_extra_info["process_used"].append(bool(process_used))

            # -------- Debug output --------
            print(f"[Reward Tensor NonZero Indexes] {torch.nonzero(reward_tensor[i], as_tuple=False).flatten().tolist()}")
            printed = printed_count_by_source.get(data_source, 0)
            if printed < self.num_examine:
                printed_count_by_source[data_source] = printed + 1
                print(f"\n{'='*60}")
                print(f"[DEBUG SAMPLE {i}] Data source: {data_source}")
                print(f"[Prompt] {prompt_text}")
                print(f"[Response] {response_text}")
                print(f"[Ground Truth] {ground_truth}")
                print(f"[Final Score] {final_score}")
                if step_scores:
                    print(f"[Step Scores] {step_scores}")
                if step_end_token_indices:
                    print(f"[Step Token Indices] {step_end_token_indices}")
                if reward_tensor.numel() > 0:
                    print(f"[Reward Tensor] {reward_tensor[i]}")
                print(f"{'='*60}\n")

        print(f"[ProcessNaiveRewardManager] Batch processing complete\n")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
