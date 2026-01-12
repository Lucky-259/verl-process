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

from collections import defaultdict

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.utils.reward_score import math

import torch

@register("shorter")
class ShorterRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, num_generation, compute_score=None, reward_fn_key="data_source", beta=0.001) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.num_generation = num_generation
        self.reward_fn_key = reward_fn_key
        self.beta = beta

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores
    

    def check_correctness_and_length(self, data):
        """in GRPO, the input data here should have len=train_batch_size * num_generations(i.e. config.actor_rollout_ref.rollout.n)"""
        num_generation = self.num_generation
        
        assert len(data) % num_generation == 0, f"Implementation Error: the input data len(data)={len(data)} does not divide num_generation={num_generation} evenly."
        
        # Extract unique problem IDs to group responses
        problem_ids = [item.non_tensor_batch['index'] for item in data] # ids of problems of this {batch*rollout_n}
        unique_problem_ids = []
        for pid in problem_ids:
            if pid not in unique_problem_ids:
                unique_problem_ids.append(pid) # ids of problem of this batch (unique)
        
        # Create a mapping from data index to results
        correctness_map = {}
        length_map = {}
        optimal_length_map = {}
        
        # Process each unique problem
        for problem_id in unique_problem_ids:
            # Find all instances of this problem in the batch
            indices = [i for i in range(len(data)) if data[i].non_tensor_batch['index'] == problem_id]
            completions_given_prompt = [data[i] for i in indices]
            
            # Should have num_generation repetitions of each problem
            assert len(completions_given_prompt) == num_generation, f"Expected {num_generation} repetitions for problem {problem_id}, got {len(completions_given_prompt)}"
            
            prompts = []  # they should be the same
            response_lengths_given_prompt = [] # the response length of each y_j  ( l(y_j) )
            correctnesses_given_prompt = [] # the correctnesses of each y_j
            
            for idx, completion in zip(indices, completions_given_prompt):
                prompt_ids = completion.batch['prompts'] # the token ids of each same prompt for this problem
                
                prompt_length = prompt_ids.shape[-1]
                
                valid_prompt_length = completion.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:] # valid token ids of each same prompt for this problem
                
                response_ids = completion.batch['responses']
                valid_response_length = completion.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length] # valid token ids of each same response for this problem
                
                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                
                ground_truth = completion.non_tensor_batch['reward_model']['ground_truth'] # get the ground truth of this problem
                
                # check correctness
                correct_or_not = math.compute_score(response_str, ground_truth)
                correctnesses_given_prompt.append(correct_or_not)
                response_lengths_given_prompt.append(valid_response_length)
                
                # Store in the map
                correctness_map[idx] = correct_or_not
                length_map[idx] = valid_response_length
                
                prompts.append(prompt_str) # the prompts str of each y_j, this should be same
            
            # the shortest correct response length is selected as optimal length
            correct_lengths = [l for c, l in zip(correctnesses_given_prompt, response_lengths_given_prompt) if c]


            # --------------------------------------------------
            # TEST: WHAT IF JUST CHOOSE THE MIN LENGTH, REGRARDLESS OF CORRECTNESS?
            optimal_length = min(correct_lengths) if correct_lengths else sum(response_lengths_given_prompt)/len(response_lengths_given_prompt)
            # optimal_length = min(response_lengths_given_prompt) else avg
            # --------------------------------------------------



            # Just print a compact summary line for each problem
            correct_count = sum(correctnesses_given_prompt)
            print(f"Lengths={response_lengths_given_prompt}, Correct={correct_lengths}", flush=True)
            
            # Store optimal length for all instances of this problem
            for idx in indices:
                optimal_length_map[idx] = optimal_length # same problem_id, same opt_length
            
            assert len(set(prompts)) == 1, f"Implementation Error: in this given dataproto prompts should be the same, while prompts={prompts}"
        
        
        # Convert maps back to lists in the original order
        correctness_set = [correctness_map[i] for i in range(len(data))]
        length_set = [length_map[i] for i in range(len(data))]
        optimal_length_set = [optimal_length_map[i] for i in range(len(data))]
        
        assert len(correctness_set) == len(length_set) == len(optimal_length_set) == len(data), f"Implementation Error: the length of correctness_set={len(correctness_set)}, length_set={len(length_set)}, optimal_length_set={len(optimal_length_set)}, data={len(data)}, they should all be the same length"
        
        return correctness_set, length_set, optimal_length_set
    
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        # Check correctness and length for each sample in batch
        correctness_set, length_set, optimal_length_set = self.check_correctness_and_length(data)
        # compute the reward r(y_j)
        def sb_compute_score(optimal_length, completion_length, correct_or_not):
            """return the reward score based on sb func; optimal length is the shortest length among all correct completions"""
            alpha = 2.0
            beta = self.beta
            if correct_or_not == True:
                correctness = alpha
            else:
                correctness = 0.0
            length_gap = abs(completion_length - optimal_length) * beta

            return correctness - length_gap

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            
            correct_or_not = correctness_set[i] # correctness: I(y_j=y_i*)
            completion_length = length_set[i] # l(y_j)
            optimal_length = optimal_length_set[i] # l^SOL(G(x_i))

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            # attention mask is (prompt, response)
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum() # Count real tokens (not 0) in attn_mask
            valid_prompt_ids = prompt_ids[-valid_prompt_length:] # valid prompt token ids (prompt is left padding)

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() # Count real tokens (not 0) in attn_mask
            valid_response_ids = response_ids[:valid_response_length] # valid response token ids (response is right padding)

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = sb_compute_score(optimal_length, completion_length, correct_or_not)

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = score

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
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
