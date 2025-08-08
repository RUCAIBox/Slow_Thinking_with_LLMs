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

import os
import json
from datetime import datetime
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict


class NaiveLogRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        log_dir="logs",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir,
            f"reward_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )

        if self.overlong_buffer_cfg is not None:
            assert (
                self.max_resp_len is not None
            ), f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    # TODO: Is this still necessary in algorithms other than PRIME?
    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch["acc"] = torch.tensor(
            scores, dtype=torch.float32, device=prompt_ids.device
        )
        return scores

    def __call__(self, data: DataProto, return_dict: bool = False,save_data: bool=False,step: int=0):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # print("*****************************************")
            # print(valid_response_ids.shape)
            # valid_response_ids=[int(x) for x in valid_response_ids]
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            # print("*****************************************")
            # print(response_str)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            import re
            # def contains_chinese_or_garbage(text):
            #     # 匹配中文字符的正则表达式
            #     chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
            #     # # 匹配非ASCII字符的正则表达式（可能是乱码）
            #     # garbage_pattern = re.compile(r'[^\x00-\x7F]')
                
            #     contains_chinese = bool(chinese_pattern.search(text))
            #     # contains_garbage = bool(garbage_pattern.search(text))
            #     # print("Yes")
            #     # return contains_chinese or contains_garbage
            #     return contains_chinese
            def contains_chinese_or_garbage(text):
                # 匹配中文字符的正则表达式
                chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
                # 检查是否出现\\box 2次以上
                box_pattern = re.compile(r'boxed{')
                # 检查字符是否连续出现500次以上
                repeated_char_pattern = re.compile(r'(.)\1{499,}')  # .匹配任何字符，\1{499,}表示该字符连续出现500次及以上
                
                contains_chinese = bool(chinese_pattern.search(text))
                contains_box = len(box_pattern.findall(text.lower())) >= 5
                contains_repeated_char = bool(repeated_char_pattern.search(text))
                # 如果包含中文、出现了两次以上的\\box，或者有字符连续出现500次以上，返回True
                return contains_chinese #or contains_box or contains_repeated_char
            import json
            import re
            from collections import Counter
            def redundancy_3gram(text):
                count = 1
                for i in range(1, len(text.split())):
                    if tokens[i] == tokens[i - 1]:
                        count += 1
                        if count >= 500:
                            return True
                    else:
                        count = 1
                tokens = text.split()
                trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
                trigram_counts = Counter(trigrams)
                total = len(trigrams)+1
                redundant_count = sum(count for count in trigram_counts.values() if count > 1)
                return redundant_count / total >0.35 
            # if contains_chinese_or_garbage(response_str):# or redundancy_3gram(response_str)>0.35:
            #     result['score']=-1
            #     result['acc']=False
            # print("********************************************")
            # print(result)
            # print("********************************************")
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score 

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(
                    -exceed_len / overlong_buffer_len * overlong_penalty_factor, 0
                )
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
            # Log the data to JSONL file
            log_entry = {
                "prompt": prompt_str,
                "response": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "score": score if not isinstance(result, dict) else result,
            }
            if save_data:
                save_path=os.path.join(
                self.log_dir,
                f"test_logs_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            )
                with open(save_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
