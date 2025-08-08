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

import re
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from math_verify.errors import TimeoutException


def extract_answer_math(s):
    ans = s.split("boxed")
    if len(ans) == 1:
        return ans[0]
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    return a


def compute_score(model_output: str, ground_truth: str) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    # assert isinstance(ground_truth, str), f"[Error] ground_truth must be str, but got {type(ground_truth)}: {ground_truth}"
    ground_truth_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception as e:
        print(e)
    except TimeoutException as e:
        print(e)

    acc = bool(ret_score)
    reward = 1.0 if acc else -1.0

    return {"score": reward, "acc": acc, "pred": extract_answer_math(model_output)}
