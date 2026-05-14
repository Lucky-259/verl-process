import re
from math_verify import parse, verify, ExprExtractionConfig, LatexExtractionConfig

# Parse the gold and answer
# If you know that gold will only contain latex or expr (no latex env), use
# parse(gold, extraction_config=[LatexExtractionConfig()]) or parse(gold, extraction_config=[ExprExtractionConfig()])

boxed_answer = "\\boxed{\\dfrac{5}{6}}"
gold = parse(boxed_answer, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
print(gold)

# 示例文本
sentence = "Therefore, the minimal possible value of P(X=0) is 5/6."
answer = parse(sentence, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
print(answer)

# Order here is important!
print(verify(gold, answer))
# >>> True

import sys
# sys.path.append('/mnt/luoyingfeng/changkaiyan/verl-process')
sys.path.append('/mnt/nvme1/luoyingfeng/lucky/verl-process')
from deepscaler.rewards.math_utils.utils import grade_answer_verl
print(answer and grade_answer_verl("\\boxed{" + str(answer[-1]) + "}", boxed_answer))