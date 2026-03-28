#!/bin/bash
set -e

cd /opt/tiger/hqz_debug/cky/verl-process
# cd /mnt/luoyingfeng/changkaiyan/verl-process

pip install -U transformers==4.57.3
pip install -e .
pip install sentence_transformers
pip install -r requirements_process.txt

# # 你的原脚本文件名
# BASE_SCRIPT="eval/vllm_all_8k.sh"   # 改成你的实际脚本路径

# # 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
# JOBS=(
#   "Redundancy DS_test 2 150"
#   "Redundancy DS_test_1 2 150"
# )

# Baseline
BASE_SCRIPT="eval/vllm_baseline_8k.sh"   # 改成你的实际脚本路径

# 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
JOBS=(
  "AdaptThink-1.5B-delta0.05 16 200"
  "L1-Qwen-1.5B-Max 16 200"
  "alpha_0.1_DeepSeek-R1-Distill-Qwen-1.5B 16 200"
  "DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k 16 200"
  "Laser-DE-L4096-1.5B 16 200"
  "Laser-L8192-1.5B 16 200"
  "DLER-R1-1.5B-Research 16 200"
)

for job in "${JOBS[@]}"; do
  echo "========================================"
  echo "Running: $job"
  echo "========================================"
  bash $BASE_SCRIPT $job
done
