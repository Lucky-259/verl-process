#!/bin/bash
set -e

cd /opt/tiger/hqz_debug/cky/verl-process
# cd /mnt/luoyingfeng/changkaiyan/verl-process

pip install -e .
pip install sentence_transformers

# 你的原脚本文件名
BASE_SCRIPT="eval/vllm_all_8k.sh"   # 改成你的实际脚本路径

# 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
JOBS=(
  "Redundancy DS1.5B_8k_redundancy_ratio_2_self 16 200"
  "ShorterBetter DS1.5B_8k_shorter_1e-3 16 200"
  "ShorterBetter DS1.5B_16k_shorter_1e-4 16 200"
)

for job in "${JOBS[@]}"; do
  echo "========================================"
  echo "Running: $job"
  echo "========================================"
  bash $BASE_SCRIPT $job
done
