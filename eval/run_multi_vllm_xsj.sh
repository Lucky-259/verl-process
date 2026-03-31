#!/bin/bash
set -e

cd /opt/tiger/hqz_debug/cky/verl-process
# cd /mnt/luoyingfeng/changkaiyan/verl-process

pip install -U transformers==4.57.3
pip install -e .
pip install sentence_transformers
pip install -r requirements_process.txt

# 你的原脚本文件名
BASE_SCRIPT="eval/vllm_all_8k_xsj.sh"   # 改成你的实际脚本路径

# 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
JOBS=(
  "AVPO AVPO_v2-Qwen3-1.7B-Base-beta_0.01-lambda_0.01 16 100 120"
  "AVPO DAPO-Qwen3-1_7B-Base 16 100 340"
)

# # Baseline
# BASE_SCRIPT="eval/vllm_baseline_8k.sh"   # 改成你的实际脚本路径

# # 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
# JOBS=(
#   "DeepSeek-R1-Distill-Qwen-1.5B 16 200"
#   "DeepSeek-R1-Distill-Qwen-7B 16 200"
# )

for job in "${JOBS[@]}"; do
  echo "========================================"
  echo "Running: $job"
  echo "========================================"
  bash $BASE_SCRIPT $job
done
