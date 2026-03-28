#!/bin/bash
set -e

cd /opt/tiger/hqz_debug/cky/verl-process
# cd /mnt/luoyingfeng/changkaiyan/verl-process

pip install -U transformers==4.57.3
pip install -e .
pip install sentence_transformers
pip install -r requirements_process.txt

# 你的原脚本文件名
BASE_SCRIPT="eval/code_eval/example_scripts/eval_code_baseline.sh"   # 改成你的实际脚本路径

# 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
JOBS=(
  "DeepSeek-R1-Distill-Qwen-1.5B"
  "Qwen3-1.7B"
  "L1-Qwen-1.5B-Max"
  "DLER-R1-1.5B-Research"
  "AdaptThink-1.5B-delta0.05"
  "alpha_0.1_DeepSeek-R1-Distill-Qwen-1.5B"
  "DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k"
  "Laser-DE-L4096-1.5B"
  "Laser-L8192-1.5B"
  "DeepSeek-R1-Distill-Qwen-7B"
)

for job in "${JOBS[@]}"; do
  echo "========================================"
  echo "Running: $job"
  echo "========================================"
  bash $BASE_SCRIPT $job
done
