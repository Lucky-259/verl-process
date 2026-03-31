#!/bin/bash
set -e

cd /opt/tiger/hqz_debug/cky/verl-process
# cd /mnt/luoyingfeng/changkaiyan/verl-process

pip install -U transformers==4.57.3
pip install -e .
pip install sentence_transformers
pip install -r requirements_process.txt

# 你的原脚本文件名
BASE_SCRIPT="eval/code_eval/example_scripts/eval_code_all_csv.sh"   # 改成你的实际脚本路径

# 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
JOBS=(
  "Redundancy_DAPO DS1.5B_8k_redundancy_self_correct_1_5e-4_DAPO 250"
  "Redundancy_DAPO DS7B_8k_redundancy_self_correct_1_5e-4_1_DAPO 250"
  "Redundancy_DAPO DS1.5B_8k_DAPO_base 100,200,250,300"
  "Redundancy_DAPO DS1.5B_8k_1e_4_4096_DAPO_l1 100,200,250,300"
  "Redundancy_DAPO DS1.5B_8k_2e_4_7168_DAPO_l1 100,200,250,300"
  "Redundancy_DAPO qwen3_1.7B_8k_redundancy_self_correct_1_5e-4_DAPO 100,200,250,300"
  "Redundancy_DAPO qwen3_1.7B_8k_DAPO_base 100,200,250,300"
  "Redundancy_DAPO qwen3_8B_8k_redundancy_self_correct_1_5e-4_DAPO 100,200,250,300"
  "Redundancy_DAPO DS7B_8k_DAPO_base 100,200,250,300"
)

for job in "${JOBS[@]}"; do
  echo "========================================"
  echo "Running: $job"
  echo "========================================"
  bash $BASE_SCRIPT $job
done
