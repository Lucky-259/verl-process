#!/bin/bash
set -e

cd /opt/tiger/hqz_debug/cky/verl-process
# cd /mnt/luoyingfeng/changkaiyan/verl-process

pip install -U transformers==4.57.3
pip install -e .
pip install sentence_transformers
pip install -r requirements_process.txt

# 你的原脚本文件名
BASE_SCRIPT="eval/vllm_all_8k.sh"   # 改成你的实际脚本路径

# 你可以在这里列出多组：PROJECT_NAME EXPERIMENT_NAME [REPEAT] [CONCURRENCY]
JOBS=(
  "Redundancy_DAPO DS1.5B_8k_redundancy_self_correct_1_5e-4_DAPO 16 150 250"
  # "Redundancy_DAPO DS1.5B_8k_redundancy_self_correct_1_5e-4_1_DAPO 16 150"
  # "Redundancy_DAPO DS7B_8k_redundancy_self_correct_1_5e-4_1_DAPO 16 150"
  # "Redundancy_DAPO DS1.5B_8k_DAPO_base 16 150"
  # "Redundancy_DAPO DS1.5B_8k_1e_4_4096_DAPO_l1 16 150"
  # "Redundancy_DAPO DS1.5B_8k_2e_4_7168_DAPO_l1 16 150"
  # "Redundancy_DAPO qwen3_1.7B_8k_redundancy_self_correct_1_5e-4_DAPO 16 150 50,100,150,200"
  # "Redundancy_DAPO qwen3_1.7B_8k_DAPO_base 16 150 50,100,150,200"
  # "Redundancy_DAPO DS7B_8k_DAPO_base 16 150 100,200,250,300"
  # "Redundancy_DAPO qwen3_8B_8k_redundancy_self_correct_1_5e-4_DAPO 16 150 50,100,150,200"
  # "Redundancy_DAPO DS1.5B_8k_redundancy_self_correct_1_5e-4_DAPO 16 150"
  # "Redundancy_DAPO DS7B_8k_redundancy_self_correct_1_2e-4_1_DAPO 16 150"
  # "Redundancy DS1.5B_8k_redundancy_correct_self_1_2e-4 16 150"
  # "Redundancy_new DS1.5B_8k_redundancy_correct_self_1_5e-4 16 150"
  # "Redundancy DS7B_8k_redundancy_correct_self_1_5e-4 16 150"
  # "Redundancy DS7B_8k_redundancy_correct_self_1_2e-4 16 150"
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
