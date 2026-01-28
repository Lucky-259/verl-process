#!/usr/bin/env bash
set -e

# 获取所有参数
ALL_ARGS=("$@")
NUM_ARGS=$#

# 提取最后两个参数
BASE_DIR="${ALL_ARGS[0]}"
MODEL_PREFIX="${ALL_ARGS[1]}"
EXTRACTION="${ALL_ARGS[$((NUM_ARGS-2))]}"
WAY="${ALL_ARGS[$((NUM_ARGS-1))]}"
MIN_SCORE=2

if [ "$#" -ne 4 ]; then
  echo "❌ 用法: ./run_verify.sh <INPUT_PATH> <OUTPUT_PATH> <EXTRACTION> <WAY>"
  echo "例子: ./run_verify.sh eval/baseline_res/DeepSeek-R1-Distill-Qwen-1.5B✓ eval_results/analysis/DeepSeek-R1-Distill-Qwen-1.5B✓ self correct"
  exit 1
fi

# 提取数据集名称（除了最后两个参数）
DATASETS=("aime" "aime25" "amc" "math" "minerva" "olympiad_bench")

# 构建输入文件路径和数据集名称数组
INPUT_FILES=()
DATASET_NAMES=()

for DATASET in "${DATASETS[@]}"; do
  INPUT_JSON="${BASE_DIR}/${DATASET}/results_details.json"
  if [ ! -f "$INPUT_JSON" ]; then
    echo "⚠️  警告：找不到输入文件: $INPUT_JSON"
    echo "  将跳过此数据集"
  else
    INPUT_FILES+=("$INPUT_JSON")
    DATASET_NAMES+=("$DATASET")
  fi
done

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
  echo "❌ 没有有效的输入文件"
  exit 1
fi

echo
echo "=============================="
echo "Processing ${#DATASET_NAMES[@]} datasets"
echo "Datasets: ${DATASET_NAMES[@]}"
echo "Extraction: $EXTRACTION"
echo "Way: $WAY"
echo "=============================="

OUTPUT_DIR="${MODEL_PREFIX}-${EXTRACTION}-${WAY}"

echo "▶ Running verify_analysis.py for all datasets"
python dataset_verify_analysis.py \
  --inputs "${INPUT_FILES[@]}" \
  --datasets "${DATASET_NAMES[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --extraction "$EXTRACTION" \
  --way "$WAY" \
  --min_score "$MIN_SCORE"

echo
echo "🎉 All datasets finished. Results saved to: $OUTPUT_DIR"