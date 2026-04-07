#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "❌ 用法: bash dataset_run_verify.sh <BASE_ROOT_DIR> <STEP_LIST> [EXTRACTION] [WAY]"
  echo "例子: bash dataset_run_verify.sh /mnt/hdfs/if_au/saves/cky/eval_results/DS1.5B_8k_DAPO_base 50,100,150 self correct"
  exit 1
fi

BASE_ROOT_DIR="$1"
STEP_LIST_RAW="$2"
EXTRACTION="${3:-self}"
WAY="${4:-correct}"
MIN_SCORE=2

DATASETS=("aime" "aime25" "amc" "math" "minerva" "olympiad_bench")


IFS=',' read -r -a STEPS <<< "$STEP_LIST_RAW"

INPUT_FILES=()
DATASET_NAMES=()
STEP_NAMES=()

for STEP in "${STEPS[@]}"; do
  STEP="$(echo "$STEP" | xargs)"
  STEP_DIR="${BASE_ROOT_DIR}/global_step_${STEP}"

  if [ ! -d "$STEP_DIR" ]; then
    echo "⚠️  警告：找不到 step 目录: $STEP_DIR"
    continue
  fi

  for DATASET in "${DATASETS[@]}"; do
    INPUT_JSON="${STEP_DIR}/${DATASET}/results_details.json"
    if [ ! -f "$INPUT_JSON" ]; then
      echo "⚠️  警告：找不到输入文件: $INPUT_JSON"
    else
      INPUT_FILES+=("$INPUT_JSON")
      DATASET_NAMES+=("$DATASET")
      STEP_NAMES+=("$STEP")
    fi
  done
done

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
  echo "❌ 没有有效的输入文件"
  exit 1
fi

OUTPUT_DIR="${BASE_ROOT_DIR}/ast_step_analysis-${EXTRACTION}-${WAY}"

echo
echo "=============================="
echo "Base root: $BASE_ROOT_DIR"
echo "Steps: ${STEPS[*]}"
echo "Found ${#INPUT_FILES[@]} input files"
echo "Extraction: $EXTRACTION"
echo "Way: $WAY"
echo "Output dir: $OUTPUT_DIR"
echo "=============================="

python dataset_verify_analysis.py \
  --inputs "${INPUT_FILES[@]}" \
  --datasets "${DATASET_NAMES[@]}" \
  --steps "${STEP_NAMES[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --extraction "$EXTRACTION" \
  --way "$WAY" \
  --min_score "$MIN_SCORE"

echo
echo "🎉 All steps finished. Results saved to: $OUTPUT_DIR"