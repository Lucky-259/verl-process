# bash run_verify.sh AIME24-16K AMC MATH500 self correct

#!/usr/bin/env bash
set -e

MODEL_PREFIX="../eval_results/analysis/DS-1.5B"
BASE_DIR="../eval_results/DS-1.5B"
MIN_SCORE=2

if [ "$#" -eq 0 ]; then
  echo "❌ 用法: ./run_verify.sh <DATASET1> [DATASET2 ...]"
  echo "例子: ./run_verify.sh AIME24-16K AMC12"
  exit 1
fi

DATASETS=("${@:1:$(($#-2))}")
EXTRACTION="${@: -2:1}"
WAY="${@: -1}"

for DATASET in "$DATASETS"; do
  INPUT_JSON="${BASE_DIR}/${DATASET}/results_details.json"
  MID_JSONL="${MODEL_PREFIX}-${DATASET}-${EXTRACTION}-${WAY}/details.jsonl"
  OCC_PNG="${MODEL_PREFIX}-${DATASET}-${EXTRACTION}-${WAY}/occurence.png"
  OUT_PNG="${MODEL_PREFIX}-${DATASET}-${EXTRACTION}-${WAY}/length_line.png"
  PRO_BAR_PNG="${MODEL_PREFIX}-${DATASET}-${EXTRACTION}-${WAY}/proportion_bar.png"
  LEN_SCA_PNG="${MODEL_PREFIX}-${DATASET}-${EXTRACTION}-${WAY}/length_heatmap.png"
  PRO_SCA_PNG="${MODEL_PREFIX}-${DATASET}-${EXTRACTION}-${WAY}/proportion_heatmap.png"

  echo
  echo "=============================="
  echo "Dataset: $DATASET"
  echo "=============================="

  if [ ! -f "$INPUT_JSON" ]; then
    echo "⚠️  跳过（找不到输入文件）: $INPUT_JSON"
    continue
  fi

  if [ -f "$MID_JSONL" ]; then
    echo "⏭️  已存在，跳过 analysis: $MID_JSONL"
  else
    echo "▶ Running verify_analysis.py"
    python verify_analysis.py \
      --input "$INPUT_JSON" \
      --output "$MID_JSONL" \
      --occ "$OCC_PNG" \
      --proportion_bar "$PRO_BAR_PNG" \
      --length_scatter "$LEN_SCA_PNG" \
      --proportion_scatter "$PRO_SCA_PNG" \
      --extraction "$EXTRACTION" \
      --way "$WAY" \
      --min_score "$MIN_SCORE"
  fi

  echo "▶ Running verify_statitics.py"
  python verify_statitics.py \
    --input "$MID_JSONL" \
    --output "$OUT_PNG"

  echo "✅ Done: $DATASET"
done

echo
echo "🎉 All datasets finished."
