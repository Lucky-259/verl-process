#!/usr/bin/env bash
set -e

MODEL_PREFIX="../eval_results/rollout_analysis/Qwen3-8B✓"
BASE_DIR="../../eval/baseline_res/Qwen3-8B✓"
MIN_SCORE=2
TOKENIZER_PATH="/mnt/nvme1/luoyingfeng/lucky/model_card/DeepSeek-R1-Distill-Qwen-1.5B"
# 是否只统计正确样本的token数（true/false）
TOKENS_CORRECT_ONLY=false

if [ "$#" -lt 4 ]; then
  echo "❌ 用法: ./run_verify.sh <DATASET1> <DATASET2> ... <EXTRACTION> <WAY>"
  echo "例子: ./run_verify.sh AIME24-16K AMC12 self base"
  echo
  echo "可选参数可以通过环境变量设置:"
  echo "  TOKENIZER_PATH: tokenizer路径 (默认: $TOKENIZER_PATH)"
  echo "  TOKENS_CORRECT_ONLY: 是否只统计正确样本 (默认: $TOKENS_CORRECT_ONLY)"
  echo
  echo "环境变量使用示例:"
  echo "  TOKENS_CORRECT_ONLY=true TOKENIZER_PATH=/path/to/tokenizer ./run_verify.sh AIME24 AMC12 self base"
  exit 1
fi

# 允许通过环境变量覆盖配置
if [ ! -z "$ENV_TOKENIZER_PATH" ]; then
  TOKENIZER_PATH="$ENV_TOKENIZER_PATH"
  echo "使用环境变量TOKENIZER_PATH: $TOKENIZER_PATH"
fi

if [ ! -z "$ENV_TOKENS_CORRECT_ONLY" ]; then
  TOKENS_CORRECT_ONLY="$ENV_TOKENS_CORRECT_ONLY"
  echo "使用环境变量TOKENS_CORRECT_ONLY: $TOKENS_CORRECT_ONLY"
fi

# 获取所有参数
ALL_ARGS=("$@")
NUM_ARGS=$#

# 提取最后两个参数
EXTRACTION="${ALL_ARGS[$((NUM_ARGS-2))]}"
WAY="${ALL_ARGS[$((NUM_ARGS-1))]}"

# 提取数据集名称（除了最后两个参数）
DATASETS=("${ALL_ARGS[@]:0:$((NUM_ARGS-2))}")

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
echo "配置信息"
echo "=============================="
echo "处理数据集数量: ${#DATASET_NAMES[@]}"
echo "数据集: ${DATASET_NAMES[@]}"
echo "答案提取方式: $EXTRACTION"
echo "检测方法: $WAY"
echo "最小分数阈值: $MIN_SCORE"
echo "Tokenizer路径: $TOKENIZER_PATH"
echo "只统计正确样本: $TOKENS_CORRECT_ONLY"
echo "=============================="

OUTPUT_DIR="${MODEL_PREFIX}-${EXTRACTION}-${WAY}"

echo
echo "▶ 运行命令:"
echo "python draw_dataset_rollout_pictures.py \\"

# 显示inputs参数
echo -n "  --inputs"
for file in "${INPUT_FILES[@]}"; do
  echo -n " \"$file\""
done
echo " \\"

# 显示datasets参数
echo -n "  --datasets"
for dataset in "${DATASET_NAMES[@]}"; do
  echo -n " \"$dataset\""
done
echo " \\"

# 显示其他参数
echo "  --output_dir \"$OUTPUT_DIR\" \\"
echo "  --extraction \"$EXTRACTION\" \\"
echo "  --way \"$WAY\" \\"
echo "  --min_score \"$MIN_SCORE\" \\"
echo "  --tokenizer_path \"$TOKENIZER_PATH\""
if [ "$TOKENS_CORRECT_ONLY" = "true" ]; then
  echo "  --tokens_correct_only"
fi
echo

echo
echo "开始执行..."
echo

# 执行命令
python draw_dataset_rollout_pictures.py \
  --inputs "${INPUT_FILES[@]}" \
  --datasets "${DATASET_NAMES[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --extraction "$EXTRACTION" \
  --way "$WAY" \
  --min_score "$MIN_SCORE" \
  --tokenizer_path "$TOKENIZER_PATH" \
  $([ "$TOKENS_CORRECT_ONLY" = "true" ] && echo "--tokens_correct_only")

echo
echo "=============================="
echo "输出结果统计"
echo "=============================="
echo "主输出目录: $OUTPUT_DIR"
echo

# 显示各数据集的输出结构
for DATASET in "${DATASET_NAMES[@]}"; do
  DATASET_DIR="$OUTPUT_DIR/$DATASET"
  if [ -d "$DATASET_DIR" ]; then
    echo "📁 数据集 '$DATASET' 输出:"
    
    # 查找该数据集的所有JSONL文件
    JSONL_FILES=$(find "$DATASET_DIR" -name "details_*.jsonl" -type f | sort)
    if [ ! -z "$JSONL_FILES" ]; then
      echo "  ├── 详细结果文件:"
      for JSONL in $JSONL_FILES; do
        BASENAME=$(basename "$JSONL")
        WAY_NAME=$(echo "$BASENAME" | sed 's/details_//' | sed 's/.jsonl//')
        echo "  │   - $BASENAME (${WAY_NAME:-base}方法)"
      done
    fi
    
    # 查找该数据集的图表文件
    CHARTS=$(find "$DATASET_DIR" -name "*.png" -type f | head -10)
    if [ ! -z "$CHARTS" ]; then
      echo "  └── 图表文件 (最多显示10个):"
      for CHART in $CHARTS; do
        BASENAME=$(basename "$CHART")
        WAY_NAME=$(echo "$BASENAME" | grep -oP '(base|correct|llm|together)(?=_resp_)' || echo "unknown")
        RESP_IDX=$(echo "$BASENAME" | grep -oP 'resp_\K\d+')
        echo "      - $BASENAME (${WAY_NAME:-unknown}, response ${RESP_IDX:-0})"
      done
    fi
  fi
  echo
done

# 显示多数据集对比图
echo
echo "📊 多数据集对比图:"
MULTI_CHARTS=$(find "$OUTPUT_DIR" -name "multi_proportion_bar_*.png" -type f)
if [ ! -z "$MULTI_CHARTS" ]; then
  for CHART in $MULTI_CHARTS; do
    BASENAME=$(basename "$CHART")
    # 提取way和response信息
    if [[ $BASENAME =~ multi_proportion_bar_([a-z]+)_resp_([0-9]+)\.png ]]; then
      WAY_NAME="${BASH_REMATCH[1]}"
      RESP_IDX="${BASH_REMATCH[2]}"
      echo "  - $BASENAME (${WAY_NAME}方法, response ${RESP_IDX})"
    elif [[ $BASENAME =~ multi_proportion_bar_resp_([0-9]+)\.png ]]; then
      RESP_IDX="${BASH_REMATCH[1]}"
      echo "  - $BASENAME (base方法, response ${RESP_IDX})"
    else
      echo "  - $BASENAME"
    fi
  done
else
  echo "  (暂无多数据集对比图)"
fi

echo
echo "🎉 所有分析完成！结果保存在: $OUTPUT_DIR"