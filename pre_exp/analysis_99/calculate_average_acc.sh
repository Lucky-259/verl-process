#!/bin/bash
# 分析rollout数据，计算每个数据集的平均准确率

set -e

EXPERIMENT_NAME=${1}
if [ -z "$EXPERIMENT_NAME" ]; then
    echo "用法: $0 <experiment_name>"
    echo ""
    echo "示例: $0 my_experiment"
    echo ""
    echo "描述: 分析指定实验的rollout数据，计算每个数据集的平均准确率"
    echo "      在每个数据集文件夹中生成统计JSON文件"
    echo "      在模型文件夹中生成汇总CSV文件"
    exit 1
fi

CHECKPOINT_ROOT="/mnt/nvme1/luoyingfeng/lucky/verl-process/eval/redundancy/DS1.5B_8k_redundancy_self_1_5e-4_1_correct_DAPO"

# 数据集列表
DATASETS=("aime" "aime25" "amc" "math" "minerva" "olympiad_bench")

echo "================================================================"
echo "开始分析rollout数据，计算数据集平均准确率"
echo "实验: $EXPERIMENT_NAME"
echo "数据集: ${DATASETS[*]}"
echo "================================================================"

# 检查模型文件夹是否存在
MODEL_DIR="$CHECKPOINT_ROOT/$EXPERIMENT_NAME"
if [ ! -d "$MODEL_DIR" ]; then
    echo "错误: 模型文件夹不存在: $MODEL_DIR"
    echo "可用的实验文件夹:"
    find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name "*" 2>/dev/null | grep -v "^$CHECKPOINT_ROOT$" | xargs -I {} basename {} | sort
    exit 1
fi

# 检查数据集文件夹
echo "检查数据集文件夹..."
MISSING_DATASETS=()
for dataset in "${DATASETS[@]}"; do
    DATASET_DIR="$MODEL_DIR/$dataset"
    if [ ! -d "$DATASET_DIR" ]; then
        MISSING_DATASETS+=("$dataset")
        echo "  警告: 数据集文件夹不存在: $DATASET_DIR"
    else
        # 检查数据集文件夹中是否有json文件
        JSON_FILES=$(find "$DATASET_DIR" -maxdepth 1 -name "*.json" 2>/dev/null | wc -l)
        if [ "$JSON_FILES" -eq 0 ]; then
            echo "  警告: 在 $dataset 中未找到JSON文件"
        else
            echo "  ✓ $dataset: 找到 $JSON_FILES 个JSON文件"
        fi
    fi
done

if [ ${#MISSING_DATASETS[@]} -eq ${#DATASETS[@]} ]; then
    echo "错误: 未找到任何数据集文件夹"
    exit 1
fi

# 检查Python脚本是否存在
if [ ! -f "calculate_average_acc.py" ]; then
    echo "错误: calculate_average_acc.py 不存在于当前目录"
    echo "请确保该文件与脚本在同一目录下"
    exit 1
fi

# 运行Python分析脚本
echo ""
echo "运行Python分析脚本..."
echo "------------------------------------------------"

python3 calculate_average_acc.py \
    --experiment "$EXPERIMENT_NAME" \
    --checkpoint-root "$CHECKPOINT_ROOT"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 分析完成!"
else
    echo ""
    echo "✗ 分析失败"
    exit 1
fi

# 显示结果文件
echo ""
echo "================================================================"
echo "分析结果文件:"
echo "================================================================"

# 显示每个数据集的统计文件
echo ""
echo "数据集统计文件:"
echo "----------------"

for dataset in "${DATASETS[@]}"; do
    DATASET_DIR="$MODEL_DIR/$dataset"
    if [ -d "$DATASET_DIR" ]; then
        STATS_FILE="$DATASET_DIR/${dataset}_rollout_stats.json"
        if [ -f "$STATS_FILE" ]; then
            echo "✓ $dataset: $STATS_FILE"
            # 显示文件内容
            if command -v jq &> /dev/null; then
                ACCURACY=$(jq -r '.accuracy_percentage' "$STATS_FILE")
                echo "  准确率: ${ACCURACY}%"
            fi
        else
            echo "  - $dataset: 未生成统计文件"
        fi
    fi
done

# 显示汇总CSV文件
echo ""
echo "汇总文件:"
echo "----------------"

SUMMARY_CSV="$MODEL_DIR/${EXPERIMENT_NAME}_rollout_summary.csv"
if [ -f "$SUMMARY_CSV" ]; then
    echo "✓ 汇总CSV: $SUMMARY_CSV"
    echo ""
    echo "CSV内容预览:"
    echo "------------"
    head -5 "$SUMMARY_CSV" | while IFS= read -r line; do
        echo "  $line"
    done
    
    if [ $(wc -l < "$SUMMARY_CSV") -gt 6 ]; then
        echo "  ... (更多行)"
    fi
else
    echo "  - 未找到汇总CSV文件"
fi

# 验证文件
echo ""
echo "================================================================"
echo "验证生成的文件:"
echo "================================================================"

FILES_GENERATED=0
for dataset in "${DATASETS[@]}"; do
    DATASET_DIR="$MODEL_DIR/$dataset"
    if [ -d "$DATASET_DIR" ]; then
        STATS_FILE="$DATASET_DIR/${dataset}_rollout_stats.json"
        if [ -f "$STATS_FILE" ]; then
            FILES_GENERATED=$((FILES_GENERATED + 1))
        fi
    fi
done

if [ -f "$SUMMARY_CSV" ]; then
    CSV_LINES=$(wc -l < "$SUMMARY_CSV" | tr -d ' ')
    echo "✓ 成功生成汇总CSV文件，包含 $CSV_LINES 行"
fi

echo "✓ 成功生成 $FILES_GENERATED 个数据集统计文件"
echo ""
echo "================================================================"
echo "分析完成!"
echo "================================================================"