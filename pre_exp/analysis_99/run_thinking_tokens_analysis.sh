#!/bin/bash

# run_thinking_tokens_analysis.sh
# 使用说明: run_thinking_tokens_analysis.sh <模型文件夹路径>

# 检查输入参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供模型文件夹路径"
    echo "使用方法: run_thinking_tokens_analysis.sh <模型文件夹路径>"
    exit 1
fi

MODEL_DIR="$1"

# 检查文件夹是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo "错误: 文件夹 '$MODEL_DIR' 不存在"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="../eval_results/thinking_tokens_analysis_3"
mkdir -p "$OUTPUT_DIR"

echo "开始分析模型文件夹: $MODEL_DIR"
echo "========================================="

# 查找所有数据集文件夹
datasets=()
while IFS= read -r -d '' dir; do
    if [ -f "$dir/results_details.json" ]; then
        datasets+=("$dir")
        echo "找到数据集: $(basename "$dir")"
    fi
done < <(find "$MODEL_DIR" -maxdepth 1 -type d -print0 | sort -z)

if [ ${#datasets[@]} -eq 0 ]; then
    echo "错误: 没有找到包含 results_details.json 的数据集文件夹"
    exit 1
fi

echo "========================================="
echo "找到 ${#datasets[@]} 个数据集"
echo "开始分析..."

# 为每个数据集运行Python分析脚本
for dataset_path in "${datasets[@]}"; do
    dataset_name=$(basename "$dataset_path")
    json_file="$dataset_path/results_details.json"
    
    echo "处理数据集: $dataset_name"
    
    # 运行Python脚本进行分析
    python3 thinking_tokens_analysis.py "$json_file" "$OUTPUT_DIR" "$dataset_name"
    
    if [ $? -eq 0 ]; then
        echo "✓ 完成分析: $dataset_name"
    else
        echo "✗ 分析失败: $dataset_name"
    fi
    echo "---"
done

echo "========================================="
echo "所有分析完成！结果保存在: $OUTPUT_DIR/"
echo "生成的图表:"
ls -la "$OUTPUT_DIR/"*.png 2>/dev/null || echo "暂无图表文件"
echo "生成的数据文件:"
ls -la "$OUTPUT_DIR/"*.csv 2>/dev/null || echo "暂无数据文件"

# 生成汇总报告
if [ -f "$OUTPUT_DIR/summary.csv" ]; then
    echo "========================================="
    echo "汇总报告:"
    column -t -s, "$OUTPUT_DIR/summary.csv" 2>/dev/null || cat "$OUTPUT_DIR/summary.csv"
fi