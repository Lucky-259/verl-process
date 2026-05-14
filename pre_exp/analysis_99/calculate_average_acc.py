#!/usr/bin/env python3
"""
分析rollout数据，计算每个数据集的平均准确率
"""
import json
import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def analyze_rollouts(experiment_name: str, checkpoint_root: str, output_csv: str = None):
    """
    分析实验的rollout数据，计算每个数据集的平均准确率
    
    Args:
        experiment_name: 实验名称
        checkpoint_root: checkpoint根目录
        output_csv: 输出CSV文件路径
    """
    # 数据集列表
    DATASETS = ["aime", "aime25", "amc", "math", "minerva", "olympiad_bench"]
    
    # 构建模型文件夹路径
    model_dir = os.path.join(checkpoint_root, experiment_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"模型文件夹不存在: {model_dir}")
    
    print(f"开始分析实验: {experiment_name}")
    print(f"模型文件夹: {model_dir}")
    print(f"数据集: {DATASETS}")
    print("=" * 60)
    
    # 存储所有数据集的统计结果
    all_results = []
    dataset_results = {}
    
    # 处理每个数据集
    for dataset in DATASETS:
        dataset_dir = os.path.join(model_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"警告: 数据集文件夹不存在: {dataset_dir}")
            continue
        
        # 查找json文件
        json_files = list(Path(dataset_dir).glob("*.json"))
        if not json_files:
            print(f"警告: 在 {dataset_dir} 中未找到json文件")
            continue
        
        # 使用第一个找到的json文件
        json_file = json_files[0]
        print(f"处理数据集: {dataset}")
        print(f"  JSON文件: {json_file}")
        
        try:
            # 读取json文件
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"  警告: JSON文件不是数组格式: {json_file}")
                continue
            
            # 收集所有rollout的正确性标志
            all_correct_flags = []
            rollout_count = 0
            
            for item in data:
                if isinstance(item, dict) and 'correct_flags' in item:
                    correct_flags = item['correct_flags']
                    if isinstance(correct_flags, list):
                        all_correct_flags.extend(correct_flags)
                        rollout_count += 1
            
            if not all_correct_flags:
                print(f"  警告: 在 {dataset} 中未找到有效的correct_flags")
                continue
            
            # 计算统计信息
            total_rollouts = len(all_correct_flags)
            correct_count = sum(all_correct_flags)
            accuracy = correct_count / total_rollouts if total_rollouts > 0 else 0.0
            
            # 创建结果字典
            result = {
                'dataset': dataset,
                'total_rollouts': total_rollouts,
                'correct_count': correct_count,
                'accuracy': accuracy,
                'accuracy_percentage': accuracy * 100,
                'json_file': str(json_file)
            }
            
            print(f"  统计结果:")
            print(f"    - 总rollouts数: {total_rollouts}")
            print(f"    - 正确数: {correct_count}")
            print(f"    - 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # 保存数据集的结果到单独的json文件
            output_json = os.path.join(dataset_dir, f"{dataset}_rollout_stats.json")
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"  结果已保存到: {output_json}")
            
            # 存储结果
            dataset_results[dataset] = result
            all_results.append(result)
            
        except Exception as e:
            print(f"  处理 {dataset} 时出错: {e}")
            continue
    
    print("=" * 60)
    print("所有数据集处理完成")
    
    # 保存汇总的CSV文件
    if output_csv is None:
        output_csv = os.path.join(model_dir, f"{experiment_name}_rollout_summary.csv")
    
    # 创建CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['dataset', 'total_rollouts', 'correct_count', 
                     'accuracy', 'accuracy_percentage', 'json_file']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"汇总CSV文件已保存到: {output_csv}")
    
    # 显示汇总统计
    if all_results:
        print("\n汇总统计:")
        print("-" * 60)
        print(f"{'数据集':<15} {'总rollouts':>12} {'正确数':>10} {'准确率':>10} {'百分比':>10}")
        print("-" * 60)
        
        for result in all_results:
            print(f"{result['dataset']:<15} "
                  f"{result['total_rollouts']:>12} "
                  f"{result['correct_count']:>10} "
                  f"{result['accuracy']:>10.4f} "
                  f"{result['accuracy_percentage']:>9.2f}%")
        
        # 计算总体平均准确率
        total_rollouts_all = sum(r['total_rollouts'] for r in all_results)
        if total_rollouts_all > 0:
            weighted_accuracy = sum(r['correct_count'] for r in all_results) / total_rollouts_all
            print("-" * 60)
            print(f"{'总体平均':<15} "
                  f"{total_rollouts_all:>12} "
                  f"{sum(r['correct_count'] for r in all_results):>10} "
                  f"{weighted_accuracy:>10.4f} "
                  f"{weighted_accuracy*100:>9.2f}%")
    
    return dataset_results, output_csv

def main():
    parser = argparse.ArgumentParser(description='分析rollout数据，计算每个数据集的平均准确率')
    parser.add_argument('--experiment', required=True, help='实验名称')
    parser.add_argument('--checkpoint-root', default='/mnt/nvme1/luoyingfeng/lucky/verl-process/eval/redundancy',
                       help='checkpoint根目录')
    parser.add_argument('--output-csv', help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    try:
        dataset_results, csv_file = analyze_rollouts(
            args.experiment,
            args.checkpoint_root,
            args.output_csv
        )
        
        # 保存处理完成的标记文件
        model_dir = os.path.join(args.checkpoint_root, args.experiment)
        complete_file = os.path.join(model_dir, "rollout_analysis_complete.txt")
        with open(complete_file, 'w') as f:
            f.write(f"Rollout分析完成于: {os.path.basename(csv_file)}\n")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        exit(1)

if __name__ == "__main__":
    main()