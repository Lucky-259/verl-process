#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from typing import List, Dict, Any, Tuple, Optional

THINK_END_TAG = "</think>"

# 定义关键词类别
KEYWORD_CATEGORIES = {
    'Reasoning_Stages_Problem_Setup': [
        'let me', "let's", 'we have', 'given that', 'the problem states',
        'we need to find', 'we are asked'
    ],
    'Reasoning_Stages_Exploration': [
        'suppose', 'consider', 'try', 'perhaps', 'alternatively',
        'first', 'then', 'next', 'finally', 'step by step',
    ],
    'Reasoning_Stages_Verification': [
        "check", "verify", "confirm", "make sure", "double-check",
        "wait", "actually", "hmm", "on second thought", "hold on",
    ],
    'Reasoning_Stages_Conclusion': [
        'thus', 'in conclusion', "answer", "therefore", "final", "conclude", "result",
        "equals", "solution", "hence",
    ],
}

# 类别显示名称（用于图表）
CATEGORY_DISPLAY_NAMES = {
    'Reasoning_Stages_Problem_Setup': 'Problem Setup',
    'Reasoning_Stages_Exploration': 'Exploration & Planning',
    'Reasoning_Stages_Verification': 'Verification & Correction',
    'Reasoning_Stages_Conclusion': 'Conclusion',
    #'Backtracking_Correction': 'Correction',
    #'Backtracking_Reconsideration': 'Reconsideration',
    #'Backtracking_Restart': 'Restart',
    #'Planning': 'Planning'
}

# ---------- Think/Answer split ----------
def split_think_answer(text: str) -> Tuple[Optional[str], str]:
    """Return (think_part or None if missing, answer_part). Split at the LAST </think>."""
    if THINK_END_TAG not in text:
        return None, text
    idx = text.rfind(THINK_END_TAG)
    return text[:idx], text[idx + len(THINK_END_TAG):]

# ---------- Sentence splitting with spans ----------
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\.])\s+|\n+")

def split_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
    """
    Split text into sentences but keep (start,end) char spans in the original text.
    Returns: [{"start": int, "end": int, "text": str}, ...]
    """
    spans: List[Dict[str, Any]] = []
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        end = m.start()
        raw = text[start:end]
        chunk = raw.strip()
        if chunk:
            lstrip_len = len(raw) - len(raw.lstrip())
            rstrip_len = len(raw) - len(raw.rstrip())
            real_start = start + lstrip_len
            real_end = end - rstrip_len
            spans.append(text[real_start:real_end])
        start = m.end()

    tail_raw = text[start:]
    tail = tail_raw.strip()
    if tail:
        lstrip_len = len(tail_raw) - len(tail_raw.lstrip())
        real_start = start + lstrip_len
        real_end = len(text)
        spans.append(text[real_start:real_end])

    return spans

def normalize_text(text: str) -> str:
    """将文本转换为小写并清理"""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def contains_keyword(sentence: str, keywords: List[str]) -> bool:
    """检查句子中是否包含任一关键词"""
    norm_sentence = normalize_text(sentence)
    for keyword in keywords:
        # 使用正则表达式确保匹配完整单词
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, norm_sentence):
            return True
    return False

def analyze_sentence(sentence: str) -> Dict[str, bool]:
    """分析单个句子，返回每个类别是否出现"""
    results = {}
    for category, keywords in KEYWORD_CATEGORIES.items():
        results[category] = contains_keyword(sentence, keywords)
    return results

def analyze_responses(json_file: str) -> Tuple[Dict[str, int], int]:
    """分析JSON文件中的所有responses"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取JSON文件 {json_file}: {e}")
        return None, 0
    
    # 初始化计数器
    category_counts = defaultdict(int)
    total_sentences = 0
    
    # 遍历JSON中的每个元素
    for item in data:
        if 'responses' in item and isinstance(item['responses'], list) and len(item['responses']) > 0:
            response_text = item['responses'][0]
            
            if not isinstance(response_text, str):
                continue
            
            think, answer = split_think_answer(response_text)
            think = think if think else response_text
            
            # 划分句子
            try:
                sentences = split_sentences_with_spans(think)
            except:
                # 如果nltk失败，使用简单的句子分割
                sentences = [s.strip() for s in re.split(r'[.!?]+', think) if s.strip()]
            
            # 分析每个句子
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                total_sentences += 1
                sentence_results = analyze_sentence(sentence)
                
                # 更新计数器
                for category, found in sentence_results.items():
                    if found:
                        category_counts[category] += 1
    
    return dict(category_counts), total_sentences

def calculate_frequencies(category_counts: Dict[str, int], total_sentences: int) -> Dict[str, float]:
    """计算频率（出现次数/总句子数）"""
    frequencies = {}
    for category, count in category_counts.items():
        if total_sentences > 0:
            frequencies[category] = count / total_sentences
        else:
            frequencies[category] = 0.0
    return frequencies

def create_bar_chart(frequencies: Dict[str, float], dataset_name: str, output_dir: str):
    """创建柱状图"""
    # 定义固定的类别顺序（按照KEYWORD_CATEGORIES的顺序）
    fixed_categories_order = list(KEYWORD_CATEGORIES.keys())
    
    # 按照固定顺序获取数据
    categories = []
    values = []
    for category in fixed_categories_order:
        if category in frequencies:
            categories.append(CATEGORY_DISPLAY_NAMES.get(category, category))
            values.append(frequencies[category])
    
    # 如果没有数据，使用传入的顺序
    if not categories:
        categories = [CATEGORY_DISPLAY_NAMES.get(c, c) for c in frequencies.keys()]
        values = list(frequencies.values())
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建柱状图
    bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 设置图表属性
    plt.title(f'Keyword Category Frequencies - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Frequency (occurrences per sentence)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(0, max(values) * 1.2 if values else 0.1)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加说明文本
    # 这里修改一下，因为frequencies.values()现在是频率，不是计数
    # 我们需要从别处获取总句子数
    plt.figtext(0.5, 0.01, 
                f'Note: Frequencies = occurrences / total sentences',
                ha='center', fontsize=10, style='italic')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    output_file = os.path.join(output_dir, f'{dataset_name}_frequencies.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图表已保存: {output_file}")
    return output_file

def save_results_to_csv(frequencies: Dict[str, float], counts: Dict[str, int], 
                       total_sentences: int, dataset_name: str, output_dir: str):
    """保存结果到CSV文件"""
    # 准备数据
    data = []
    for category in frequencies.keys():
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        data.append({
            'Dataset': dataset_name,
            'Category': display_name,
            'Count': counts.get(category, 0),
            'Frequency': frequencies[category],
            'Total_Sentences': total_sentences
        })
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存数据集特定文件
    dataset_csv = os.path.join(output_dir, f'{dataset_name}_results.csv')
    df.to_csv(dataset_csv, index=False)
    
    # 保存到汇总文件
    summary_csv = os.path.join(output_dir, 'summary.csv')
    if os.path.exists(summary_csv):
        summary_df = pd.read_csv(summary_csv)
        summary_df = pd.concat([summary_df, df], ignore_index=True)
    else:
        summary_df = df
    
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"结果已保存: {dataset_csv}")
    return dataset_csv, summary_csv

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("错误: 请提供JSON文件路径")
        print("使用方法: python analyze_responses.py <json_file> [output_dir] [dataset_name]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis_results"
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.dirname(json_file))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在分析: {dataset_name}")
    print(f"JSON文件: {json_file}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 分析数据
    category_counts, total_sentences = analyze_responses(json_file)
    
    if category_counts is None:
        print("分析失败")
        return
    
    # 计算频率
    frequencies = calculate_frequencies(category_counts, total_sentences)
    
    # 打印结果
    print(f"总句子数: {total_sentences}")
    print("\n类别统计:")
    print("-" * 60)
    print(f"{'Category':<25} {'Count':<10} {'Frequency':<10}")
    print("-" * 60)
    
    for category, count in sorted(category_counts.items()):
        display_name = CATEGORY_DISPLAY_NAMES.get(category, category)
        freq = frequencies[category]
        print(f"{display_name:<25} {count:<10} {freq:<10.4f}")
    
    print("-" * 50)
    
    # 创建图表
    chart_file = create_bar_chart(frequencies, dataset_name, output_dir)
    
    # 保存结果到CSV
    csv_file, summary_file = save_results_to_csv(frequencies, category_counts, 
                                                total_sentences, dataset_name, output_dir)
    
    print(f"\n分析完成!")
    print(f"图表: {chart_file}")
    print(f"数据: {csv_file}")
    print(f"汇总: {summary_file}")

if __name__ == "__main__":
    main()