import re
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from math_verify import parse, ExprExtractionConfig, LatexExtractionConfig, verify
from tqdm import tqdm
import sys
sys.path.append('/mnt/nvme1/luoyingfeng/lucky/verl-process')
from deepscaler.rewards.math_utils.utils import grade_answer_verl
plt.rcParams['font.sans-serif'] = ['SimHei']

def read_jsonl(file_path):
    """
    读取 JSONL 文件并将每一行解析为字典，返回列表。
    """
    data_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                # 跳过空行
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    data_list.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_number} 行解析失败: {e}")
                    # 根据需求，这里可以选择 continue 跳过或者 raise 报错
                    
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
        
    return data_list


def plot_bar_from_dict(data_dict, save_path="bar_chart.png", title="Redundancy Verification Proportion", xlabel="difficulty", ylabel="proportion", 
                      color='orange', figsize=(10, 6), dpi=300):
    """
    根据字典数据绘制柱状图
    
    参数:
    data_dict: dict - 键值对字典，键为字符串，值为浮点数
    title: str - 图表标题
    xlabel: str - x轴标签
    ylabel: str - y轴标签
    color: str - 柱状图颜色
    figsize: tuple - 图表大小
    """
    if not data_dict:
        print("字典为空，无法绘制图表")
        return
    
    # 提取键和值
    keys = list(data_dict.keys())
    values = list(data_dict.values())
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 绘制柱状图
    bars = plt.bar(keys, values, color=color, edgecolor='black')
    
    # 设置标题和标签
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 在每个柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 自动调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"图表已保存为: {save_path}")

# 基础版本
def simple_process_jsonl(file_path, save_figure_path):
    data_list = []
    
    diff_dict = defaultdict(lambda: [0, 0])

    data = read_jsonl(file_path)
    for item in tqdm(data, total=len(data), desc="Processing JSONL data"):
        prompt = item.get('problem')
        thinking = item.get('thinking')
        boxed = item.get('boxed')
        target = item.get('target')
        difficulty = item.get('difficulty')
        diff_dict[difficulty][1] += 1
        gold = parse("\\boxed{" + boxed + "}", extraction_config=[LatexExtractionConfig()])
        pred = parse(target, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        if boxed in target or verify(gold, pred, timeout_seconds=20):
            diff_dict[difficulty][0] += 1
        else:
            data_list.append(item)
    
    print(diff_dict)
    veri_dict = {k: v[0] / v[1] if v[1] > 0 else 0 for k, v in diff_dict.items()}
    plot_bar_from_dict(veri_dict, save_path=save_figure_path)
    
    return data_list

if __name__ == "__main__":
    input_file_path = "/mnt/nvme1/luoyingfeng/lucky/verl-process/pre_exp/eval_results/veri_analysis/QwQ-32B/math/veri_details.jsonl"
    not_same_samples_path = "/mnt/nvme1/luoyingfeng/lucky/verl-process/pre_exp/eval_results/veri_analysis/QwQ-32B/math/notsame_samples.json"
    save_figure_path = "/mnt/nvme1/luoyingfeng/lucky/verl-process/pre_exp/eval_results/veri_analysis/QwQ-32B/math/veri_bar.png"
    not_same_samples = simple_process_jsonl(input_file_path, save_figure_path)
    with open(not_same_samples_path, "w", encoding="utf-8") as f:
        json.dump(not_same_samples, f, ensure_ascii=False, indent=4)