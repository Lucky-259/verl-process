import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def create_pareto_for_dataset(dataset_data, dataset_name, model_type):
    """为特定数据集和模型类型创建帕累托前沿图"""
    save_folder = f"/mnt/nvme1/luoyingfeng/lucky/verl-process/pre_exp/eval_results/pareto_analysis_2"
    os.makedirs(save_folder, exist_ok=True)
    
    # 创建图表
    plt.figure(figsize=(14, 10))
    
    # 提取数据
    accuracy_data = []
    length_data = []
    method_names = []
    
    for method, accuracy in dataset_data['accuracy'].items():
        length = dataset_data['avg_tokens'][method]
        accuracy_data.append(accuracy)
        length_data.append(length)
        method_names.append(method)
    
    if not accuracy_data:       
        print(f"没有找到{dataset_name}的{model_type}数据")
        return
    
    # 绘制所有数据点
    colors = plt.cm.viridis(np.linspace(0, 1, len(accuracy_data)))
    scatter = plt.scatter(length_data, accuracy_data, 
                         alpha=0.8, s=120, c=colors, edgecolors='black', linewidth=0.5)
    
    # 添加数据点标签
    for i, (x, y, name) in enumerate(zip(length_data, accuracy_data, method_names)):
        plt.annotate(name, (x, y), xytext=(8, 8), textcoords='offset points',
                    fontsize=9, alpha=0.9, bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightblue', alpha=0.3))
    
    # 计算帕累托前沿
    points = list(zip(length_data, accuracy_data))
    pareto_points = []
    
    for point in points:
        is_pareto = True
        for other in points:
            # 帕累托最优：没有其他点同时具有更短的token和更高或相等的准确率
            if (other[0] < point[0] and other[1] >= point[1]) or \
               (other[0] <= point[0] and other[1] > point[1]):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(point)
    
    # 绘制帕累托前沿
    if pareto_points:
        # 按长度排序
        pareto_points.sort(key=lambda x: x[0])
        pareto_x, pareto_y = zip(*pareto_points)
        
        # 绘制帕累托前沿线
        plt.plot(pareto_x, pareto_y, 'r--', linewidth=3, label='帕累托前沿', alpha=0.8)
        
        # 突出显示帕累托最优解
        plt.scatter(pareto_x, pareto_y, c='red', s=200, marker='*', 
                   edgecolors='darkred', linewidth=2, label='帕累托最优解', zorder=5)
    
    # 图表美化
    plt.xlabel('平均推理长度 (Tokens)', fontsize=14, fontweight='bold')
    plt.ylabel('准确率 (%)', fontsize=14, fontweight='bold')
    plt.title(f'{dataset_name} - {model_type}模型\n性能 vs 效率帕累托分析', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12)
    
    # 设置坐标轴范围
    plt.xlim(min(length_data) * 0.8, max(length_data) * 1.1)
    plt.ylim(min(accuracy_data) * 0.9, max(accuracy_data) * 1.05)
    
    # 添加目标方向指示
    plt.annotate('↑ 更高准确率', xy=(0.02, 0.98), xycoords='axes fraction', 
                fontsize=12, color='green', fontweight='bold')
    plt.annotate('← 更少tokens', xy=(0.98, 0.02), xycoords='axes fraction', 
                fontsize=12, color='blue', fontweight='bold', ha='right')
    
    plt.tight_layout()
    
    # 保存图片
    filename = f"{dataset_name}_{model_type.replace(' ', '_')}_pareto.png"
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"图表已保存: {save_path}")
    
    plt.show()
    
    # 打印帕累托最优解信息
    if pareto_points:
        print(f"\n{dataset_name} - {model_type}帕累托最优解:")
        for i, (length, accuracy) in enumerate(pareto_points):
            method_idx = points.index((length, accuracy))
            method_name = method_names[method_idx]
            print(f"  {i+1}. {method_name}: 准确率={accuracy:.2f}%, 长度={length:.0f}tokens")

def alternative_approach():
    """
    替代方法：直接使用硬编码的Excel数据
    每个数据集分别绘制1.5B和7B模型的帕累托图
    """
    # 完整的数据结构 - 添加了APR-1.5B和APR-7B数据
    all_datasets_data = {
        'AIME24': {
            '1.5B': {
                'accuracy': {
                    'Original Model': 19.2, 'AdaptThink-1.5B-delta0.05': 26.2,
                    'L1-Qwen-1.5B-Max': 28.8, 'TrainingEfficient_alpha_0.1_DS-1.5B': 29.2,
                    'DS-1.5B-thinkprune-iter2k': 30.0, 'Laser-L8192-1.5B': 27.3,
                    'Laser-DE-L4096-1.5B': 25.6, 'DLER-R1-1.5B-Research': 32.1,
                    'APR-1.5B (Ours)': 29.0
                },
                'avg_tokens': {
                    'Original Model': 7278, 'AdaptThink-1.5B-delta0.05': 6172,
                    'L1-Qwen-1.5B-Max': 2893, 'TrainingEfficient_alpha_0.1_DS-1.5B': 6245,
                    'DS-1.5B-thinkprune-iter2k': 4670, 'Laser-L8192-1.5B': 6139,
                    'Laser-DE-L4096-1.5B': 5173, 'DLER-R1-1.5B-Research': 3376,
                    'APR-1.5B (Ours)': 3767.2
                }
            },
            '7B': {
                'accuracy': {
                    'Original Model': 37.7, 'AdaptThink-7B-delta0.05': 45.6,
                    'L1-Qwen-7B-Max': 44.8, 'TrainingEfficient_alpha_0.1_DS-7B': 42.1,
                    'SB_DS7B_alpha_2': 45.4, 'Laser-DE-L4096-7B': 47.7,
                    'DLER-R1-7B-Research': 51.3, 'APR-7B (Ours)': 43.96
                },
                'avg_tokens': {
                    'Original Model': 6707, 'AdaptThink-7B-delta0.05': 6255,
                    'L1-Qwen-7B-Max': 4041, 'TrainingEfficient_alpha_0.1_DS-7B': 6230,
                    'SB_DS7B_alpha_2': 3955, 'Laser-DE-L4096-7B': 4642,
                    'DLER-R1-7B-Research': 3209, 'APR-7B (Ours)': 3050.6
                }
            }
        },
        
        'AIME25': {
            '1.5B': {
                'accuracy': {
                    'Original Model': 20.4, 'AdaptThink-1.5B-delta0.05': 22.1,
                    'L1-Qwen-1.5B-Max': 24.6, 'TrainingEfficient_alpha_0.1_DS-1.5B': 22.9,
                    'DS-1.5B-thinkprune-iter2k': 19.6, 'Laser-L8192-1.5B': 24.4,
                    'Laser-DE-L4096-1.5B': 22.5, 'DLER-R1-1.5B-Research': 27.3,
                    'APR-1.5B (Ours)': 22.7
                },
                'avg_tokens': {
                    'Original Model': 7045, 'AdaptThink-1.5B-delta0.05': 6002,
                    'L1-Qwen-1.5B-Max': 2734, 'TrainingEfficient_alpha_0.1_DS-1.5B': 6186,
                    'DS-1.5B-thinkprune-iter2k': 4320, 'Laser-L8192-1.5B': 5797,
                    'Laser-DE-L4096-1.5B': 4674, 'DLER-R1-1.5B-Research': 3108,
                    'APR-1.5B (Ours)': 3284
                }
            },
            '7B': {
                'accuracy': {
                    'Original Model': 20.4, 'AdaptThink-7B-delta0.05': 32.1,
                    'L1-Qwen-7B-Max': 31.5, 'TrainingEfficient_alpha_0.1_DS-7B': 31.3,
                    'SB_DS7B_alpha_2': 29.6, 'Laser-DE-L4096-7B': 37.5,
                    'DLER-R1-7B-Research': 35.6, 'APR-7B (Ours)': 32.1
                },
                'avg_tokens': {
                    'Original Model': 7045, 'AdaptThink-7B-delta0.05': 6239,
                    'L1-Qwen-7B-Max': 3891, 'TrainingEfficient_alpha_0.1_DS-7B': 6193,
                    'SB_DS7B_alpha_2': 3897, 'Laser-DE-L4096-7B': 4636,
                    'DLER-R1-7B-Research': 3176, 'APR-7B (Ours)': 3025
                }
            }
        },
        
        'AMC': {
            '1.5B': {
                'accuracy': {
                    'Original Model': 51.5, 'AdaptThink-1.5B-delta0.05': 61.7,
                    'L1-Qwen-1.5B-Max': 67.8, 'TrainingEfficient_alpha_0.1_DS-1.5B': 58.9,
                    'DS-1.5B-thinkprune-iter2k': 65.2, 'Laser-L8192-1.5B': 68.3,
                    'Laser-DE-L4096-1.5B': 65.9, 'DLER-R1-1.5B-Research': 74.2,
                    'SB_DS1.5B_alpha_2': 35.4, 'APR-1.5B (Ours)': 69.6
                },
                'avg_tokens': {
                    'Original Model': 5727, 'AdaptThink-1.5B-delta0.05': 3428,
                    'L1-Qwen-1.5B-Max': 2300, 'TrainingEfficient_alpha_0.1_DS-1.5B': 4231,
                    'DS-1.5B-thinkprune-iter2k': 2964, 'Laser-L8192-1.5B': 4212,
                    'Laser-DE-L4096-1.5B': 3333, 'DLER-R1-1.5B-Research': 2559,
                    'SB_DS1.5B_alpha_2': 826, 'APR-1.5B (Ours)': 2352
                }
            },
            '7B': {
                'accuracy': {
                    'Original Model': 69.5, 'AdaptThink-7B-delta0.05': 75.2,
                    'L1-Qwen-7B-Max': 77.4, 'TrainingEfficient_alpha_0.1_DS-7B': 75.3,
                    'SB_DS7B_alpha_2': 74.4, 'Laser-DE-L4096-7B': 82.1,
                    'DLER-R1-7B-Research': 83.3, 'APR-7B (Ours)': 81.4
                },
                'avg_tokens': {
                    'Original Model': 5014, 'AdaptThink-7B-delta0.05': 4103,
                    'L1-Qwen-7B-Max': 2742, 'TrainingEfficient_alpha_0.1_DS-7B': 4169,
                    'SB_DS7B_alpha_2': 2272, 'Laser-DE-L4096-7B': 2793,
                    'DLER-R1-7B-Research': 2230, 'APR-7B (Ours)': 2255.7
                }
            }
        },
        
        'MATH500': {
            '1.5B': {
                'accuracy': {
                    'Original Model': 85.1, 'AdaptThink-1.5B-delta0.05': 80.8,
                    'L1-Qwen-1.5B-Max': 84.8, 'TrainingEfficient_alpha_0.1_DS-1.5B': 81.8,
                    'DS-1.5B-thinkprune-iter2k': 83.1, 'Laser-L8192-1.5B': 84.9,
                    'Laser-DE-L4096-1.5B': 82.8, 'DLER-R1-1.5B-Research': 86.9,
                    'APR-1.5B (Ours)': 84.7
                },
                'avg_tokens': {
                    'Original Model': 3112, 'AdaptThink-1.5B-delta0.05': 1532,
                    'L1-Qwen-1.5B-Max': 1899, 'TrainingEfficient_alpha_0.1_DS-1.5B': 2285,
                    'DS-1.5B-thinkprune-iter2k': 1822, 'Laser-L8192-1.5B': 2608,
                    'Laser-DE-L4096-1.5B': 1909, 'DLER-R1-1.5B-Research': 1787,
                    'APR-1.5B (Ours)': 1513
                }
            },
            '7B': {
                'accuracy': {
                    'Original Model': 86.7, 'AdaptThink-7B-delta0.05': 88.2,
                    'L1-Qwen-7B-Max': 90.2, 'TrainingEfficient_alpha_0.1_DS-7B': 89.1,
                    'SB_DS7B_alpha_2': 82.6, 'Laser-DE-L4096-7B': 91.5,
                    'DLER-R1-7B-Research': 91.8, 'APR-7B (Ours)': 90.3
                },
                'avg_tokens': {
                    'Original Model': 3274, 'AdaptThink-7B-delta0.05': 1946,
                    'L1-Qwen-7B-Max': 2125, 'TrainingEfficient_alpha_0.1_DS-7B': 2427,
                    'SB_DS7B_alpha_2': 1037, 'Laser-DE-L4096-7B': 1634,
                    'DLER-R1-7B-Research': 1429, 'APR-7B (Ours)': 1493.5
                }
            }
        },
        
        'Minerva': {
            '1.5B': {
                'accuracy': {
                    'Original Model': 30.3, 'AdaptThink-1.5B-delta0.05': 24.7,
                    'L1-Qwen-1.5B-Max': 29.4, 'TrainingEfficient_alpha_0.1_DS-1.5B': 27.0,
                    'DS-1.5B-thinkprune-iter2k': 27.6, 'Laser-L8192-1.5B': 31.1,
                    'Laser-DE-L4096-1.5B': 29.0, 'DLER-R1-1.5B-Research': 31.5,
                    'APR-1.5B (Ours)': 30.2
                },
                'avg_tokens': {
                    'Original Model': 4082, 'AdaptThink-1.5B-delta0.05': 1637,
                    'L1-Qwen-1.5B-Max': 2779, 'TrainingEfficient_alpha_0.1_DS-1.5B': 3018,
                    'DS-1.5B-thinkprune-iter2k': 2118, 'Laser-L8192-1.5B': 3638,
                    'Laser-DE-L4096-1.5B': 2289, 'DLER-R1-1.5B-Research': 2225,
                    'APR-1.5B (Ours)': 1985
                }
            },
            '7B': {
                'accuracy': {
                    'Original Model': 36.2, 'AdaptThink-7B-delta0.05': 35.1,
                    'L1-Qwen-7B-Max': 38.9, 'TrainingEfficient_alpha_0.1_DS-7B': 37.7,
                    'SB_DS7B_alpha_2': 31.6, 'Laser-DE-L4096-7B': 38.9,
                    'DLER-R1-7B-Research': 39.5, 'APR-7B (Ours)': 38.4
                },
                'avg_tokens': {
                    'Original Model': 4266, 'AdaptThink-7B-delta0.05': 2570,
                    'L1-Qwen-7B-Max': 2120, 'TrainingEfficient_alpha_0.1_DS-7B': 2940,
                    'SB_DS7B_alpha_2': 901, 'Laser-DE-L4096-7B': 1850,
                    'DLER-R1-7B-Research': 1798, 'APR-7B (Ours)': 1647
                }
            }
        },
        
        'Olympiad_bench': {
            '1.5B': {
                'accuracy': {
                    'Original Model': 37.5, 'AdaptThink-1.5B-delta0.05': 40.1,
                    'L1-Qwen-1.5B-Max': 46.2, 'TrainingEfficient_alpha_0.1_DS-1.5B': 40.9,
                    'DS-1.5B-thinkprune-iter2k': 42.9, 'Laser-L8192-1.5B': 47.1,
                    'Laser-DE-L4096-1.5B': 44.2, 'DLER-R1-1.5B-Research': 49.7,
                    'APR-1.5B (Ours)': 46.4
                },
                'avg_tokens': {
                    'Original Model': 5775, 'AdaptThink-1.5B-delta0.05': 3612,
                    'L1-Qwen-1.5B-Max': 2311, 'TrainingEfficient_alpha_0.1_DS-1.5B': 4523,
                    'DS-1.5B-thinkprune-iter2k': 3162, 'Laser-L8192-1.5B': 4076,
                    'Laser-DE-L4096-1.5B': 3335, 'DLER-R1-1.5B-Research': 2595,
                    'APR-1.5B (Ours)': 2450
                }
            },
            '7B': {
                'accuracy': {
                    'Original Model': 46.1, 'AdaptThink-7B-delta0.05': 50.8,
                    'L1-Qwen-7B-Max': 52.5, 'TrainingEfficient_alpha_0.1_DS-7B': 51.8,
                    'SB_DS7B_alpha_2': 49.6, 'Laser-DE-L4096-7B': 56.0,
                    'DLER-R1-7B-Research': 57.2, 'APR-7B (Ours)': 54.0
                },
                'avg_tokens': {
                    'Original Model': 5395, 'AdaptThink-7B-delta0.05': 4402,
                    'L1-Qwen-7B-Max': 2835, 'TrainingEfficient_alpha_0.1_DS-7B': 4437,
                    'SB_DS7B_alpha_2': 2361, 'Laser-DE-L4096-7B': 2998,
                    'DLER-R1-7B-Research': 2316, 'APR-7B (Ours)': 2235
                }
            }
        },

        'Overall': {
            '1.5B': {
                'accuracy': {
                    'Original Model': 44.7, 'AdaptThink-1.5B-delta0.05': 46.7,
                    'L1-Qwen-1.5B-Max': 51.4, 'TrainingEfficient_alpha_0.1_DS-1.5B': 47.6,
                    'DS-1.5B-thinkprune-iter2k': 49.8, 'Laser-L8192-1.5B': 51.7,
                    'Laser-DE-L4096-1.5B': 49.5, 'DLER-R1-1.5B-Research': 54.9,
                    'APR-1.5B (Ours)': 52.0
                },
                'avg_tokens': {
                    'Original Model': 5195, 'AdaptThink-1.5B-delta0.05': 3276,
                    'L1-Qwen-1.5B-Max': 2436, 'TrainingEfficient_alpha_0.1_DS-1.5B': 4060,
                    'DS-1.5B-thinkprune-iter2k': 2947, 'Laser-L8192-1.5B': 4135,
                    'Laser-DE-L4096-1.5B': 3208, 'DLER-R1-1.5B-Research': 2508,
                    'APR-1.5B (Ours)': 2449
                }
            },
            '7B': {
                'accuracy': {
                    'Original Model': 55.2, 'AdaptThink-7B-delta0.05': 59.0,
                    'L1-Qwen-7B-Max': 60.8, 'TrainingEfficient_alpha_0.1_DS-7B': 59.2,
                    'SB_DS7B_alpha_2': 56.7, 'Laser-DE-L4096-7B': 63.2,
                    'DLER-R1-7B-Research': 64.6, 'APR-7B (Ours)': 61.6
                },
                'avg_tokens': {
                    'Original Model': 4931, 'AdaptThink-7B-delta0.05': 3855,
                    'L1-Qwen-7B-Max': 2773, 'TrainingEfficient_alpha_0.1_DS-7B': 4041,
                    'SB_DS7B_alpha_2': 2105, 'Laser-DE-L4096-7B': 2783,
                    'DLER-R1-7B-Research': 2197, 'APR-7B (Ours)': 2136
                }
            }
        },
    }

    print("开始生成帕累托前沿图...")
    
    # 为每个数据集分别生成1.5B和7B的帕累托图
    datasets = ['AIME24', 'AIME25', 'AMC', 'MATH500', 'Minerva', 'Olympiad_bench', 'Overall']
    
    for dataset in datasets:
        print("\n" + "="*60)
        print(f"正在处理 {dataset} 数据集")
        print("="*60)
        
        if dataset in all_datasets_data:
            dataset_data = all_datasets_data[dataset]
            
            # 生成1.5B模型的帕累托图
            print(f"\n生成 {dataset} - 1.5B模型帕累托图...")
            create_pareto_for_dataset(dataset_data['1.5B'], dataset, "1.5B模型")
            
            # 生成7B模型的帕累托图
            print(f"\n生成 {dataset} - 7B模型帕累托图...")
            create_pareto_for_dataset(dataset_data['7B'], dataset, "7B模型")
    
    print("\n" + "="*60)
    print("所有数据集的帕累托前沿图生成完成！")
    print("每个数据集包含：")
    print("1. 1.5B模型所有方法的帕累托图")
    print("2. 7B模型所有方法的帕累托图")
    print("="*60)

def create_comparison_plots():
    """创建模型对比图（可选功能）"""
    # 可以添加1.5B和7B在同一图中的对比功能
    pass

if __name__ == "__main__":
    alternative_approach()