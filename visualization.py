import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from config import COLORS

# Cấu hình phong cách biểu đồ chuẩn học thuật
plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120, 'font.size': 11})

def ensure_out_dir():
    os.makedirs('outputs', exist_ok=True)

def plot_loss_curves(all_results, title='Quỹ đạo Loss theo Epoch', save_name=None):
    ensure_out_dir()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    act_names = list(dict.fromkeys(r['activation'] for r in all_results))
    
    for act_name in act_names:
        act_res = [r for r in all_results if r['activation'] == act_name]
        color = COLORS.get(act_name, '#888888')
        
        train_arr = np.array([r['history']['train_loss'] for r in act_res])
        val_arr = np.array([r['history']['val_loss'] for r in act_res])
        epochs = np.arange(1, train_arr.shape[1] + 1)
        
        # Vẽ giá trị trung bình trên các Seed
        axes[0].plot(epochs, train_arr.mean(axis=0), label=act_name, color=color, linewidth=2)
        axes[1].plot(epochs, val_arr.mean(axis=0), label=act_name, color=color, linewidth=2)
        
    axes[0].set_title('Training Loss'); axes[0].set_xlabel('Epoch'); axes[0].grid(True); axes[0].legend()
    axes[1].set_title('Validation Loss'); axes[1].set_xlabel('Epoch'); axes[1].grid(True); axes[1].legend()
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_name: plt.savefig(f'outputs/{save_name}')
    plt.close()

def plot_accuracy_curves(all_results, title='Sự biến thiên Accuracy', save_name=None):
    ensure_out_dir()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    act_names = list(dict.fromkeys(r['activation'] for r in all_results))
    
    for act_name in act_names:
        act_res = [r for r in all_results if r['activation'] == act_name]
        color = COLORS.get(act_name, '#888888')
        
        train_arr = np.array([r['history']['train_acc'] for r in act_res])
        val_arr = np.array([r['history']['val_acc'] for r in act_res])
        epochs = np.arange(1, train_arr.shape[1] + 1)
        
        axes[0].plot(epochs, train_arr.mean(axis=0)*100, label=act_name, color=color, linewidth=2)
        axes[1].plot(epochs, val_arr.mean(axis=0)*100, label=act_name, color=color, linewidth=2)
        
    axes[0].set_title('Training Accuracy (%)'); axes[0].set_xlabel('Epoch'); axes[0].grid(True); axes[0].legend()
    axes[1].set_title('Validation Accuracy (%)'); axes[1].set_xlabel('Epoch'); axes[1].grid(True); axes[1].legend()
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_name: plt.savefig(f'outputs/{save_name}')
    plt.close()

def plot_gradient_by_layer(all_results, epoch_idx=-1, title='Phân bố Gradient ở Từng Lớp', save_name=None):
    ensure_out_dir()
    data = []
    act_names = list(dict.fromkeys(r['activation'] for r in all_results))
    for act_name in act_names:
        act_res = [r for r in all_results if r['activation'] == act_name]
        grad_list = []
        for r in act_res:
            grad_list.append(r['history']['grad_stats'][epoch_idx])
            
        n_layers = len(grad_list[0])
        for i in range(n_layers):
            mean_val = np.mean([g[i]['mean_abs'] for g in grad_list])
            data.append({'Activation': act_name, 'Layer': i, 'Gradient Mean Abs': mean_val})
            
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Layer', y='Gradient Mean Abs', hue='Activation', palette=COLORS, inner="stick")
    plt.yscale('log')
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    if save_name: plt.savefig(f'outputs/{save_name}')
    plt.close()

def plot_gradient_evolution(all_results, title='Quỹ đạo tiến hóa Gradient', save_name=None):
    ensure_out_dir()
    plt.figure(figsize=(12, 6))
    act_names = list(dict.fromkeys(r['activation'] for r in all_results))
    for act_name in act_names:
        act_res = [r for r in all_results if r['activation'] == act_name]
        epochs = len(act_res[0]['history']['grad_stats'])
        evo = []
        for e in range(epochs):
            # Lấy mean gradient của Layer 0 (Lớp nhạy cảm nhất với Vanishing)
            val = np.mean([r['history']['grad_stats'][e][0]['mean_abs'] for r in act_res])
            evo.append(val)
        plt.plot(range(1, epochs+1), evo, label=act_name, color=COLORS.get(act_name), linewidth=2)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Gradient (Layer 0)')
    plt.grid(True)
    plt.legend()
    if save_name: plt.savefig(f'outputs/{save_name}')
    plt.close()

def create_summary_table(results):
    # Tổng hợp bảng số liệu báo cáo
    act_names = list(dict.fromkeys(r['activation'] for r in results))
    summary = []
    for act in act_names:
        r_act = [r for r in results if r['activation'] == act]
        acc_mean = np.mean([r['test_acc'] for r in r_act]) * 100
        time_mean = np.mean([r['time'] for r in r_act])
        summary.append({
            'Hàm kích hoạt': act, 
            'Accuracy (%)': f"{acc_mean:.2f}", 
            'Thời gian trung bình (s)': f"{time_mean:.1f}"
        })
    return pd.DataFrame(summary)

def plot_shallow_vs_deep_accuracy(results_shallow, results_deep, title='So sánh Accuracy: Mạng Nông vs Mạng Sâu', save_name=None):
    ensure_out_dir()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    act_names = ['Sigmoid', 'Tanh', 'ReLU']
    x_pos = np.arange(len(act_names))
    width = 0.35
    
    # 1) Test Accuracy
    acc_A = [np.mean([r['test_acc'] for r in results_shallow if r['activation'] == a]) * 100 for a in act_names]
    acc_B = [np.mean([r['test_acc'] for r in results_deep if r['activation'] == a]) * 100 for a in act_names]
    std_A = [np.std([r['test_acc'] for r in results_shallow if r['activation'] == a]) * 100 for a in act_names]
    std_B = [np.std([r['test_acc'] for r in results_deep if r['activation'] == a]) * 100 for a in act_names]
    
    axes[0].bar(x_pos - width/2, acc_A, width, yerr=std_A, label='Mạng nông (3 layers)', capsize=5, color='#4c72b0')
    axes[0].bar(x_pos + width/2, acc_B, width, yerr=std_B, label='Mạng sâu (8 layers)', capsize=5, color='#c44e52')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy theo Độ sâu mạng')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(act_names)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2) Training Time
    time_A = [np.mean([r['time'] for r in results_shallow if r['activation'] == a]) for a in act_names]
    time_B = [np.mean([r['time'] for r in results_deep if r['activation'] == a]) for a in act_names]
    std_tA = [np.std([r['time'] for r in results_shallow if r['activation'] == a]) for a in act_names]
    std_tB = [np.std([r['time'] for r in results_deep if r['activation'] == a]) for a in act_names]
    
    axes[1].bar(x_pos - width/2, time_A, width, yerr=std_tA, label='Mạng nông (3 layers)', capsize=5, color='#55a868')
    axes[1].bar(x_pos + width/2, time_B, width, yerr=std_tB, label='Mạng sâu (8 layers)', capsize=5, color='#dd8452')
    axes[1].set_ylabel('Thời gian huấn luyện (giây)')
    axes[1].set_title('Thời gian huấn luyện theo Độ sâu mạng')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(act_names)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3) Gradient (Layer cuối)
    grad_A = [np.mean([r['history']['grad_stats'][-1][-1]['mean_abs'] for r in results_shallow if r['activation'] == a]) for a in act_names]
    grad_B = [np.mean([r['history']['grad_stats'][-1][-1]['mean_abs'] for r in results_deep if r['activation'] == a]) for a in act_names]
    
    axes[2].bar(x_pos - width/2, grad_A, width, label='Mạng nông (3 layers)', color='#8172b2')
    axes[2].bar(x_pos + width/2, grad_B, width, label='Mạng sâu (8 layers)', color='#937860')
    axes[2].set_ylabel('Mean Absolute Gradient (Layer Output)')
    axes[2].set_title('Gradient tại Output Layer')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(act_names)
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_name: plt.savefig(f'outputs/{save_name}')
    plt.close()
