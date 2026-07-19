from config import *
from data_loader import get_dataloaders
from engine import run_training_experiment, count_dead_neurons
from visualization import *

def main():
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    
    print("--- 1. Mạng nông (3 lớp) ---")
    results_shallow = []
    for act in ['Sigmoid', 'Tanh', 'ReLU']:
        for seed in SEEDS:
            res = run_training_experiment(act, SHALLOW_LAYERS, SHALLOW_HIDDEN, seed, train_loader, val_loader, test_loader)
            results_shallow.append(res)
        
    plot_loss_curves(results_shallow, 'Loss (Mạng Nông)', 'fig_01_shallow_loss.png')
    plot_accuracy_curves(results_shallow, 'Accuracy (Mạng Nông)', 'fig_02_shallow_acc.png')
    print(create_summary_table(results_shallow).to_string(index=False))

    print("\n--- 2. Dying ReLU (LR = 0.1) ---")
    results_dying = []
    for act in ['ReLU', 'Leaky ReLU']:
        for seed in SEEDS:
            res = run_training_experiment(act, DEEP_LAYERS, DEEP_HIDDEN, seed, train_loader, val_loader, test_loader, epochs=10, lr=0.1)
            results_dying.append(res)
            # Tính lượng neuron chết tại vòng cuối của từng mô hình (seed)
            if seed == SEEDS[-1]: 
                dead = count_dead_neurons(res['model'], val_loader)['TOTAL']
                print(f"{act}: Chết {dead['dead']}/{dead['total']} Nơ-ron (Seed {seed})")
        
    plot_loss_curves(results_dying, 'Dying ReLU - Loss (LR=0.1)', 'fig_03_dying_loss.png')
    plot_accuracy_curves(results_dying, 'Dying ReLU - Accuracy (LR=0.1)', 'fig_06_dying_acc.png')

    print("\n--- 3. Mạng sâu (8 lớp) ---")
    results_deep = []
    for act in ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU']:
        for seed in SEEDS:
            res = run_training_experiment(act, DEEP_LAYERS, DEEP_HIDDEN, seed, train_loader, val_loader, test_loader)
            results_deep.append(res)
            
    plot_gradient_by_layer(results_deep, -1, 'Vanishing Gradient', 'fig_04_vanishing_grad.png')
    plot_gradient_evolution(results_deep, 'Gradient Evolution', 'fig_05_gradient_evo.png')
    print(create_summary_table(results_deep).to_string(index=False))
    
    print("\n--- 4. So sánh Mạng Nông vs Mạng Sâu ---")
    plot_shallow_vs_deep_accuracy(results_shallow, results_deep, save_name='fig_09_shallow_vs_deep.png')
    print("Hoàn tất kết xuất hình ảnh so sánh (Hình 9).")
    
    print("\nHoàn tất. Đồ thị đã lưu vào 'outputs/'.")

if __name__ == '__main__':
    main()
