from config import *
from data_loader import get_dataloaders
from engine import run_training_experiment, count_dead_neurons
from visualization import *

def main():
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    
    print("--- 1. Mạng nông (3 lớp) ---")
    results_shallow = []
    for act in ['Sigmoid', 'Tanh', 'ReLU']:
        res = run_training_experiment(act, SHALLOW_LAYERS, SHALLOW_HIDDEN, SEEDS[0], train_loader, val_loader, test_loader)
        results_shallow.append(res)
        
    plot_loss_curves(results_shallow, 'Loss (Mạng Nông)', 'fig_01_shallow_loss.png')
    plot_accuracy_curves(results_shallow, 'Accuracy (Mạng Nông)', 'fig_02_shallow_acc.png')
    print(create_summary_table(results_shallow).to_string(index=False))

    print("\n--- 2. Dying ReLU (LR = 0.1) ---")
    results_dying = []
    for act in ['ReLU', 'Leaky ReLU']:
        res = run_training_experiment(act, DEEP_LAYERS, DEEP_HIDDEN, SEEDS[0], train_loader, val_loader, test_loader, epochs=10, lr=0.1)
        results_dying.append(res)
        dead = count_dead_neurons(res['model'], val_loader)['TOTAL']
        print(f"{act}: Chết {dead['dead']}/{dead['total']} Nơ-ron")
        
    plot_loss_curves(results_dying, 'Dying ReLU (LR=0.1)', 'fig_03_dying_loss.png')

    print("\n--- 3. Mạng sâu (8 lớp) ---")
    results_deep = []
    for act in ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU']:
        res = run_training_experiment(act, DEEP_LAYERS, DEEP_HIDDEN, SEEDS[0], train_loader, val_loader, test_loader)
        results_deep.append(res)
            
    plot_gradient_by_layer(results_deep, -1, 'Vanishing Gradient', 'fig_04_vanishing_grad.png')
    plot_gradient_evolution(results_deep, 'Gradient Evolution', 'fig_05_gradient_evo.png')
    print(create_summary_table(results_deep).to_string(index=False))
    
    print("\nHoàn tất. Đồ thị đã lưu vào 'outputs/'.")

if __name__ == '__main__':
    main()
