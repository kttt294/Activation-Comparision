import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from config import DEVICE, INPUT_SIZE, OUTPUT_SIZE, MOMENTUM, LR_DEFAULT, EPOCHS
from models import MLP, ACTIVATIONS

def get_gradient_stats(model):
    # Đo lường cường độ lan truyền ngược (Gradient) tại mỗi lớp
    stats = []
    layer_idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad = param.grad.data
            stats.append({
                'layer_idx': layer_idx,
                'name': name,
                'mean_abs': grad.abs().mean().item(),
                'std': grad.std().item()
            })
            layer_idx += 1
    return stats

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    grad_stats_batch = []
    
    for inputs, labels in loader:
        inputs = inputs.view(inputs.size(0), -1).to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Lan truyền ngược
        loss.backward()
        
        # Ghi nhận trạng thái gradient trước khi update
        grad_stats_batch.append(get_gradient_stats(model))
        
        # Cập nhật trọng số
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += inputs.size(0)
        
    return running_loss/total, correct/total, grad_stats_batch[-1]

def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.view(inputs.size(0), -1).to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += inputs.size(0)
    return running_loss/total, correct/total

def count_dead_neurons(model, loader):
    # Kiểm tra và đếm các nơ-ron bị kẹt ở trạng thái Dying ReLU
    model.eval()
    dead_stats = {}
    activation_sums = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if name not in activation_sums:
                activation_sums[name] = torch.zeros(output.size(1), device=output.device)
            activation_sums[name] += (output > 0).sum(dim=0)
        return hook
        
    hooks = []
    layer_idx = 0
    # Gắn máy nghe lén (hook) vào các lớp kích hoạt họ ReLU
    for name, module in model.network.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            hooks.append(module.register_forward_hook(get_activation(f'Layer_{layer_idx}')))
            layer_idx += 1
            
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.view(inputs.size(0), -1).to(DEVICE)
            model(inputs)
            
    # Xóa hook để tiết kiệm RAM
    for h in hooks: h.remove()
    
    total_dead, total_neurons = 0, 0
    for name, acts in activation_sums.items():
        dead_count = (acts == 0).sum().item()
        total_count = acts.size(0)
        dead_stats[name] = {'dead': dead_count, 'total': total_count}
        total_dead += dead_count
        total_neurons += total_count
        
    dead_stats['TOTAL'] = {'dead': total_dead, 'total': total_neurons}
    return dead_stats

def run_training_experiment(activation_name, num_hidden_layers, hidden_size, seed, train_loader, val_loader, test_loader, epochs=EPOCHS, lr=LR_DEFAULT):
    # Hàm lõi gói gọn toàn bộ quy trình chạy thực nghiệm cho 1 mô hình
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    
    # Chuẩn bị model, Loss và Optimizer
    act_cls = ACTIVATIONS[activation_name]
    model = MLP(INPUT_SIZE, hidden_size, num_hidden_layers, OUTPUT_SIZE, act_cls).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
    
    # Từ điển lưu dấu vết học
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'grad_stats':[]}
    start_time = time.time()
    
    for epoch in range(epochs):
        tr_loss, tr_acc, g_stats = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)
        history['grad_stats'].append(g_stats)
        
    elapsed = time.time() - start_time
    ts_loss, ts_acc = evaluate(model, test_loader, criterion)
    
    return {
        'activation': activation_name,
        'seed': seed,
        'test_acc': ts_acc,
        'time': elapsed,
        'history': history,
        'model': model
    }
