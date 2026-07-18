import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

# Từ điển ánh xạ hàm kích hoạt
ACTIVATIONS = {
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'ReLU': nn.ReLU,
    'Leaky ReLU': nn.LeakyReLU
}

class MLP(nn.Module):
    # Mô hình mạng Multi-Layer Perceptron với khả năng tùy biến cấu trúc
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, activation_cls):
        super().__init__()
        self.activation = activation_cls()
        
        layers = []
        in_features = input_size
        
        # Cấu trúc các lớp ẩn
        for _ in range(num_hidden_layers):
            linear = nn.Linear(in_features, hidden_size)
            self._init_weights(linear, activation_cls)
            layers.append(linear)
            layers.append(self.activation)
            in_features = hidden_size
            
        # Lớp đầu ra (Không kèm activation để tính Cross-Entropy chuẩn)
        final_layer = nn.Linear(in_features, output_size)
        self._init_weights(final_layer, activation_cls)
        layers.append(final_layer)
        
        self.network = nn.Sequential(*layers)
        
    def _init_weights(self, m, activation_cls):
        """Khởi tạo trọng số thông minh theo đặc tính của hàm kích hoạt"""
        if isinstance(m, nn.Linear):
            # Sigmoid và Tanh phù hợp với Xavier (Glorot)
            if activation_cls in [nn.Sigmoid, nn.Tanh]:
                init.xavier_uniform_(m.weight)
            # ReLU và Leaky ReLU phù hợp với Kaiming He
            else:
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
                
    def forward(self, x):
        return self.network(x)


# CÁC HÀM TRỰC QUAN HÓA TOÁN HỌC DÙNG NUMPY

def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_grad(x): s = sigmoid_np(x); return s * (1.0 - s)
def tanh_np(x): return np.tanh(x)
def tanh_grad(x): return 1.0 - np.tanh(x) ** 2
def relu_np(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)
def leaky_relu_np(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def leaky_relu_grad(x, alpha=0.01): return np.where(x > 0, 1.0, alpha)
