import torch

# Random seeds để báo cáo kết quả ổn định (mean ± std)
SEEDS = [42, 123, 456]

# Siêu tham số cấu hình (Hyperparameters)
BATCH_SIZE = 128
EPOCHS = 50
LR_DEFAULT = 0.01
MOMENTUM = 0.9

# Cấu hình Mạng nơ-ron
SHALLOW_LAYERS = 3
SHALLOW_HIDDEN = 256
DEEP_LAYERS = 8
DEEP_HIDDEN = 256
INPUT_SIZE = 784
OUTPUT_SIZE = 10

# Tự động nhận diện thiết bị tính toán (GPU hay CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Màu đồ thị
COLORS = {
    'Sigmoid': '#1f77b4',
    'Tanh': '#2ca02c',
    'ReLU': '#ff7f0e',
    'Leaky ReLU': '#d62728'
}
