import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import os

def get_dataloaders(batch_size, data_dir='./data', val_split=0.1, seed=42):
    """
    Tải tập dữ liệu MNIST, chia tách Validation và bọc thành DataLoader.
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Tải dữ liệu
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Chia Validation theo tỷ lệ phần trăm (val_split)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    split = int(np.floor(val_split * num_train))
    train_idx, val_idx = indices[split:], indices[:split]
    
    # Bộ lấy mẫu độc lập
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Khởi tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
