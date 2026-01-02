import os
import scipy.io as sio
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from sklearn.manifold import TSNE

from cfo_est import estimate_cfo_multi_sts, compensate_cfo


class WiFiIQDataset(Dataset):
    def __init__(self, data_path, transform=None, max_samples_per_device=2000):
        self.data_path = data_path
        self.transform = transform
        self.max_samples_per_device = max_samples_per_device

        # 获取所有设备文件夹
        self.devices = []
        self.device_labels = {}
        self.data_files = []
        self.labels = []

        device_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

        for idx, device_dir in enumerate(device_dirs):
            device_path = os.path.join(data_path, device_dir)
            self.devices.append(device_dir)
            self.device_labels[device_dir] = idx

            # 获取该设备下的所有.mat文件
            mat_files = [f for f in os.listdir(device_path) if f.endswith('.mat')]
            # 只取前max_samples_per_device个文件
            mat_files = sorted(mat_files)[:max_samples_per_device]

            for mat_file in mat_files:
                file_path = os.path.join(device_path, mat_file)
                self.data_files.append(file_path)
                self.labels.append(idx)

        print(f"Total samples: {len(self.data_files)}, Devices: {len(self.devices)}")
        print(f"Device mapping: {self.device_labels}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        label = self.labels[idx]

        # 读取.mat文件
        data = sio.loadmat(file_path)
        sig_data = data['sig_data']

        fs = 20e6
        cfo_norm = estimate_cfo_multi_sts(sig_data, 10, 16)
        cfo_hz = cfo_norm * fs if not np.isnan(cfo_norm) else 0.0
        sig_comp = compensate_cfo(sig_data, cfo_hz, fs)
        sig_data = sig_comp

        # 确保是160长度的复数信号
        if sig_data.shape[0] != 160:
            raise ValueError(f"Expected 160 samples, got {sig_data.shape[0]} in {file_path}")



        # 转换为实虚部表示 (2, 160)
        iq_signal = np.stack([np.real(sig_data.flatten()), np.imag(sig_data.flatten())], axis=0)

        # 转换为tensor并归一化
        iq_tensor = torch.from_numpy(iq_signal).float()
        iq_tensor = (iq_tensor - iq_tensor.mean()) / (iq_tensor.std() + 1e-8)

        if self.transform:
            iq_tensor = self.transform(iq_tensor)

        return iq_tensor, label


# 简化版CNN模型
class WiFiCNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.5):
        super(WiFiCNN, self).__init__()

        # 输入: (2, 160) - 2通道复数信号的实部和虚部
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),  # (32, 160)
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # (32, 80)

            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # (64, 80)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # (64, 40)

            nn.Conv1d(64, 128, kernel_size=3, padding=1),  # (128, 40)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),  # (128, 20)
        )

        # 全连接层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),  # (128, 16)
            nn.Flatten(),  # 128 * 16 = 2048
            nn.Linear(128 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


# 梯度反转层 (GRL) - DANN核心组件
class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# DANN模型
class WiFiDANN(nn.Module):
    def __init__(self, num_classes=6, num_domains=3):
        super(WiFiDANN, self).__init__()

        # 特征提取器
        self.feature_extractor = WiFiCNN(num_classes=512)

        # 分类器
        self.classifier = nn.Linear(512, num_classes)

        # 域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_domains)
        )

        # 梯度反转层
        self.grl = GradientReverseLayer()

    def forward(self, x, alpha=1.0):
        # 提取特征
        features = self.feature_extractor(x)

        # 分类输出
        class_output = self.classifier(features)

        # 通过梯度反转层传递特征用于域分类
        reverse_features = self.grl.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output, features


# DANN训练函数
def train_dann_model(train_loaders, val_loader, model, device, num_epochs=50):
    """训练DANN模型"""
    model = model.to(device)

    # 分别定义优化器
    optimizer = optim.Adam(list(model.feature_extractor.parameters()) +
                           list(model.classifier.parameters()), lr=0.001, weight_decay=1e-5)
    optimizer_domain = optim.Adam(model.domain_classifier.parameters(), lr=0.001, weight_decay=1e-5)

    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    scheduler_domain = StepLR(optimizer_domain, step_size=20, gamma=0.5)

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    domain_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_class_loss = 0.0
        running_domain_loss = 0.0
        correct_class = 0
        correct_domain = 0
        total_class = 0
        total_domain = 0

        # 动态调整梯度反转层的alpha值
        p = float(epoch) / num_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1  # 从0逐渐增加到1

        # 合并所有训练数据
        all_train_data = []
        for loader in train_loaders.values():
            all_train_data.extend(list(loader))

        for batch_idx, (data, target) in enumerate(tqdm(all_train_data, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            data, target = data.to(device), target.to(device)

            # 为每个域创建域标签 (这里简化处理，实际需要更复杂的域标签)
            batch_size = data.size(0)
            domain_labels = torch.zeros(batch_size, dtype=torch.long).to(device)  # 假设为第一个域

            # 更新分类器
            optimizer.zero_grad()
            class_output, domain_output, features = model(data, alpha)

            class_loss = criterion_class(class_output, target)
            # 在更新分类器时，域损失应该是最小化域分类（对抗目标）
            domain_loss = -criterion_domain(domain_output, domain_labels)  # 负号用于对抗训练
            total_loss = class_loss + domain_loss
            total_loss.backward()
            optimizer.step()

            # 更新域分类器
            optimizer_domain.zero_grad()
            class_output, domain_output, features = model(data, alpha)
            domain_loss = criterion_domain(domain_output, domain_labels)
            domain_loss.backward()
            optimizer_domain.step()

            running_class_loss += class_loss.item()
            running_domain_loss += domain_loss.item()

            # 计算准确率
            _, predicted_class = class_output.max(1)
            _, predicted_domain = domain_output.max(1)

            total_class += target.size(0)
            correct_class += predicted_class.eq(target).sum().item()
            total_domain += domain_labels.size(0)
            correct_domain += predicted_domain.eq(domain_labels).sum().item()

        avg_class_loss = running_class_loss / len(all_train_data)
        avg_domain_loss = running_domain_loss / len(all_train_data)
        class_acc = 100. * correct_class / total_class
        domain_acc = 100. * correct_domain / total_domain

        train_losses.append(avg_class_loss)
        train_accuracies.append(class_acc)
        domain_accuracies.append(domain_acc)

        # 验证阶段
        val_acc = evaluate_model(model, val_loader, device, is_dann=True)
        val_accuracies.append(val_acc)

        scheduler.step()
        scheduler_domain.step()

        print(f'Epoch {epoch + 1}/{num_epochs}: Class Loss: {avg_class_loss:.4f}, '
              f'Class Acc: {class_acc:.2f}%, Domain Acc: {domain_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    return train_losses, train_accuracies, val_accuracies, domain_accuracies


def evaluate_model(model, data_loader, device, is_dann=False):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if is_dann:
                output, _, _ = model(data)
            else:
                output = model(data)

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def test_cross_domain_performance(train_path, test_paths, device, use_dann=True):
    """测试跨域性能"""
    # 创建数据集
    train_dataset = WiFiIQDataset(train_path)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

    if use_dann:
        # 使用DANN模型
        model = WiFiDANN(num_classes=6, num_domains=len(test_paths) + 1)  # 包括训练域
        train_loaders = {'train': train_loader}
        train_losses, train_accuracies, val_accuracies, domain_accuracies = train_dann_model(
            train_loaders, val_loader, model, device)
    else:
        # 使用普通CNN模型
        model = WiFiCNN(num_classes=6)
        train_losses, train_accuracies, val_accuracies = train_model(
            train_loader, val_loader, model, device)

    # 保存模型
    torch.save(model.state_dict(), 'wifi_fingerprint_model_dann.pth')
    print("Model saved as wifi_fingerprint_model_dann.pth")

    # 测试跨域性能
    results = {}
    model.load_state_dict(torch.load('wifi_fingerprint_model_dann.pth'))

    for domain_name, test_path in test_paths.items():
        print(f"\nTesting on {domain_name}...")
        test_dataset = WiFiIQDataset(test_path)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

        test_acc = evaluate_model(model, test_loader, device, is_dann=use_dann)
        results[domain_name] = test_acc
        print(f"{domain_name} Test Accuracy: {test_acc:.2f}%")

    return results, model, (train_losses, train_accuracies, val_accuracies)


def train_model(train_loader, val_loader, model, device, num_epochs=50):
    """训练普通CNN模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证阶段
        val_acc = evaluate_model(model, val_loader, device, is_dann=False)
        val_accuracies.append(val_acc)

        scheduler.step()

        print(
            f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

    return train_losses, train_accuracies, val_accuracies


def visualize_results(train_losses, train_accuracies, val_accuracies, results, use_dann=False):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 训练损失
    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title(f'Training Loss ({"DANN" if use_dann else "CNN"})')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # 训练准确率
    axes[0, 1].plot(train_accuracies, label='Train')
    axes[0, 1].plot(val_accuracies, label='Validation')
    axes[0, 1].set_title(f'Training and Validation Accuracy ({"DANN" if use_dann else "CNN"})')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 跨域测试结果
    domains = list(results.keys())
    accuracies = list(results.values())
    bars = axes[1, 0].bar(domains, accuracies)
    axes[1, 0].set_title(f'Cross-Domain Test Accuracy ({"DANN" if use_dann else "CNN"})')
    axes[1, 0].set_xlabel('Domain')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_ylim(0, 100)

    # 在柱状图上添加数值
    for bar, acc in zip(bars, accuracies):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom')

    # 保存模型性能对比
    axes[1, 1].axis('off')
    performance_text = f"Performance Summary ({'DANN' if use_dann else 'CNN'}):\n\n"
    for domain, acc in results.items():
        performance_text += f"{domain}: {acc:.2f}%\n"
    axes[1, 1].text(0.1, 0.5, performance_text, fontsize=14, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    plt.tight_layout()
    plt.savefig(f'wifi_fingerprint_results_{"dann" if use_dann else "cnn"}.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据路径
    base_path = r"C:\Users\arise\Desktop\wifi_audio"
    train_path = os.path.join(base_path, "Config_wifi_20M", "2025_12_23_wifi_IQ")
    test_paths = {
        "Config_wifi_10M": os.path.join(base_path, "Config_wifi_10M", "2025_12_23_wifi_IQ"),
        "Config_wifi_5M": os.path.join(base_path, "Config_wifi_5M", "2025_12_23_wifi_IQ")
    }

    # 验证路径是否存在
    if not os.path.exists(train_path):
        print(f"Training path does not exist: {train_path}")
        return
    for name, path in test_paths.items():
        if not os.path.exists(path):
            print(f"Test path does not exist: {path}")
            return

    print("Starting WiFi fingerprint recognition experiment with DANN...")

    # 测试DANN模型
    results_dann, model_dann, training_metrics_dann = test_cross_domain_performance(
        train_path, test_paths, device, use_dann=True)

    # 可视化DANN结果
    train_losses, train_accuracies, val_accuracies = training_metrics_dann
    visualize_results(train_losses, train_accuracies, val_accuracies, results_dann, use_dann=True)

    print("\n" + "=" * 50)
    print("DANN FINAL RESULTS:")
    for domain, acc in results_dann.items():
        print(f"{domain}: {acc:.2f}% accuracy")

    # 计算平均准确率
    avg_acc_dann = sum(results_dann.values()) / len(results_dann)
    print(f"DANN Average cross-domain accuracy: {avg_acc_dann:.2f}%")

    # 作为对比，也测试普通CNN模型
    print("\nTesting CNN model for comparison...")
    results_cnn, model_cnn, training_metrics_cnn = test_cross_domain_performance(
        train_path, test_paths, device, use_dann=False)

    # 可视化CNN结果
    train_losses, train_accuracies, val_accuracies = training_metrics_cnn
    visualize_results(train_losses, train_accuracies, val_accuracies, results_cnn, use_dann=False)

    print("\nCNN FINAL RESULTS:")
    for domain, acc in results_cnn.items():
        print(f"{domain}: {acc:.2f}% accuracy")

    avg_acc_cnn = sum(results_cnn.values()) / len(results_cnn)
    print(f"CNN Average cross-domain accuracy: {avg_acc_cnn:.2f}%")

    # 比较结果
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS:")
    print(f"DANN Average Accuracy: {avg_acc_dann:.2f}%")
    print(f"CNN Average Accuracy: {avg_acc_cnn:.2f}%")
    print(f"DANN Improvement: {avg_acc_dann - avg_acc_cnn:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()



