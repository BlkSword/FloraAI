#!/usr/bin/env python3
"""
工具函数模块
包含数据处理、可视化、评估等功能
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class FlowerDataset(Dataset):
    """花卉数据集类 - 用于有标签的训练数据"""

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 创建类别ID到索引的映射
        unique_categories = sorted(self.data['category_id'].unique())
        self.class_to_idx = {cat_id: idx for idx, cat_id in enumerate(unique_categories)}
        self.idx_to_class = {idx: cat_id for cat_id, idx in self.class_to_idx.items()}

        print(f"Dataset loaded: {len(self.data)} samples, {len(unique_categories)} classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为fallback
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取标签索引
        category_id = row['category_id']
        label = self.class_to_idx[category_id]

        return image, label, row['filename']


class UnlabeledDataset(Dataset):
    """无标签数据集类 - 用于真实预测场景"""

    def __init__(self, img_dir, transform=None, img_extensions=None):
        if img_extensions is None:
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

        self.img_dir = img_dir
        self.transform = transform

        # 获取所有图片文件
        self.image_files = []
        for ext in img_extensions:
            self.image_files.extend(Path(img_dir).glob(f'*{ext}'))
            self.image_files.extend(Path(img_dir).glob(f'*{ext.upper()}'))

        # 排序确保一致性
        self.image_files = sorted([str(f.name) for f in self.image_files])

        print(f"Found {len(self.image_files)} images for prediction")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.img_dir, filename)

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像作为fallback
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, filename


class AdvancedTransform:
    """高级数据增强变换"""
    
    def __init__(self, img_size=300, phase='train'):
        self.img_size = img_size
        self.phase = phase
        
    def __call__(self, image):
        if self.phase == 'train':
            # 随机调整大小和裁剪
            image = TF.resize(image, (int(self.img_size * 1.2), int(self.img_size * 1.2)))
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
            image = TF.crop(image, i, j, h, w)
            
            # 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                
            # 随机旋转
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                
            # 颜色抖动
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
                
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
                
            if random.random() > 0.5:
                image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))
                
            # 转换为张量并标准化
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            return image
        else:
            # 验证/测试阶段
            image = TF.resize(image, (self.img_size, self.img_size))
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            return image


def get_transforms(phase='train', img_size=300):
    """获取数据变换"""
    
    if phase == 'train':
        transform = AdvancedTransform(img_size=img_size, phase='train')
    else:  # val or test
        transform = AdvancedTransform(img_size=img_size, phase='val')

    return transform


def create_data_loaders(train_csv, test_csv, train_img_dir, test_img_dir,
                       batch_size=32, num_workers=4, img_size=300):
    """创建数据加载器"""

    # 获取变换
    train_transform = get_transforms('train', img_size)
    test_transform = get_transforms('test', img_size)

    # 创建数据集
    train_dataset = FlowerDataset(train_csv, train_img_dir, train_transform)
    test_dataset = FlowerDataset(test_csv, test_img_dir, test_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.class_to_idx


def set_seed(seed=42):
    """设置随机种子确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """记录平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets, topk=(1, 5)):
    """计算Top-K准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_config(config, save_path):
    """保存配置文件"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_predictions(model, dataset, device, class_to_idx, num_samples=8):
    """可视化预测结果"""
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    # 随机选择样本
    indices = random.sample(range(len(dataset)), num_samples)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label, filename = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)

            # 预测
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()

            # 转换为显示用的图像
            image_np = image.permute(1, 2, 0).numpy()
            image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            image_np = np.clip(image_np, 0, 1)

            # 显示
            axes[i].imshow(image_np)
            axes[i].set_title(f'True: {idx_to_class[true_label]}\nPred: {idx_to_class[predicted_label]}')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, device, class_to_idx):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 生成分类报告
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    target_names = [str(idx_to_class[i]) for i in range(len(class_to_idx))]

    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True
    )

    return accuracy, report, all_preds, all_labels


if __name__ == "__main__":
    # 测试数据集创建
    print("Testing utility functions...")

    # 设置随机种子
    set_seed(42)
    print("Random seed set to 42")

    # 测试变换
    train_transform = get_transforms('train')
    test_transform = get_transforms('test')
    print("Transforms created successfully")

    print("Utility functions test completed!")