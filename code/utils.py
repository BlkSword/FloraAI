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
from torchvision.transforms import RandAugment, RandomErasing
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from torch.cuda.amp import autocast
from contextlib import nullcontext


class FlowerDataset(Dataset):
    """花卉数据集类 - 用于有标签的训练数据"""

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # 创建类别ID到索引的映射（统一为Python原生int，避免JSON序列化问题）
        unique_categories = sorted(int(cat_id) for cat_id in self.data['category_id'].unique())
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

        # 获取标签索引（确保为Python原生int）
        category_id = int(row['category_id'])
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


def get_transforms(phase='train', img_size=384, use_advanced_aug=True):
    """获取数据变换，支持高级数据增强"""

    if phase == 'train':
        if use_advanced_aug:
            # 高级数据增强策略
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                # RandAugment数据增强
                RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Random Erasing数据增强
                RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])
        else:
            # 基本数据增强
            transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:  # val or test
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform


def create_data_loaders(train_csv, test_csv, train_img_dir, test_img_dir,
                       batch_size=32, num_workers=4, img_size=384, use_advanced_aug=True,
                       mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1):
    """创建数据加载器，支持高级数据增强和混合策略"""

    # 获取变换
    train_transform = get_transforms('train', img_size, use_advanced_aug)
    test_transform = get_transforms('test', img_size)

    # 创建数据集
    train_dataset = FlowerDataset(train_csv, train_img_dir, train_transform)
    test_dataset = FlowerDataset(test_csv, test_img_dir, test_transform)

    # 创建Mixup和CutMix增强器
    mixup_fn = None
    if use_advanced_aug and (mixup_alpha > 0 or cutmix_alpha > 0):
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=label_smoothing,
            num_classes=len(train_dataset.class_to_idx)
        )
        
        # 当使用mixup时，确保batch size为偶数
        # 通过设置drop_last=True来避免最后一个batch大小为奇数的情况
        drop_last = True
    else:
        drop_last = False

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last  # 当使用mixup时，丢弃最后一个不完整的batch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.class_to_idx, mixup_fn


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


class EMA:
    """指数移动平均(Exponential Moving Average)模型权重"""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """注册EMA参数"""
        for name, param in self.model.named_parameters():
            # 仅在缺失时注册，避免覆盖已有的EMA平均值
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 动态注册缺失的参数，避免阶段切换导致断言失败
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用EMA参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始模型参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


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


def tta_inference(model, image, device, num_augments=2):
    """使用测试时增强(TTA)进行推理
    
    Args:
        model: 模型
        image: 输入图像张量 [C, H, W]
        device: 设备
        num_augments: 增强次数
        
    Returns:
        平均预测结果
    """
    model.eval()
    
    # 原始图像预测
    with torch.no_grad():
        with amp_autocast(device):  # 使用FP16加速推理（仅 CUDA）
            outputs = model(image.unsqueeze(0).to(device))
    
    # 水平翻转预测
    if num_augments >= 2:
        flip_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
        ])
        flipped_image = flip_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            with amp_autocast(device):  # 使用FP16加速推理（仅 CUDA）
                flip_outputs = model(flipped_image)
        
        # 合并预测结果
        outputs = (outputs + flip_outputs) / 2.0
    
    return outputs


def evaluate_model(model, test_loader, device, class_to_idx, use_tta=True, use_fp16=True):
    """评估模型性能，支持TTA和FP16"""
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_tta:
                # 使用TTA进行推理
                batch_outputs = []
                for i in range(images.size(0)):
                    outputs = tta_inference(model, images[i], device, num_augments=2)
                    batch_outputs.append(outputs)
                outputs = torch.cat(batch_outputs, dim=0)
            else:
                # 普通推理
                if use_fp16 and getattr(device, 'type', 'cpu') == 'cuda':
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filenames.extend(filenames)

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
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 创建预测结果DataFrame
    results_df = pd.DataFrame({
        'filename': all_filenames,
        'true_label': [idx_to_class[label] for label in all_labels],
        'pred_label': [idx_to_class[pred] for pred in all_preds],
        'correct': [1 if pred == label else 0 for pred, label in zip(all_preds, all_labels)]
    })

    return accuracy, report, cm, results_df


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
def amp_autocast(device):
    """在 CUDA 可用时启用 autocast，否则使用空上下文。"""
    return autocast() if torch.cuda.is_available() and getattr(device, 'type', 'cpu') == 'cuda' else nullcontext()