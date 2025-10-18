#!/usr/bin/env python3
"""
花卉分类模型训练脚本
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.functional as F

from model import create_model
from utils import (
    create_data_loaders, set_seed, AverageMeter, calculate_accuracy,
    save_config, plot_training_history, evaluate_model
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # 打印进度
        if batch_idx % 50 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} '
                  f'Acc@1: {top1.avg:.2f}% '
                  f'Acc@5: {top5.avg:.2f}%')

    return losses.avg, top1.avg.item()


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 计算准确率
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

            # 更新统计
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    return losses.avg, top1.avg.item()


def mixup_data(x, y, alpha=0.2, device='cpu'):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup损失函数"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch_with_mixup(model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=0.2):
    """使用Mixup训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Mixup数据增强
        images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, device)
        images, labels_a, labels_b = map(torch.autograd.Variable, (images, labels_a, labels_b))

        # 前向传播
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率（使用原始标签）
        acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # 打印进度
        if batch_idx % 50 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} '
                  f'Acc@1: {top1.avg:.2f}% '
                  f'Acc@5: {top5.avg:.2f}%')

    return losses.avg, top1.avg.item()


def main():
    parser = argparse.ArgumentParser(description='花卉分类模型训练')
    parser.add_argument('--data_dir', type=str, default='../unified_flower_dataset',
                        help='数据集根目录')
    parser.add_argument('--model_type', type=str, default='efficientnet_b3',
                        choices=['resnet50', 'resnet101', 'efficientnet_b3', 'efficientnet_b4'],
                        help='模型类型')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='类别数量')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--img_size', type=int, default=300,
                        help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='../model',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha参数')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['step', 'plateau', 'cosine'],
                        help='学习率调度器类型')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据路径
    train_csv = os.path.join(args.data_dir, 'train_labels.csv')
    test_csv = os.path.join(args.data_dir, 'test_labels.csv')
    train_img_dir = os.path.join(args.data_dir, 'images', 'train')
    test_img_dir = os.path.join(args.data_dir, 'images', 'test')

    # 创建数据加载器
    print("Creating data loaders...")
    train_loader, val_loader, class_to_idx = create_data_loaders(
        train_csv, test_csv, train_img_dir, test_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")

    # 创建模型
    print(f"Creating {args.model_type} model...")
    model = create_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        pretrained=True
    )
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(  # 使用AdamW优化器
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    else:  # cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            history = checkpoint.get('history', history)
            print(f"Loaded checkpoint (epoch {start_epoch}, best_acc: {best_acc:.2f}%)")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # 训练循环
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # 训练（使用Mixup）
        if args.mixup_alpha > 0:
            train_loss, train_acc = train_epoch_with_mixup(
                model, train_loader, criterion, optimizer, device, epoch, args.mixup_alpha
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )

        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # 更新学习率
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - start_time

        print(f'Epoch: [{epoch+1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f} '
              f'Val Acc: {val_acc:.2f}% '
              f'Time: {epoch_time:.1f}s '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # 保存最佳模型
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'history': history,
            'config': vars(args)
        }

        # 保存最新检查点
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_checkpoint.pth'))

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    print(f'Training completed. Best validation accuracy: {best_acc:.2f}%')

    # 保存配置文件
    config = {
        'model_type': args.model_type,
        'num_classes': args.num_classes,
        'img_size': args.img_size,
        'class_to_idx': class_to_idx,
        'best_accuracy': best_acc,
        'training_params': vars(args)
    }
    save_config(config, os.path.join(args.save_dir, 'config.json'))

    # 绘制训练历史
    plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))

    # 最终评估
    print("Final evaluation on test set...")
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    test_acc, test_report, _, _ = evaluate_model(model, val_loader, device, class_to_idx)
    print(f"Final test accuracy: {test_acc:.4f}")

    # 保存测试报告
    with open(os.path.join(args.save_dir, 'test_report.json'), 'w') as f:
        import json
        json.dump(test_report, f, indent=4)


if __name__ == '__main__':
    main()