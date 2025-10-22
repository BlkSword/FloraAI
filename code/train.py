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
from math import cos, pi


class WarmupScheduler:
    """学习率预热调度器"""
    
    def __init__(self, optimizer, warmup_epochs, base_lr, scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.scheduler = scheduler
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        # 预热阶段
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # 正常调度阶段
        elif self.scheduler is not None:
            self.scheduler.step()
            
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class CosineAnnealingWarmRestartsWithWarmup:
    """带预热的余弦退火重启调度器"""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup_epochs=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.last_epoch = last_epoch
        self.T_cur = last_epoch
        self.base_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        
    def step(self):
        self.last_epoch += 1
        
        # 预热阶段
        if self.last_epoch < self.warmup_epochs:
            lr = self.base_lrs[0] * (self.last_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return
            
        # 正常调度阶段
        if self.last_epoch == self.warmup_epochs:
            self.T_cur = 0
        else:
            self.T_cur += 1
            
        if self.T_cur >= self.T_0:
            self.T_cur = 0
            self.T_0 *= self.T_mult
            
        lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * (1 + cos(pi * self.T_cur / self.T_0)) / 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class FocalLoss(nn.Module):
    """焦点损失函数，用于处理类别不平衡问题"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

from model import create_model
from utils import (
    create_data_loaders, set_seed, AverageMeter, calculate_accuracy,
    save_config, plot_training_history, evaluate_model
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, grad_clip=1.0, scheduler=None):
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
        
        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # 对于OneCycle调度器，每个batch后更新学习率
        if scheduler is not None and hasattr(scheduler, 'batch_step'):
            scheduler.step()

        # 计算准确率
        acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # 打印进度
        if batch_idx % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                f'Loss: {losses.avg:.4f} '
                f'Acc@1: {top1.avg:.2f}% '
                f'Acc@5: {top5.avg:.2f}% '
                f'LR: {current_lr:.6f}')

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


def train_epoch_with_mixup(model, train_loader, criterion, optimizer, device, epoch, mixup_alpha=0.2, grad_clip=1.0, scheduler=None):
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
        
        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # 对于OneCycle调度器，每个batch后更新学习率
        if scheduler is not None and hasattr(scheduler, 'batch_step'):
            scheduler.step()

        # 计算准确率（使用原始标签）
        acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # 打印进度
        if batch_idx % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                f'Loss: {losses.avg:.4f} '
                f'Acc@1: {top1.avg:.2f}% '
                f'Acc@5: {top5.avg:.2f}% '
                f'LR: {current_lr:.6f}')

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
    parser.add_argument('--num_workers', type=int, default=8,
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
                        choices=['step', 'plateau', 'cosine', 'cosine_warmup', 'onecycle'],
                        help='学习率调度器类型')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                        help='优化器类型')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪阈值')
    parser.add_argument('--use_advanced_aug', action='store_true',
                        help='使用高级数据增强')
    parser.add_argument('--use_attention', action='store_true',
                        help='使用注意力机制')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal', 'label_smooth'], help='损失函数类型: ce(交叉熵), focal(焦点损失), label_smooth(标签平滑)')
    parser.add_argument('--focal_alpha', type=float, default=1.0, help='Focal Loss的alpha参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss的gamma参数')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='标签平滑参数')

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
        img_size=args.img_size,
        use_advanced_aug=args.use_advanced_aug
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")

    # 读取训练CSV文件以获取类别名称信息
    import pandas as pd
    train_df = pd.read_csv(train_csv)
    class_names = {}
    for category_id in class_to_idx.keys():
        category_data = train_df[train_df['category_id'] == category_id].iloc[0]
        class_names[category_id] = {
            'chinese_name': category_data['chinese_name'],
            'english_name': category_data['english_name']
        }

    # 创建模型
    print(f"Creating {args.model_type} model...")
    model = create_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        pretrained=True,
        dropout=0.5,
        use_attention=args.use_attention
    )
    model = model.to(device)

    # 损失函数
    if args.loss_type == 'focal':
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"使用Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    elif args.loss_type == 'label_smooth':
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        print(f"使用标签平滑交叉熵 (smoothing={args.label_smoothing})")
    elif args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"使用PyTorch内置标签平滑交叉熵 (smoothing={args.label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("使用标准交叉熵损失")

    # 优化器选择
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    else:  # rmsprop
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9
        )

    # 学习率调度器
    if args.scheduler == 'step':
        base_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        if args.warmup_epochs > 0:
            scheduler = WarmupScheduler(optimizer, args.warmup_epochs, args.lr, base_scheduler)
            print(f"使用StepLR+Warmup调度器 (warmup_epochs={args.warmup_epochs}, step_size=30, gamma=0.1)")
        else:
            scheduler = base_scheduler
            print("使用StepLR调度器 (step_size=30, gamma=0.1)")
    elif args.scheduler == 'plateau':
        base_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        if args.warmup_epochs > 0:
            scheduler = WarmupScheduler(optimizer, args.warmup_epochs, args.lr, base_scheduler)
            print(f"使用ReduceLROnPlateau+Warmup调度器 (warmup_epochs={args.warmup_epochs}, patience=5, factor=0.5)")
        else:
            scheduler = base_scheduler
            print("使用ReduceLROnPlateau调度器 (patience=5, factor=0.5)")
    elif args.scheduler == 'cosine':
        base_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        if args.warmup_epochs > 0:
            scheduler = WarmupScheduler(optimizer, args.warmup_epochs, args.lr, base_scheduler)
            print(f"使用CosineAnnealingLR+Warmup调度器 (warmup_epochs={args.warmup_epochs}, T_max={args.epochs}, eta_min=1e-6)")
        else:
            scheduler = base_scheduler
            print(f"使用CosineAnnealingLR调度器 (T_max={args.epochs}, eta_min=1e-6)")
    elif args.scheduler == 'cosine_warmup':
        scheduler = CosineAnnealingWarmRestartsWithWarmup(
            optimizer, T_0=args.epochs//3, T_mult=2, eta_min=1e-6, 
            warmup_epochs=args.warmup_epochs
        )
        print(f"使用CosineAnnealingWarmRestartsWithWarmup调度器 (warmup_epochs={args.warmup_epochs}, T_0={args.epochs//3})")
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr * 10, epochs=args.epochs, 
            steps_per_epoch=len(train_loader), pct_start=0.3
        )
        print(f"使用OneCycleLR调度器 (max_lr={args.lr * 10}, pct_start=0.3)")
    else:
        scheduler = None
        print("不使用学习率调度器")

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
                model, train_loader, criterion, optimizer, device, epoch, 
                args.mixup_alpha, args.grad_clip, scheduler
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch,
                args.grad_clip, scheduler
            )

        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # 更新学习率
        if args.scheduler == 'plateau':
            # 对于ReduceLROnPlateau，需要传入验证损失
            if isinstance(scheduler, WarmupScheduler):
                # 如果使用了预热包装器，需要调用内部调度器
                if epoch > args.warmup_epochs:
                    scheduler.scheduler.step(val_loss)
            else:
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

        # 保存检查点
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_checkpoint.pth'))

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    # 所有训练完成后保存训练历史图表
    plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))

    print(f'Training completed. Best validation accuracy: {best_acc:.2f}%')

    # 保存配置文件
    config = {
        'model_type': args.model_type,
        'num_classes': args.num_classes,
        'img_size': args.img_size,
        'class_to_idx': class_to_idx,
        'class_names': class_names,  # 添加类别名称信息
        'best_accuracy': best_acc,
        'training_params': vars(args)
    }
    save_config(config, os.path.join(args.save_dir, 'config.json'))

    # 绘制训练历史
    # plot_training_history(history, os.path.join(args.save_dir, 'training_history.png'))

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