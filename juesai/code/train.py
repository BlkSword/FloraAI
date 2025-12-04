#!/usr/bin/env python3
"""
花卉分类模型训练脚本
"""

import os
import time
import argparse
import warnings
from pathlib import Path
#from pydantic.warnings import UnsupportedFieldAttributeWarning

#warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils as nn_utils

from model import create_model
from utils import (
    create_data_loaders, set_seed, AverageMeter, calculate_accuracy,
    save_config, plot_training_history, evaluate_model, EMA
)

def build_optimizer(model, lr, weight_decay):
    """为不同模块设置分层学习率：backbone使用lr，分类头使用lr*2。"""
    head_params = []
    backbone_params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            tokens = name.split('.')
            is_head = (
                ('head' in tokens) or
                ('classifier' in tokens) or
                (tokens[-1] == 'fc')
            )
            if is_head:
                head_params.append(p)
            else:
                backbone_params.append(p)

    if len(head_params) == 0 or len(backbone_params) == 0:
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    optimizer = optim.AdamW(
        [
            {'params': backbone_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr * 2.0, 'weight_decay': weight_decay},
        ]
    )
    return optimizer


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
               mixup_fn=None, use_amp=True, ema=None):
    """训练一个epoch，支持Mixup/CutMix和FP16"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 应用Mixup和CutMix数据增强
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # 使用FP16混合精度训练（仅在 CUDA 上启用）
        if use_amp and device.type == 'cuda':
            with autocast():
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                        
            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # AMP下需先unscale再裁剪
            if scaler is not None:
                scaler.unscale_(optimizer)
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 更新EMA模型
        if ema is not None:
            ema.update()

        # 计算准确率
        if mixup_fn is not None:
            # 对于mixup，我们使用原始标签计算准确率
            acc1, acc5 = calculate_accuracy(outputs, torch.argmax(labels, dim=1), topk=(1, 5))
        else:
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # 打印进度
        if batch_idx % 20 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}] '
                  f'Loss: {losses.avg:.4f} '
                  f'Acc@1: {top1.avg:.2f}% '
                  f'Acc@5: {top5.avg:.2f}%')

    return losses.avg, top1.avg.item()


def validate_epoch(model, val_loader, criterion, device, use_amp=True, ema=None):
    """验证一个epoch，支持FP16和EMA"""
    # 如果使用EMA，应用EMA权重
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)

            # 使用FP16混合精度（仅在 CUDA 上启用）
            if use_amp and device.type == 'cuda':
                with autocast():
                    # 前向传播
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 计算准确率
            acc1, acc5 = calculate_accuracy(outputs, labels, topk=(1, 5))

            # 更新统计
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    if ema is not None:
        ema.restore()
        
    return losses.avg, top1.avg.item()


def main():
    parser = argparse.ArgumentParser(description='花卉分类模型训练')
    parser.add_argument('--data_dir', type=str, default='./unified_flower_dataset',
                        help='数据集根目录')
    parser.add_argument('--model_type', type=str, default='convnext_base_timm',
                        choices=['convnext_base', 'convnext_base_timm', 'resnet50', 'resnet101', 'efficientnet_b4'],
                        help='模型类型')
    parser.add_argument('--num_classes', type=int, default=152,
                        help='类别数量')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=80,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--min_lr', type=float, default=5e-6,
                        help='最小学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--img_size', type=int, default=512,
                        help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./model',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的模型路径')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='使用FP16混合精度训练')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='使用EMA模型平均')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA衰减率')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='标签平滑系数')
    parser.add_argument('--mixup_alpha', type=float, default=0.4,
                        help='Mixup alpha参数')
    parser.add_argument('--cutmix_alpha', type=float, default=0.3,
                        help='CutMix alpha参数')
    parser.add_argument('--use_advanced_aug', action='store_true', default=True,
                        help='使用高级数据增强')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=3,
                        help='前N个epoch冻结backbone，仅训练头部')

    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值，N个epoch无提升则停止训练')
    
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
    train_loader, val_loader, class_to_idx, mixup_fn = create_data_loaders(
        train_csv, test_csv, train_img_dir, test_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_advanced_aug=args.use_advanced_aug,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        label_smoothing=args.label_smoothing
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")

    # 强制使用数据集中的类别数，避免与 --num_classes 不一致
    args.num_classes = len(class_to_idx)
    print(f"Using num_classes from dataset: {args.num_classes}")

    # 创建模型
    print(f"Creating {args.model_type} model...")
    model = create_model(
        num_classes=args.num_classes,
        model_type=args.model_type,
        pretrained=True
    )
    model = model.to(device)

    def set_backbone_freeze(model, freeze=True):
        """冻结/解冻backbone，仅保留分类头可训练。
        兼容timm ConvNeXt、torchvision ResNet/ConvNeXt以及自定义封装。
        """
        total_params = 0
        trainable_params = 0
        trainable_names_preview = []
        frozen_names_preview = []

        for name, param in model.named_parameters():
            tokens = name.split('.')
            is_head = (
                ('head' in tokens) or
                ('fc' in tokens) or
                ('classifier' in tokens)
            )

            # 设置 requires_grad
            if freeze:
                param.requires_grad = is_head
            else:
                param.requires_grad = True

            # 统计信息与示例名称采样
            numel = param.numel()
            total_params += numel
            if param.requires_grad:
                trainable_params += numel
                if len(trainable_names_preview) < 6:
                    trainable_names_preview.append(name)
            else:
                if len(frozen_names_preview) < 6:
                    frozen_names_preview.append(name)

        phase = '仅训练分类头(冻结骨干)' if freeze else '全量微调(解冻骨干)'
        print(
            f"[Freeze] 阶段: {phase} | 可训练参数: {trainable_params:,}/{total_params:,} | "
            f"示例可训练: {trainable_names_preview} | 示例冻结: {frozen_names_preview}"
        )

    set_backbone_freeze(model, freeze=True)

    # 创建EMA模型
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        print(f"Using EMA with decay: {args.ema_decay}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 余弦学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    patience_counter = 0  # 早停计数器
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
            patience_counter = checkpoint.get('patience_counter', 0) # 恢复计数器
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            history = checkpoint.get('history', history)
            print(f"Loaded checkpoint (epoch {start_epoch}, best_acc: {best_acc:.2f}%)")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # 训练循环
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # 两阶段微调：前N个epoch只训练头部，之后解冻全量
        if epoch < args.freeze_backbone_epochs:
            set_backbone_freeze(model, freeze=True)
            freeze_status = 'head-only'
        else:
            set_backbone_freeze(model, freeze=False)
            freeze_status = 'full-finetune'
            # 在解冻切换的首个epoch，补充EMA跟踪的新可训练参数
            if args.use_ema and ema is not None and epoch == args.freeze_backbone_epochs:
                ema.register()

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            mixup_fn=mixup_fn, use_amp=args.use_amp, ema=ema
        )

        # 验证
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, 
            use_amp=args.use_amp, ema=ema if args.use_ema else None
        )

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

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
              f'LR: {current_lr:.6f} '
              f'Phase: {freeze_status} '
              f'Time: {epoch_time:.1f}s')

        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            patience_counter = 0  # 重置计数器
        else:
            # 仅在全量微调阶段才开始计算早停，避免热身阶段被误杀
            if freeze_status == 'full-finetune':
                patience_counter += 1
                print(f'EarlyStopping counter: {patience_counter} out of {args.patience}')

        # 保存检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'patience_counter': patience_counter, # 保存计数器状态
            'history': history,
            'config': vars(args)
        }
        
        # 如果使用EMA，保存EMA模型状态
        if args.use_ema and ema is not None:
            ema.apply_shadow()
            checkpoint['ema_model_state_dict'] = model.state_dict()
            ema.restore()

        # 保存最新检查点
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_checkpoint.pth'))

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

        # 检查早停条件
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

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
        if 'ema_model_state_dict' in checkpoint and args.use_ema:
            model.load_state_dict(checkpoint['ema_model_state_dict'])
            print("Using EMA model for final evaluation")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

    # 使用TTA和FP16进行最终评估
    test_acc, test_report, _, _ = evaluate_model(
        model, val_loader, device, class_to_idx, 
        use_tta=True, use_fp16=args.use_amp
    )
    print(f"Final test accuracy: {test_acc:.4f}")

    # 保存测试报告
    with open(os.path.join(args.save_dir, 'test_report.json'), 'w') as f:
        import json
        json.dump(test_report, f, indent=4)


if __name__ == '__main__':
    main()
