#!/usr/bin/env python3
"""
花卉分类模型预测脚本

使用方法:
    python ./code/predict.py <测试集文件夹> <输出文件路径>

示例:
    python ./code/predict.py ./unified_flower_dataset/images/test ./results/submission.csv

输出格式:
    CSV文件包含三列: filename, category_id, confidence
    - filename: 测试图片文件名
    - category_id: 预测的类别ID (对应花卉类别编号)
    - confidence: 预测置信度 (0-1之间)
"""

import os
import argparse
import pandas as pd
import torch
from pathlib import Path
import json

from model import load_model
from utils import UnlabeledDataset, set_seed
import torchvision.transforms as transforms
import torch.nn.functional as F


def load_class_mapping(config_path):
    """加载类别映射"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    class_to_idx = config.get('class_to_idx', {})
    # 转换为idx_to_class映射
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


def predict_with_model(model, test_loader, device, idx_to_class):
    """使用模型进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算概率
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)
            
            # 转换为类别ID
            if len(idx_to_class) > 0:
                predicted_classes = [idx_to_class[idx.item()] for idx in predicted_indices]
            else:
                # 如果idx_to_class为空，则直接使用索引作为类别ID
                predicted_classes = [idx.item() for idx in predicted_indices]
            
            # 收集预测结果
            for filename, category_id, confidence in zip(filenames, predicted_classes, confidences):
                predictions.append({
                    'filename': filename,
                    'category_id': int(category_id),
                    'confidence': confidence.item()
                })
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='花卉分类模型预测')

    # 位置参数：测试集文件夹和输出文件
    parser.add_argument('test_img_dir', type=str,
                        help='测试图片目录')
    parser.add_argument('output_path', type=str,
                        help='预测结果输出路径 (CSV文件)')

    # 可选参数
    parser.add_argument('--model_path', type=str, default='../model/best_model.pth',
                        help='训练好的模型路径')
    parser.add_argument('--config_path', type=str, default='../model/config.json',
                        help='模型配置文件路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--img_size', type=int, default=300,
                        help='图像尺寸')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（用于可重复性）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    print(f'测试集目录: {args.test_img_dir}')
    print(f'输出文件: {args.output_path}')
    print(f'模型路径: {args.model_path}')
    print(f'设备: {args.device}')
    print()

    # 检查测试集目录是否存在
    if not os.path.exists(args.test_img_dir):
        print(f"错误: 测试集目录不存在: {args.test_img_dir}")
        return

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    # 检查配置文件是否存在
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在: {args.config_path}")
        return

    # 加载类别映射
    print("正在加载类别映射...")
    idx_to_class = load_class_mapping(args.config_path)
    print(f"加载了 {len(idx_to_class)} 个类别映射")
    print()

    # 加载模型
    print("正在加载模型...")
    device = torch.device(args.device)
    
    # 从配置文件中读取类别数量
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    num_classes = config.get('num_classes', len(idx_to_class))
    
    model = load_model(args.model_path, num_classes=num_classes, device=device)
    
    if model is None:
        print("错误: 模型加载失败")
        return
        
    model.to(device)
    model.eval()
    print("模型加载成功")
    print()

    # 创建数据集和数据加载器
    print("正在准备测试数据...")
    # 简单的变换（与训练时的验证变换一致）
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = UnlabeledDataset(args.test_img_dir, transform=transform)
    
    if len(test_dataset) == 0:
        print(f"错误: 在目录 {args.test_img_dir} 中未找到图片文件")
        return
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"找到 {len(test_dataset)} 张图片")
    print()

    # 进行预测
    print("正在进行预测...")
    predictions = predict_with_model(model, test_loader, device, idx_to_class)
    print(f"完成预测 {len(predictions)} 张图片")
    print()

    # 创建 DataFrame
    results_df = pd.DataFrame(predictions)

    # 按照文件名排序
    results_df = results_df.sort_values('filename').reset_index(drop=True)

    # 创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    # 保存结果
    results_df.to_csv(args.output_path, index=False)
    print(f"预测结果已保存到: {args.output_path}")
    print()   

    print("预测完成!")


if __name__ == '__main__':
    main()