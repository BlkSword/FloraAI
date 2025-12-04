#!/usr/bin/env python3
"""
花卉分类模型预测脚本

使用方法:
    python ./code/predict.py <测试集文件夹> <输出文件路径>

示例:
    python ./code/predict.py ./unified_flower_dataset/images/test ./results/submission.csv
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
# from tqdm import tqdm  # 暂时禁用进度条输出
import json
import warnings
from torch.cuda.amp import autocast

# 屏蔽 pydantic 相关警告（如 UnsupportedFieldAttributeWarning）
try:
    from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    warnings.filterwarnings("ignore", module=r"pydantic.*")

# 导入自定义模块
from model import load_model
from utils import UnlabeledDataset, tta_inference, get_transforms


def get_image_files(img_dir, img_extensions=None):
    """获取目录中的所有图片文件"""
    if img_extensions is None:
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    else:
        img_extensions = set(ext.lower() for ext in img_extensions)

    image_files = []
    
    with os.scandir(img_dir) as entries:
        for entry in entries:
            if entry.is_file():
                # 检查扩展名（忽略大小写）
                _, ext = os.path.splitext(entry.name)
                if ext.lower() in img_extensions:
                    image_files.append(entry.name)

    image_files.sort()

    return image_files


def load_config(config_path):
    """加载模型配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def predict(model, dataloader, device, idx_to_class=None, use_tta=True, use_fp16=True):
    """使用模型进行预测"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, file_names = batch
            images = images.to(device)
            
            if use_tta:
                batch_outputs = []
                for i in range(images.size(0)):
                    out = tta_inference(model, images[i], device)
                    batch_outputs.append(out)
                outputs = torch.cat(batch_outputs, dim=0)
            else:
                if use_fp16 and device.type == 'cuda':
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
            
            probs = F.softmax(outputs, dim=1)
            confidences, class_ids = torch.max(probs, dim=1)
            
            for i, file_name in enumerate(file_names):
                pred_idx = class_ids[i].item()
                mapped_id = idx_to_class[pred_idx] if idx_to_class is not None else pred_idx
                predictions.append({
                    'filename': file_name,
                    'category_id': mapped_id,
                    'confidence': confidences[i].item()
                })
    
    return predictions


def main():
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='花卉分类模型预测')
    
    parser.add_argument('test_dir', type=str, help='测试图片目录')
    parser.add_argument('output', type=str, help='预测结果输出路径 (CSV文件)')
    
    parser.add_argument('--model_path', type=str, default=str(base_dir / 'model' / 'best_model.pth'), help='模型路径')
    parser.add_argument('--config_path', type=str, default=str(base_dir / 'model' / 'config.json'), help='配置文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--use_tta', action='store_true', default=True, help='是否使用测试时增强')
    parser.add_argument('--use_fp16', action='store_true', default=False, help='是否使用FP16混合精度')
    
    args = parser.parse_args()

    args.model_path = str(Path(args.model_path).resolve())
    args.config_path = str(Path(args.config_path).resolve())

    if not os.path.exists(args.config_path):
        raise FileNotFoundError(
            f"配置文件未找到: {args.config_path}。请确认路径是否正确，或通过 --config_path 指定正确的配置文件路径。"
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载配置
    config = load_config(args.config_path)
    model_type = config.get('model_type', 'convnext_base')
    num_classes = config.get('num_classes', 152)
    img_size = config.get('img_size', 448)
    class_to_idx = config.get('class_to_idx', {})
    idx_to_class = {int(v): int(k) for k, v in class_to_idx.items()} if class_to_idx else None
    
    # 创建模型
    model = load_model(args.model_path, model_type=model_type, num_classes=num_classes, device=device)
    model.to(device)
    
    if not os.path.exists(args.test_dir):
        return
        
    # 获取测试图像文件
    image_files = get_image_files(args.test_dir)
    
    if not image_files:
        return
    
    test_transform = get_transforms(phase='test', img_size=img_size)
    dataset = UnlabeledDataset(args.test_dir, transform=test_transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 进行预测
    predictions = predict(
        model,
        dataloader,
        device,
        idx_to_class=idx_to_class,
        use_tta=args.use_tta,
        use_fp16=args.use_fp16
    )
    
    # 将预测结果转换为DataFrame
    df = pd.DataFrame(predictions)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果
    df.to_csv(args.output, index=False)
    



if __name__ == '__main__':
    main()
