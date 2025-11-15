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
from utils import UnlabeledDataset, tta_inference_probs, get_transforms


def get_image_files(img_dir, img_extensions=None):
    """获取目录中的所有图片文件"""
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']

    img_dir_path = Path(img_dir)
    image_files = []

    # 获取所有图片文件
    for ext in img_extensions:
        image_files.extend(img_dir_path.glob(f'*{ext}'))
        image_files.extend(img_dir_path.glob(f'*{ext.upper()}'))

    # 只保留文件名，并排序确保一致性
    image_files = sorted([f.name for f in image_files])

    return image_files


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def predict_ensemble(models, weights, dataloader, device, idx_to_class=None, use_tta=True, use_fp16=True):
    for m in models:
        m.eval()
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    w = w / (w.sum() if w.sum() > 0 else torch.tensor(1.0, device=device))
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            images, file_names = batch
            images = images.to(device)
            probs_list = []
            for model in models:
                if use_tta:
                    batch_outputs = []
                    for i in range(images.size(0)):
                        out = tta_inference_probs(model, images[i], device)
                        batch_outputs.append(out)
                    outputs = torch.cat(batch_outputs, dim=0)
                else:
                    if use_fp16 and device.type == 'cuda':
                        with autocast():
                            outputs = model(images)
                    else:
                        outputs = model(images)
                probs_list.append(outputs if use_tta else F.softmax(outputs, dim=1))
            stacked = torch.stack(probs_list, dim=0)
            combined = (w.view(-1, 1, 1) * stacked).sum(dim=0)
            confidences, class_ids = torch.max(combined, dim=1)
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
    parser.add_argument('--model_paths', type=str, default=','.join([
        str(base_dir / 'model' / 'best_model_co.pth'),
        str(base_dir / 'model' / 'best_model_v2.pth')
    ]), help='多个模型路径，逗号分隔')
    parser.add_argument('--config_paths', type=str, default=','.join([
        str(base_dir / 'model' / 'config_co.json'),
        str(base_dir / 'model' / 'config_v2.json')
    ]), help='多个配置文件路径，逗号分隔')
    parser.add_argument('--weights', type=str, default='0.7,0.3', help='集成权重，逗号分隔')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--use_tta', action='store_true', default=True, help='是否使用测试时增强')
    parser.add_argument('--use_fp16', action='store_true', default=False, help='是否使用FP16混合精度')
    args = parser.parse_args()

    model_paths = [str(Path(p).resolve()) for p in args.model_paths.split(',') if p.strip()]
    config_paths = [str(Path(p).resolve()) for p in args.config_paths.split(',') if p.strip()]
    if len(model_paths) != len(config_paths) or len(model_paths) == 0:
        return
    for cp in config_paths:
        if not os.path.exists(cp):
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = [load_config(cp) for cp in config_paths]
    model_types = [c.get('model_type', 'convnext_base') for c in configs]
    num_classes_list = [c.get('num_classes', 102) for c in configs]
    img_sizes = [c.get('img_size', 384) for c in configs]
    class_to_idx_candidates = [c.get('class_to_idx', {}) for c in configs]
    idx_to_class = None
    for m in class_to_idx_candidates:
        if m:
            idx_to_class = {int(v): int(k) for k, v in m.items()}
            break
    num_classes = num_classes_list[0]
    img_size = max(img_sizes)

    models = []
    for mp, mt, nc in zip(model_paths, model_types, num_classes_list):
        model = load_model(mp, model_type=mt, num_classes=nc, device=device)
        if model is not None:
            model.to(device)
            models.append(model)
    if len(models) == 0:
        return

    if not os.path.exists(args.test_dir):
        return
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

    if args.weights:
        try:
            weights = [float(x) for x in args.weights.split(',')]
        except Exception:
            weights = [1.0] * len(models)
    else:
        perf = []
        for c in configs:
            val = c.get('best_accuracy', 0.0)
            try:
                perf.append(float(val))
            except Exception:
                perf.append(0.0)
        if sum(perf) <= 0:
            weights = [1.0] * len(models)
        else:
            weights = perf

    predictions = predict_ensemble(
        models,
        weights,
        dataloader,
        device,
        idx_to_class=idx_to_class,
        use_tta=args.use_tta,
        use_fp16=args.use_fp16
    )

    df = pd.DataFrame(predictions)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
