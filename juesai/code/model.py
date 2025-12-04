#!/usr/bin/env python3
"""
花卉分类模型定义
基于ConvNeXt-Base的图像分类模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import convnext_base
from torchvision.models.convnext import ConvNeXt_Base_Weights
import torch.nn.functional as F
import timm


class FlowerClassifier(nn.Module):
    """花卉分类器模型"""

    def __init__(self, num_classes=100, pretrained=True, dropout=0.5):
        super(FlowerClassifier, self).__init__()

        self.backbone = models.resnet50(pretrained=pretrained)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)

    def extract_features(self, x):
        """提取特征向量（不包含最后的分类层）"""
        # 获取除了最后一层之外的所有层
        features = nn.Sequential(*list(self.backbone.children())[:-1])
        x = features(x)
        x = torch.flatten(x, 1)
        return x


class EnsembleModel(nn.Module):
    """集成模型"""

    def __init__(self, models_list, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        self.weights = weights if weights is not None else [1.0] * len(models_list)

    def forward(self, x):
        """集成预测"""
        outputs = []
        for i, model in enumerate(self.models):
            output = model(x) * self.weights[i]
            outputs.append(output)

        # 平均集成
        ensemble_output = torch.stack(outputs).mean(dim=0)
        return ensemble_output


class ConvNeXtClassifier(nn.Module):
    """基于ConvNeXt-Base的花卉分类器"""
    
    def __init__(self, num_classes=100, pretrained=True):
        super(ConvNeXtClassifier, self).__init__()
        
        # 使用预训练的ConvNeXt-Base作为backbone
        if pretrained:
            # 使用ImageNet-22k预训练并在ImageNet-1k微调的权重
            self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            self.backbone = convnext_base(weights=None)
            
        # 替换最后的分类层
        self.backbone.classifier[2] = nn.Linear(1024, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """前向传播"""
        return self.backbone(x)
    
    def extract_features(self, x):
        """提取特征向量（不包含最后的分类层）"""
        # 获取除了最后一层之外的所有层
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

def create_model(num_classes=100, model_type='convnext_base', pretrained=True):
    """创建模型的工厂函数"""

    if model_type == 'convnext_base':
        model = ConvNeXtClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif model_type == 'convnext_base_timm':
        model = timm.create_model(
            'convnext_base.fb_in22k_ft_in1k', 
            pretrained=pretrained,
            num_classes=num_classes
        )
    elif model_type == 'resnet50':
        model = FlowerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=0.5
        )
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'efficientnet_b4':
        try:
            from efficientnet_pytorch import EfficientNet
            if pretrained:
                model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
            else:
                model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
        except ImportError:
            print("EfficientNet not available, falling back to ResNet-50")
            model = FlowerClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def load_model(model_path, model_type='convnext_base', num_classes=100, device='cpu'):
    """加载训练好的模型
    
    Args:
        model_path: 已训练模型权重路径
        model_type: 模型类型，与训练时保持一致
        num_classes: 分类数
        device: 设备
    """
    # 根据提供的模型类型创建对应结构
    model = create_model(num_classes=num_classes, model_type=model_type, pretrained=False)

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        print(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == "__main__":
    # 测试模型创建
    model = create_model(num_classes=100)
    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")