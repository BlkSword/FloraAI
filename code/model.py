#!/usr/bin/env python3
"""
花卉分类模型定义
基于EfficientNet-B3的图像分类模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class FlowerClassifier(nn.Module):
    """花卉分类器模型"""

    def __init__(self, num_classes=100, pretrained=True, dropout=0.5):
        super(FlowerClassifier, self).__init__()

        # 使用预训练的EfficientNet-B3作为backbone
        try:
            from efficientnet_pytorch import EfficientNet
            if pretrained:
                self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
            else:
                self.backbone = EfficientNet.from_name('efficientnet-b3')
            
            # 获取EfficientNet-B3最后一层的输入特征数
            num_features = self.backbone._fc.in_features
            
            # 替换最后的分类层
            self.backbone._fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            print("EfficientNet not available, falling back to ResNet-50")
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
        # 对于EfficientNet，我们可以使用除最后一层外的所有层提取特征
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


def create_model(num_classes=100, model_type='efficientnet_b3', pretrained=True):
    """创建模型的工厂函数"""

    if model_type == 'resnet50':
        model = FlowerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=0.5
        )
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'efficientnet_b3':
        model = FlowerClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=0.5
        )
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


def load_model(model_path, num_classes=100, device='cpu'):
    """加载训练好的模型"""
    model = create_model(num_classes=num_classes, model_type='efficientnet_b3')

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
    model = create_model(num_classes=100, model_type='efficientnet_b3')
    print(f"Model created: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 300, 300)  # EfficientNet-B3推荐输入尺寸
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")