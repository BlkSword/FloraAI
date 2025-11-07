# FloraAI

FloraAI 是一个基于深度学习的花卉识别和分类系统，能够识别和分类100种不同的花卉类别。本项目采用了先进的深度学习优化技术，显著提升了模型的训练准确率和泛化能力。

## 🚀 最新优化特性

### 模型结构优化
- **改进分类头设计**: 三层结构（1024→512→num_classes）
- **批归一化层**: 每层线性层后添加BatchNorm1d，提高训练稳定性
- **SiLU激活函数**: 替代ReLU，提供更平滑的梯度流
- **注意力机制**: 可选的门控注意力模块，增强特征提取能力
- **优化的dropout策略**: 最后一层dropout比例减少，平衡过拟合和表达能力

### 高级数据增强策略
- **基础增强**: 随机裁剪、水平翻转、旋转、颜色抖动
- **高级增强**: 垂直翻转、仿射变换、透视变换、弹性变换
- **图像处理增强**: 高斯模糊、锐化、灰度化、噪声添加
- **Cutout增强**: 随机遮挡图像区域，提高模型鲁棒性
- **扩展颜色抖动**: 调整亮度/对比度/饱和度/色调范围，新增gamma调整

### 优化器和学习率调度
- **多种优化器**: Adam、AdamW、SGD、RMSprop
- **高级调度策略**: cosine_warmup、onecycle等
- **学习率预热**: 渐进式预热，提高训练稳定性
- **梯度裁剪**: 防止梯度爆炸，确保训练收敛

### 高级损失函数
- **Focal Loss**: 处理类别不平衡问题，支持alpha和gamma参数调节
- **标签平滑交叉熵**: 提高模型泛化能力
- **灵活的损失函数选择**: 支持标准交叉熵、标签平滑、焦点损失三种模式

## 技术细节

### 核心技术栈
- **深度学习框架**: PyTorch
- **模型架构**: 
  - 主要使用 EfficientNet-B3 作为特征提取器
  - 备选方案包括 ResNet-50 和 ResNet-101
- **图像处理**: Pillow, torchvision
- **数据处理**: pandas, NumPy
- **可视化**: matplotlib, seaborn
- **评估指标**: scikit-learn

### 模型架构
- 基于 EfficientNet-B3 或 ResNet 架构
- 改进的三层分类头设计，包含批归一化和SiLU激活函数
- 可选注意力机制，增强特征提取能力
- 支持集成模型（多个模型的组合）

### 数据增强技术
- **基础增强**: 随机裁剪、水平翻转、旋转、颜色抖动
- **高级增强**: 垂直翻转、仿射变换、透视变换、弹性变换等
- **Mixup数据增强**: 支持alpha参数调节
- **智能增强控制**: 可通过参数开关高级增强功能

## 训练模型

### 环境准备
```bash
pip install -r code/requirements.txt
```

### 数据集结构
```
unified_flower_dataset/
├── images/
│   ├── train/
│   └── test/
├── train_labels.csv
└── test_labels.csv
```

### 训练命令

#### 基础训练
```bash
cd code
python train.py --data_dir ../unified_flower_dataset --save_dir ../model
```

#### 推荐配置（针对EfficientNet-B3优化）
```bash
cd code
python train.py \
    --data_dir ../unified_flower_dataset \
    --save_dir ../model \
    --model efficientnet_b3 \
    --loss_type focal \
    --scheduler cosine_warmup \
    --warmup_epochs 5 \
    --use_attention \
    --use_advanced_aug \
    --grad_clip 1.0 \
    --batch_size 32 \
    --epochs 150
```

#### 高级配置（追求最佳性能）
```bash
cd code
python train.py \
    --data_dir ../unified_flower_dataset \
    --save_dir ../model \
    --model efficientnet_b3 \
    --optimizer adamw \
    --loss_type focal \
    --focal_alpha 0.8 \
    --focal_gamma 2.0 \
    --scheduler onecycle \
    --warmup_epochs 10 \
    --use_attention \
    --use_advanced_aug \
    --grad_clip 0.5 \
    --dropout 0.3 \
    --batch_size 64 \
    --epochs 200
```

#### 使用本地预训练模型
```bash
cd code
python train.py \
    --data_dir ../unified_flower_dataset \
    --save_dir ../model \
    --model efficientnet_b3 \
    --model_path /path/to/your/pretrained_model.pth \
    --batch_size 32 \
    --epochs 100
```

### 训练参数
- `--data_dir`: 数据集目录路径
- `--model_type`: 模型类型 (resnet50, resnet101, efficientnet_b3, efficientnet_b4)
- `--num_classes`: 类别数量 (默认: 100)
- `--batch_size`: 批次大小 (默认: 16)
- `--epochs`: 训练轮数 (默认: 100)
- `--lr`: 学习率 (默认: 0.001)
- `--img_size`: 图像尺寸 (默认: 300)
- `--mixup_alpha`: Mixup 参数 (默认: 0.2)
- `--scheduler`: 学习率调度器 (step, plateau, cosine, cosine_warmup, onecycle)
- `--optimizer`: 优化器类型 (adam, adamw, sgd, rmsprop)
- `--loss_type`: 损失函数类型 (ce, focal, label_smooth)
- `--focal_alpha`: Focal Loss的alpha参数 (默认: 1.0)
- `--focal_gamma`: Focal Loss的gamma参数 (默认: 2.0)
- `--label_smoothing`: 标签平滑参数 (默认: 0.0)
- `--warmup_epochs`: 学习率预热轮数 (默认: 0)
- `--grad_clip`: 梯度裁剪阈值 (默认: 1.0)
- `--use_attention`: 启用注意力机制
- `--use_advanced_aug`: 启用高级数据增强
- `--dropout`: 分类头dropout比例 (默认: 0.5)
- `--step_size`: StepLR步长 (默认: 30)
- `--gamma`: 学习率衰减因子 (默认: 0.1)
- `--patience`: ReduceLROnPlateau耐心值 (默认: 5)
- `--min_lr`: 最小学习率 (默认: 1e-6)
- `--model_path`: 预训练模型的本地路径 (默认: None，从在线下载)

## 📊 性能预期

经过全面优化后，FloraAI模型预期能够达到以下性能提升：

### 训练效果提升
- **训练准确率**: 提升5-10个百分点
- **收敛速度**: 加快20-30%
- **泛化能力**: 验证集准确率提升3-5个百分点
- **训练稳定性**: 显著减少训练过程中的波动

### 优化技术贡献
- **模型结构优化**: 贡献约2-3%准确率提升
- **数据增强策略**: 贡献约3-4%准确率提升
- **优化器调度优化**: 贡献约2-3%准确率提升
- **高级损失函数**: 贡献约1-2%准确率提升

## 使用本地预训练模型

FloraAI 现在支持从本地路径加载预训练模型，而不是从在线下载。这提供了以下优势：

### 功能特性
- **离线训练**: 在没有网络连接的环境下进行训练
- **自定义预训练**: 使用自己训练的模型作为基础
- **版本控制**: 确保使用特定版本的预训练模型
- **性能优化**: 避免网络下载延迟，加快训练启动速度

### 使用方法

#### 训练时使用本地预训练模型
```bash
cd code
python train.py \
    --data_dir ../unified_flower_dataset \
    --save_dir ../model \
    --model efficientnet_b3 \
    --model_path /path/to/your/pretrained_model.pth
```

#### 预测时使用预训练模型初始化
```bash
cd code
python predict.py \
    ../unified_flower_dataset/images/test \
    ../results/submission.csv \
    --pretrained_model_path /path/to/your/pretrained_model.pth
```

### 注意事项
- 当指定 `--model_path` 参数时，系统会优先从本地路径加载模型
- 如果本地路径不存在或文件损坏，系统会回退到在线下载
- 预训练模型需要与目标模型架构兼容
- 支持所有模型类型：EfficientNet-B3/B4、ResNet-50/101

## 验证模型

训练过程中会自动在验证集上评估模型性能，并保存最佳模型。

### 单独评估模型
```bash
cd code
python predict.py ../unified_flower_dataset/images/test ../results/submission.csv
```

### 评估参数
- `test_img_dir`: 测试图片目录
- `output_path`: 预测结果输出路径 (CSV文件)
- `--model_path`: 训练好的模型路径 (默认: ../model/best_model.pth)
- `--pretrained_model_path`: 预训练模型的本地路径，用于初始化模型权重 (默认: None)
- `--config_path`: 模型配置文件路径 (默认: ../model/config.json)
- `--batch_size`: 批次大小 (默认: 16)
- `--img_size`: 图像尺寸 (默认: 300)

### 输出格式
预测结果将保存为 CSV 文件，包含以下列：
- `filename`: 测试图片文件名
- `category_id`: 预测的类别ID
- `confidence`: 预测置信度 (0-1之间)

## 项目结构
```
FloraAI/
├── code/                 # 源代码目录
│   ├── model.py          # 模型定义
│   ├── train.py          # 训练脚本
│   ├── predict.py        # 预测脚本
│   ├── utils.py          # 工具函数
│   └── requirements.txt  # 依赖包
├── model/                # 模型保存目录
│   ├── best_model.pth    # 最佳模型
│   ├── config.json       # 模型配置
│   └── ...               # 其他模型文件
├── unified_flower_dataset/  # 数据集目录
└── results/              # 预测结果目录
```