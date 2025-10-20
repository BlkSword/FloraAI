# FloraAI

FloraAI 是一个基于深度学习的花卉识别和分类系统，能够识别和分类100种不同的花卉类别。

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
- 添加自定义分类头，包含 Dropout 层和全连接层
- 支持集成模型（多个模型的组合）

### 数据增强技术
- 随机裁剪 (Random Crop)
- 随机水平翻转 (Random Horizontal Flip)
- 随机旋转 (Random Rotation)
- 颜色抖动 (Color Jitter)
- Mixup 数据增强

### 训练优化技术
- **优化器**: AdamW
- **正则化**: Dropout (0.5) 和 L2 正则化
- **学习率调度**: 支持 StepLR、ReduceLROnPlateau 和 CosineAnnealingLR
- **早停机制**: 基于验证集准确率保存最佳模型

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
```bash
cd code
python train.py --data_dir ../unified_flower_dataset --save_dir ../model
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
- `--scheduler`: 学习率调度器 (step, plateau, cosine)

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