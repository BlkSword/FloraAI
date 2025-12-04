# FloraAI

FloraAI 是一个高精度的花卉识别模型的训练和预测框架，支持 100 个类别（每类150张图片），并在验证集取得了 98.18% 的最佳准确率（ConvNeXt-Base 主干，TTA + EMA，集成权重 0.7/0.3）。

*决赛模型详见juesai。赛题方面增大了数据集，并且设置了少样本类别。最终没有使用很复杂的训练策略，很简单的方法。最后是97.8%。

## 成果概览

- 验证集最佳准确率：98.18%（ConvNeXt-Base，见 `model/config_co.json`）
- 第二模型（EfficientNetV2-S）最佳准确率：96.99%（见 `model/config_v2.json`）
- 预测默认采用集成推理：ConvNeXt-Base + EfficientNetV2-S，并启用测试时增强（TTA）

## 技术细节

- 深度学习框架：PyTorch（`code/train.py:175`）
- 主力模型：ConvNeXt-Base（timm `convnext_base.fb_in22k_ft_in1k`，`code/train.py:179` 与 `code/model.py:109-115`）
- 辅助模型：EfficientNetV2-S（timm `tf_efficientnetv2_s_in21k_ft_in1k`，`code/train.py:243-257`）
- 数据增强：RandAugment + RandomErasing（`code/utils.py:120-129`），Mixup / CutMix（`code/utils.py:165-175`）
- 训练优化：
  - FP16 混合精度（AMP）（`code/train.py:70-83`）
  - EMA 指数移动平均（`code/utils.py:234-260` + `code/train.py:350-356`）
  - 分层学习率（分类头 2x）（`code/train.py:36-59`）
  - 线性暖起 + 余弦退火（`code/train.py:363-379`）
- 推理增强：TTA（水平/垂直翻转 + Softmax 平均，`code/utils.py:408-430`）
- 集成推理：多模型加权融合（默认 0.7/0.3，`code/predict.py:104-113`，融合逻辑 `code/predict.py:60-88`）

## 环境准备

```bash
pip install -r code/requirements.txt
```

依赖包含但不限于：`torch`、`torchvision`、`timm`、`pandas`、`numpy`、`Pillow`、`scikit-learn`、`matplotlib`、`seaborn`。

## 数据集结构

```
unified_flower_dataset/
├── images/
│   ├── train/
│   └── test/
├── train_labels.csv
└── test_labels.csv
```

## 训练

### ConvNeXt-Base（推荐，98.18%）

```bash
cd code
python train.py \
  --data_dir ../unified_flower_dataset \
  --save_dir ../model \
  --model_type convnext_base_timm
```

要点（已在代码中自动配置）：`img_size=512`、`epochs=90`、`lr=2e-4`、`min_lr=2e-6`、`weight_decay=2e-2`、`label_smoothing=0.05`、`mixup_alpha=0.3`、`cutmix_alpha=0.2`、启用 AMP 与 EMA（`code/train.py:227-243`）。

### EfficientNetV2-S（可选，96.99%）

```bash
cd code
python train.py \
  --data_dir ../unified_flower_dataset \
  --save_dir ../model \
  --model_type tf_efficientnetv2_s_in21k_ft_in1k
```

要点：`img_size=416`、其余超参同 ConvNeXt-Base（`code/train.py:243-257`）。

### 训练提示

- 冻结/解冻骨干两阶段微调（`--freeze_backbone_epochs`，`code/train.py:305-347`）。
- 动态衰减 Mixup/CutMix 强度（`code/train.py:422-435`）。
- 自动保存最佳模型与配置（`model/best_model_*.pth`、`model/config_*.json`，`code/train.py:468-509`）。

## 评估

训练结束自动在验证集评估，并输出曲线与报告：

- 训练/验证曲线：`model/training_history.png`（`code/utils.py:308-335`）
- 分类报告：`model/test_report.json`（`code/train.py:532-536`）

## 推理与提交

### 集成推理（默认）

```bash
cd code
python predict.py \
  ../unified_flower_dataset/images/test \
  ../results/submission.csv
```

默认使用 `model/best_model_co.pth` + `model/best_model_v2.pth`，配置与类别映射来自 `model/config_co.json` 与 `model/config_v2.json`（`code/predict.py:104-121`）。

### 单模型推理（示例）

```bash
cd code
python predict.py \
  ../unified_flower_dataset/images/test \
  ../results/submission.csv \
  --model_paths ../model/best_model_co.pth \
  --config_paths ../model/config_co.json \
  --weights 1.0
```

### 输出格式

CSV 列包含：`filename`、`category_id`、`confidence`（`code/predict.py:87-96`）。

## 关键参数

- `--model_type`：`convnext_base_timm`、`tf_efficientnetv2_s_in21k_ft_in1k`、`convnext_base`、`resnet50`、`resnet101`、`efficientnet_b4`（`code/train.py:179-181`）
- `--img_size`、`--batch_size`、`--epochs`、`--lr`、`--min_lr`、`--weight_decay`（见自动配置段 `code/train.py:227-257`）
- `--use_amp`（FP16）、`--use_ema`、`--ema_decay`（`code/train.py:204-209`）
- `--mixup_alpha`、`--cutmix_alpha`、`--label_smoothing`、`--use_advanced_aug`（`code/train.py:210-218`）
- `--warmup_epochs`（线性暖起）、`--freeze_backbone_epochs`（阶段冻结，`code/train.py:218-221`）

## 项目结构

```
FloraAI/
├── code/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   └── requirements.txt
├── model/
│   ├── best_model_co.pth
│   ├── best_model_v2.pth
│   ├── config_co.json
│   ├── config_v2.json
│   └── …
├── unified_flower_dataset/
└── results/
```

## 复现建议

- 保持随机种子一致（`--seed`，`code/utils.py:204-213`）。
- 使用 GPU 并启用 AMP 与 TTA 获得更高推理吞吐与稳定性。
- 若类别数与数据集不一致，训练脚本会强制使用数据集的类别数（`code/train.py:292-295`）。
