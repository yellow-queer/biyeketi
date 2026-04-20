# 📖 实验详细指南

> 本文档提供完整的实验操作说明，适合首次使用者。

---

## 📋 目录
1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [快速开始](#快速开始)
4. [详细实验步骤](#详细实验步骤)
5. [结果分析](#结果分析)
6. [常见问题](#常见问题)

---

## 项目概述

### 五种对比模型
1. **注意力融合模型** ⭐ - 自适应学习视角权重
2. **单视角模型** - 基线对比
3. **简单拼接模型** - 特征直接拼接
4. **CNN-LSTM** - 序列建模
5. **CNN-LSTM-Attention** - LSTM+ 注意力

### 支持的骨干网络
- ConvNeXt 系列 (tiny/small/base/large)
- ResNet-18

---

## 环境配置

### 硬件要求
- GPU: NVIDIA RTX 5060 或更高
- CUDA: 12.8+
- 显存：≥8GB

### 软件安装
```bash
# PyTorch (CUDA 12.8)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 其他依赖
pip install -r requirements.txt
```

### 验证安装
```python
import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.get_device_name(0))
```

---

## 快速开始

### 方法一：一键运行（推荐）
```bash
python run_all_experiments.py
```
自动训练所有模型并保存结果。

### 方法二：单独训练

#### 1. 训练注意力融合模型
```bash
python main.py --data_dir ./datasets --backbone convnext_tiny --epochs 30
```

#### 2. 训练其他模型
```bash
python train_comparison_models.py --data_dir ./datasets --model_type single_view --epochs 30
python train_comparison_models.py --data_dir ./datasets --model_type simple_concat --epochs 30
python train_comparison_models.py --data_dir ./datasets --model_type cnn_lstm --epochs 30
python train_comparison_models.py --data_dir ./datasets --model_type cnn_lstm_attention --epochs 30
```

#### 3. 骨干网络对比
```bash
python compare_backbones.py --data_dir ./datasets --epochs 30
```

---

## 详细实验步骤

### 步骤 1：准备数据集
确保数据集结构：
```
datasets/
├── pic/
│   ├── pic_healthy/    # 健康顶角照片
│   └── pic_chongju/    # 患虫顶角照片
└── video/
    ├── v_healthy/      # 健康环绕视频
    └── v_chongju/      # 患虫环绕视频
```

### 步骤 2：数据预处理
```bash
python data_preprocess/image_preprocess.py
```
- 视频抽帧（0°/90°/180°/270°）
- 图像增强
- 尺寸统一（224×224）

### 步骤 3：训练模型
```bash
python main.py --epochs 30 --batch_size 8 --lr 1e-4
```

### 步骤 4：可视化分析

#### 热力图可视化
```bash
python toshow/visualize_predictions.py
```
输入样本编号（如 A1, h1），生成 Grad-CAM 热力图。

#### 识别结果可视化
```bash
python visualize_predictions.py --model_type attention_fusion
```

---

## 结果分析

### 输出文件
每个模型生成：
- `training.log` - 训练日志
- `best_model.pth` - 最佳模型
- `training_curves.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵
- `evaluation_report.txt` - 评估报告

### 评估指标
- **准确率 (Accuracy)** - 总体正确率
- **精确率 (Precision)** - 预测为正的准确性
- **召回率 (Recall)** - 实际正例的检出率
- **F1 值 (F1-Score)** - 综合指标

### 结果对比
查看 `checkpoints/` 目录下各模型的 `evaluation_report.txt` 进行对比。

---

## 常见问题

### Q1: 显存不足
**解决方案**：
- 减小 `--batch_size 4`
- 使用 `convnext_tiny`
- 冻结骨干网络

### Q2: 训练中断
**恢复训练**：
```bash
python main.py --resume ./checkpoints/attention_fusion/best_model.pth
```

### Q3: 更换数据集
```bash
python main.py --data_dir /path/to/your/dataset
```

### Q4: 调整超参数
```bash
python main.py --epochs 50 --lr 5e-5 --batch_size 4
```

### Q5: 使用学习率调度器
```bash
python main.py --use_lr_scheduler --lr_scheduler_type cosine --min_lr 1e-6
```

---

## 技术支持

1. **检查 GPU 状态**：`torch.cuda.is_available()`
2. **查看训练日志**：`checkpoints/xxx/training.log`
3. **验证数据集**：确保目录结构正确
4. **查看错误信息**：日志文件中的 traceback

**快速指南**: [README.md](./README.md)
