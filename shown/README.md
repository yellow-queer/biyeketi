# 🍊 柑橘实蝇检测系统 - 快速指南

## 🎯 一句话介绍
基于多视角图像融合技术的柑橘实蝇无损检测系统，支持 5 种模型对比和可视化分析。

---

## 🚀 30 秒快速开始

### 1. 安装依赖
```bash
# 安装 PyTorch (CUDA 12.8+)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 一键运行所有实验
```bash
python run_all_experiments.py
```

### 3. 查看结果
结果保存在 `checkpoints/` 目录，包含：
- 训练曲线
- 混淆矩阵
- 评估报告
- 模型权重

---

## 📦 支持的模型

| 模型 | 输入 | 特点 | 训练命令 |
|------|------|------|----------|
| **注意力融合** ⭐ | 5 张图 | 自适应权重融合 | `python main.py` |
| 单视角 | 1 张图 | 基线模型 | `python train_comparison_models.py --model_type single_view` |
| 简单拼接 | 5 张图 | 特征直接拼接 | `python train_comparison_models.py --model_type simple_concat` |
| CNN-LSTM | 5 张图 | 序列建模 | `python train_comparison_models.py --model_type cnn_lstm` |
| CNN-LSTM-Attention | 5 张图 | LSTM+ 注意力 | `python train_comparison_models.py --model_type cnn_lstm_attention` |

---

## 🎨 可视化分析

### 训练过程热力图（推荐）
```bash
python toshow/visualize_predictions.py
```
- 交互式输入样本编号
- 生成 Grad-CAM 热力图
- 展示模型注意力分布

### 识别结果可视化
```bash
python visualize_predictions.py --model_type attention_fusion
```

---

## ⚙️ 常用配置

修改 `config.py`：
```python
BACKBONE_MODEL = 'convnext_tiny'  # 骨干网络
BATCH_SIZE = 8                     # 批次大小
LEARNING_RATE = 1e-4               # 学习率
NUM_EPOCHS = 30                    # 训练轮次
```

---

## 📂 目录结构

```
chongju/
├── main.py                        # 主训练程序
├── train_comparison_models.py     # 对比模型训练
├── run_all_experiments.py         # 一键运行
├── compare_backbones.py           # 骨干网络对比
├── visualize_predictions.py       # 结果可视化
├── models/                        # 模型定义
├── checkpoints/                   # 结果保存
└── datasets/                      # 数据集
```

---

## 📊 输出文件说明

每个模型训练后自动生成：
- `training.log` - 训练日志
- `best_model.pth` - 最佳模型权重
- `training_curves.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵
- `evaluation_report.txt` - 评估报告

---

## 🔧 常见问题

**Q: 显存不足？**  
A: 减小 `BATCH_SIZE` 或使用 `convnext_tiny`

**Q: 如何恢复训练？**  
A: 使用 `--resume ./checkpoints/xxx/best_model.pth`

**Q: 如何更换骨干网络？**  
A: 使用 `--backbone resnet18` 或其他

---

## 📞 技术支持

1. 检查 GPU: `torch.cuda.is_available()`
2. 查看训练日志
3. 验证数据集结构

**详细文档**: [EXPERIMENT_GUIDE.md](./EXPERIMENT_GUIDE.md)
