# 🎯 Grad-CAM 可视化功能使用说明

## 📋 功能概述

Grad-CAM（Gradient-weighted Class Activation Mapping）梯度加权类激活映射是一种深度学习可视化技术，用于生成热力图来显示模型关注的图像区域。

**本实现的特点：**
- ✅ **只在验证集上生成** - 反映模型对未见数据的泛化能力
- ✅ **定期保存** - 按设定的 epoch 间隔自动保存可视化结果
- ✅ **多视角支持** - 同时展示多个视角的关注区域
- ✅ **配置灵活** - 支持通过配置文件和命令行参数自定义

---

## ⚙️ 配置参数

### 1. 配置文件设置 (`config.py`)

在 [`config.py`](file://c:\Users\18656\Desktop\daima\chongju\config.py#L61-L64) 中定义了 Grad-CAM 的默认配置：

```python
# Grad-CAM 可视化配置
GRAD_CAM_ENABLED = True              # 是否启用 Grad-CAM 可视化
GRAD_CAM_EPOCH_INTERVAL = 10         # Grad-CAM 可视化间隔轮次（每 N 个 epoch 生成一次）
GRAD_CAM_VIEWS = [0, 2, 4]           # 要生成 Grad-CAM 的视角索引列表
GRAD_CAM_NUM_SAMPLES = 3             # 每次生成 Grad-CAM 的样本数量
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GRAD_CAM_ENABLED` | `True` | 是否启用 Grad-CAM 功能，设为 `False` 可完全禁用 |
| `GRAD_CAM_EPOCH_INTERVAL` | `10` | 每隔多少个 epoch 生成一次可视化 |
| `GRAD_CAM_VIEWS` | `[0, 2, 4]` | 视角索引：0=0°, 1=90°, 2=180°, 3=270°, 4=顶角 |
| `GRAD_CAM_NUM_SAMPLES` | `3` | 每次从验证集选取多少个样本进行可视化 |

### 2. 命令行参数覆盖

可以在运行训练时通过命令行参数覆盖配置文件中的设置：

```bash
python main.py --grad_cam_interval 5
```

这将把 `GRAD_CAM_EPOCH_INTERVAL` 临时修改为 5。

---

## 📁 输出目录结构

运行训练后，Grad-CAM 可视化结果将保存在：

```
checkpoints/attention_fusion/
├── best_model.pth
├── training.log
├── training_curves.png
└── grad_cam_visualizations/
    ├── epoch_10/
    │   ├── sample_0_view_0_class_健康.png
    │   ├── sample_0_view_2_class_健康.png
    │   ├── sample_0_view_4_class_健康.png
    │   ├── sample_1_view_0_class_患虫.png
    │   └── ...
    ├── epoch_20/
    ├── epoch_30/
    └── ...
```

**文件命名规则：**
```
sample_{样本索引}_view_{视角索引}_class_{预测类别}.png
```

---

## 🚀 使用示例

### 基础训练（启用 Grad-CAM）

```bash
# 使用默认配置（每 10 个 epoch 生成一次）
python main.py --data_dir ./datasets --model_name attention_fusion
```

### 自定义 Grad-CAM 间隔

```bash
# 每 5 个 epoch 生成一次 Grad-CAM
python main.py --data_dir ./datasets --grad_cam_interval 5

# 每 1 个 epoch 都生成（不推荐，会产生大量文件）
python main.py --data_dir ./datasets --grad_cam_interval 1
```

### 禁用 Grad-CAM

```bash
# 方法 1: 修改 config.py 中 GRAD_CAM_ENABLED = False

# 方法 2: 设置一个很大的间隔（相当于禁用）
python main.py --data_dir ./datasets --grad_cam_interval 999
```

---

## 🎨 可视化结果说明

每个 Grad-CAM 可视化图片包含三个子图：

1. **左图**：原始图像（Original Image）
2. **中图**：Grad-CAM 热力图（Heatmap），红色表示模型高度关注的区域
3. **右图**：叠加图（Overlay），热力图半透明叠加到原图上

**标题信息：**
- `True`: 真实标签
- `Pred`: 模型预测标签
- 括号内数字：预测置信度

---

## 💡 推荐配置

### 根据训练轮次调整

| 训练轮次 | 推荐间隔 | 说明 |
|---------|---------|------|
| ≤ 30 epochs | 5 epochs | 训练轮次少，频繁观察 |
| 30-100 epochs | 10 epochs | 默认配置，平衡存储和观察 |
| > 100 epochs | 20-50 epochs | 避免生成过多文件 |

### 根据研究目的调整

- **调试阶段**：设置为 1-5，详细观察模型学习过程
- **正式训练**：设置为 10-20，定期监控即可
- **最终实验**：设置为 10，保存完整可视化记录

---

## 🔍 常见问题

### Q1: Grad-CAM 为什么只在验证集上生成？

**A:** 这是最佳实践，原因如下：
- 验证集是模型未见过的数据，能反映泛化能力
- 训练集可能存在过拟合，可视化结果不够客观
- 测试集通常只在最终评估时使用

### Q2: Grad-CAM 生成会影响训练速度吗？

**A:** 会有轻微影响，但通常可以忽略：
- 每次生成约需几秒到几十秒（取决于样本数）
- 只在指定间隔执行，不是每个 epoch 都生成
- 建议在生产训练中设置合理的间隔（如 10）

### Q3: 如何选择合适的视角？

**A:** 默认 `[0, 2, 4]` 是经过考虑的：
- `0` (0°): 水平视角代表
- `2` (180°): 对面视角，可能显示不同特征
- `4` (顶角): 顶部视角，补充水平视角信息

如果只想看一个视角，可以设置为 `[0]` 或 `[4]`。

### Q4: Grad-CAM 可视化失败怎么办？

**A:** 检查以下几点：
1. 确认模型有 `backbone` 属性
2. 确认骨干网络类型正确（convnext_tiny, resnet18 等）
3. 查看日志中的错误信息
4. Grad-CAM 失败不会影响正常训练

---

## 📊 应用场景

1. **模型调试**：观察模型是否关注正确区域
2. **结果分析**：在论文中展示模型决策依据
3. **对比实验**：比较不同模型的注意力分布
4. **训练监控**：观察模型随训练进程的改进

---

## 🛠️ 技术细节

### Grad-CAM 原理简述

1. **前向传播**：输入图像，获取预测结果
2. **反向传播**：计算目标类别对特征图的梯度
3. **权重计算**：对梯度全局平均池化得到权重
4. **加权求和**：用权重对特征图加权求和
5. **可视化**：归一化后生成热力图

### 本实现的关键代码

```python
# 从验证集获取样本
for batch in val_loader:
    grad_cam_samples = batch
    break

# 定期生成可视化
if (epoch + 1) % Config.GRAD_CAM_EPOCH_INTERVAL == 0:
    # 生成 Grad-CAM
    cam, probs, view_image = grad_cam.generate_cam_for_multiview(...)
```

---

## 📝 更新日志

- **2026-04-20**: 初始版本发布
  - 集成到 main.py 训练流程
  - 支持配置文件和命令行参数
  - 只在验证集上生成
  - 多视角可视化支持

---

## 📧 联系与反馈

如有问题或建议，请联系项目组成员。
