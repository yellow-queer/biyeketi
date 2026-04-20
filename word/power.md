# PowerShell 运行指令集合

## 📌 使用说明

1. 打开 PowerShell（按 `Win + R`，输入 `powershell`，回车）
2. 进入项目目录：
   ```powershell
   cd C:\Users\18656\Desktop\daima\chongju
   ```
3. 运行对应的 Python 文件

---

## 🚀 主要训练程序

### 1. 主训练程序（多视角注意力融合模型）
```powershell
python main.py --epochs 30
```

**可选参数：**
```powershell
# 指定数据集路径
python main.py --data_dir ./datasets --epochs 30

# 调整批次大小和学习率
python main.py --batch_size 8 --lr 0.0001 --epochs 30

# 使用不同骨干网络
python main.py --backbone resnet18 --epochs 30

# 恢复训练
python main.py --resume ./checkpoints/attention_fusion/best_model.pth
```

---

### 2. 骨干网络对比实验（ConvNeXt-Tiny vs ResNet-18）
```powershell
python toshow\compare_backbones.py --epochs 30
```

**可选参数：**
```powershell
# 指定数据集路径
python toshow\compare_backbones.py --data_dir ./datasets --epochs 30
```

---

### 3. 训练对比模型（多种模型对比）
```powershell
python train_comparison_models.py
```

---

### 4. 运行所有实验
```powershell
python run_all_experiments.py
```

---

## 📊 可视化与分析

### 5. 模型可视化分析（Grad-CAM 热力图、注意力权重）
```powershell
python toshow\visualize_analysis.py
```

**使用说明：**
- 交互式选择模型
- 输入样本标签（如 A1, h1 等）
- 生成 Grad-CAM 热力图和注意力权重图

---

### 6. 可视化预测结果
```powershell
python toshow\visualize_predictions.py
```

**功能：**
- 交互式选择模型
- 生成预测结果的热力图可视化

---

### 7. 显示信心信息
```powershell
python toshow\xian_shi_xin_xi.py
```

---

## 🌐 Web 应用相关

### 8. 启动 Web 应用（柑橘虫害检测网站）
```powershell
cd wangzhan
python app.py
```

**访问地址：** `http://localhost:5000`

---

### 9. 启动 API 服务

#### 检测 API
```powershell
cd wangzhan\api
python detection_api.py
```

#### 图像质量评估 API
```powershell
cd wangzhan\api
python image_quality.py
```

#### RAG API（知识问答）
```powershell
cd wangzhan\api
python rag_api.py
```

#### 搜索 API
```powershell
cd wangzhan\api
python search_api.py
```

#### 网络搜索 API
```powershell
cd wangzhan\api
python web_search.py
```

---

### 10. RAG 知识库
```powershell
cd wangzhan\RAG
python knowledge_base.py
```

---

### 11. 技能模块
```powershell
cd wangzhan\skills
python detection_skill.py
```

---

## 🧪 数据预处理

### 12. 图像预处理器（模块文件，通常不直接运行）
```powershell
# 如需测试，可进入 Python 交互模式
python -c "from data_preprocess import ImagePreprocessor; print('模块加载成功')"
```

---

## 📁 模型定义（模块文件）

### 13. 多视角模型
```powershell
# 模块文件，不直接运行
python -c "from models import MultiViewAttentionFusionModel; print('模块加载成功')"
```

### 14. 对比模型
```powershell
# 模块文件，不直接运行
python -c "from models import SimpleConcatModel, CNNLSTMModel; print('模块加载成功')"
```

---

## ⚙️ 配置文件测试

### 15. 测试配置
```powershell
python config.py
```

### 16. 测试工具函数
```powershell
python utils.py
```

---

## 🔧 常用命令组合

### 完整训练流程
```powershell
cd C:\Users\18656\Desktop\daima\chongju

# 1. 训练主模型
python main.py --epochs 30 --batch_size 12 --lr 0.00003

# 2. 运行骨干网络对比
python toshow\compare_backbones.py --epochs 30

# 3. 可视化分析
python toshow\visualize_analysis.py
```

### 快速测试（1 个 epoch）
```powershell
python main.py --epochs 1 --batch_size 4
```

### 使用指定 GPU
```powershell
# 设置 CUDA 可见设备
$env:CUDA_VISIBLE_DEVICES="0"
python main.py --epochs 30
```

---

## 📝 注意事项

1. **确保已激活虚拟环境**（如果使用）：
   ```powershell
   # 激活虚拟环境（如果有）
   .\venv\Scripts\Activate.ps1
   ```

2. **检查 Python 版本**：
   ```powershell
   python --version
   ```

3. **检查依赖是否安装**：
   ```powershell
   pip list
   ```

4. **如遇导入错误，先安装依赖**：
   ```powershell
   pip install -r shown\requirements.txt
   ```

---

## 🎯 推荐运行方式

**最佳实践：在普通 PowerShell 终端运行，不要使用 IDE 沙箱环境**

1. 按 `Win + R`
2. 输入 `powershell`
3. 回车打开终端
4. 执行上述命令

---

**最后更新时间：** 2026-04-20
