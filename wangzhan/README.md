# 🍊 柑橘实蝇检测多智能体网站

一个现代化的、具有苹果风格毛玻璃效果的柑橘实蝇检测系统，集成深度学习检测技术和 Qwen 多智能体对话能力，支持移动端访问。

## ✨ 功能特性

### 核心功能
- **📸 图像检测**：上传柑橘图像，使用 ConvNeXt-Tiny 模型进行实蝇侵染检测
- **📊 质量评估**：自动评估图像质量（清晰度、亮度、对比度、色彩），提供 PSNR、SSIM、MSE 等专业指标和改进建议
- **💬 智能对话**：基于 Qwen-plus 的多智能体对话，解读检测结果、回答专业问题
- **🔍 联网搜索**：获取最新的柑橘实蝇防治信息
- **📚 知识库检索**：基于本地 RAG 系统，检索专业知识文档（支持 PDF、Word、JPG 等格式）

### 设计亮点
- **苹果风格毛玻璃**：Glassmorphism 设计，通透感与环境融合
- **移动端优先**：响应式布局，完美适配手机、平板和桌面端
- **动态渐变背景**：柔和的流体渐变动画（15 秒循环）
- **流畅交互动画**：所有状态切换都有平滑过渡效果
- **实时交互反馈**：图像上传引导、检测结果可视化、对话气泡展示

### 技术优势
- **本地部署**：所有模型推理在本地 GPU 完成，保护数据隐私
- **离线可用**：检测功能完全离线，无需联网
- **跨平台访问**：支持手机、平板、电脑全平台访问
- **公网访问**：支持 Ngrok/Cloudflare/Frp 内网穿透，全网可访问

## 🚀 快速开始

### 方式一：直接启动（推荐）

```bash
# 打开 PowerShell，进入项目目录
cd C:\Users\18656\Desktop\daima\chongju\wangzhan

# 启动服务器
C:\Users\18656\.conda\envs\chongju\python.exe app.py
```

### 方式二：使用启动脚本

```bash
# 创建启动脚本（首次使用）
echo 'cd C:\Users\18656\Desktop\daima\chongju\wangzhan; C:\Users\18656\.conda\envs\chongju\python.exe app.py' > start.ps1

# 以后只需运行
.\start.ps1
```

### 访问地址

启动成功后会显示：
```
============================================================
🍊 柑橘实蝇检测多智能体网站启动中...
============================================================
📍 本地访问：http://127.0.0.1:5000
📱 移动端访问：http://192.168.1.2:5000
🔧 调试模式：True
============================================================
```

- **电脑访问**：http://127.0.0.1:5000
- **手机访问**：http://[你的局域网 IP]:5000

> 📝 **注意**：确保手机和电脑连接同一 WiFi 网络

### 方式三：公网访问（任何地方都能访问）

```bash
# 使用 Ngrok 快速启动公网访问
.\start_public.ps1
```

启动后会显示公网 URL：
```
🌐 公网 URL: https://abc123.ngrok.io

现在可以在任何设备上访问！
```

**详细说明**：查看 [`公网访问配置指南.md`](公网访问配置指南.md)

## 📱 移动端访问配置

### 步骤 1：获取局域网 IP

在 PowerShell 中运行：
```powershell
ipconfig
```

找到类似信息：
```
无线局域网适配器 WLAN:
   IPv4 地址 . . . . . . . . . . . . : 192.168.1.2
```

### 步骤 2：配置防火墙（如需要）

以**管理员身份**运行 PowerShell：
```powershell
netsh advfirewall firewall add rule name="Python Web Server" dir=in action=allow protocol=TCP localport=5000
```

### 步骤 3：手机访问

1. 确保手机连接同一 WiFi
2. 打开手机浏览器（Safari/Chrome）
3. 输入：`http://192.168.1.2:5000`
4. 开始使用！

## 🎨 设计特色

### 色彩体系
- **柑橘橙**：`#FF9F0A` - 强调色，契合柑橘主题
- **苹果绿**：`#34C759` - 健康状态
- **苹果红**：`#FF3B30` - 患虫警告
- **深灰文本**：`#1D1D1F` - 主文本色
- **次文本**：`#86868B` - 副标题和提示

### 毛玻璃效果
- **背景**：`rgba(255, 255, 255, 0.7)` 半透明白色
- **模糊**：`backdrop-filter: blur(24px)` 核心毛玻璃
- **边框**：`1px solid rgba(255, 255, 255, 0.5)` 高光边缘
- **阴影**：`0 12px 40px rgba(0, 0, 0, 0.08)` 悬浮感

### 圆角系统
- **外层容器**：28px
- **内部卡片**：16px
- **胶囊按钮**：100px

## 📁 项目结构

```
wangzhan/
├── app.py                      # Flask 主应用（服务器入口）
├── config.py                   # 配置文件（API 密钥、端口等）
├── requirements.txt            # 依赖列表
├── .env                        # 环境变量（API 密钥）
├── api/                        # API 接口
│   ├── detection_api.py        # 图像检测 API
│   └── rag_api.py              # RAG 检索 API
├── skills/                     # Skill 模块
│   └── detection_skill.py      # 柑橘检测 Skill
├── rag/                        # RAG 知识库
│   ├── knowledge_base.py       # 知识库实现
│   └── knowledge_base.txt      # 示例文档
├── static/                     # 静态资源
│   └── css/
│       └── style.css           # 样式文件（毛玻璃效果）
├── templates/
│   └── index.html              # 主页面（包含所有样式和脚本）
└── test_website.py             # 系统测试脚本
```

## 🎯 使用指南

### 图像检测流程
1. **上传图像**：点击或拖拽柑橘图像到上传区域
2. **引导提示**：虚线框提示将柑橘置于中心
3. **开始检测**：点击"开始智能检测"按钮
4. **查看结果**：
   - 健康/患虫分类
   - 置信度百分比
   - 概率进度条可视化
5. **对话解读**：自动在对话窗口解读检测结果

### 对话功能
- **普通对话**：在输入框输入问题，按 Enter 或点击发送
- **功能按钮**：
  - 🔍 **联网搜索**：获取实时信息（开发中）
  - 📚 **知识库**：检索本地文档（开发中）
  - 🗑️ **清除记忆**：重置对话历史

### 知识库管理
将专业知识文档放入 `rag/` 目录，系统会自动加载和向量化。

## 🔧 技术栈

### 后端
- **Web 框架**：Flask 3.0, Flask-CORS
- **深度学习**：PyTorch 2.11+cu128, timm 0.9.12, torchvision
- **模型架构**：ConvNeXt-Tiny
- **RAG 系统**：sentence-transformers, FAISS
- **AI 对话**：Qwen-plus API

### 前端
- **HTML5**：语义化结构
- **CSS3**：自定义属性、动画、响应式布局
- **JavaScript**：原生 ES6+，异步编程

### 开发环境
- **Python**：3.11.15
- **CUDA**：12.8
- **GPU**：NVIDIA GeForce RTX 5060 (8GB)

## 📊 系统要求

### 最低配置
- **操作系统**：Windows 10/11
- **Python**：3.11+
- **内存**：8GB
- **存储**：2GB 可用空间

### 推荐配置
- **GPU**：NVIDIA RTX 3060+ (支持 CUDA 12.x)
- **内存**：16GB
- **网络**：WiFi（用于移动端访问）

## ⚙️ 配置说明

### config.py 主要参数

```python
# Qwen API 配置
QWEN_API_KEY = "sk-..."          # 必填
QWEN_MODEL = "qwen-plus"         # 模型名称

# 模型路径
SINGLE_VIEW_MODEL_PATH = "C:\\Users\\18656\\Desktop\\daima\\chongju\\checkpoints\\single_view\\best_model.pth"

# 网站配置
HOST = "0.0.0.0"                 # 0.0.0.0 支持移动端访问
PORT = 5000                      # 端口号
DEBUG = True                     # 调试模式

# 系统配置
DEVICE = "cuda"                  # "cuda" 或 "cpu"
IMAGE_SIZE = 224                 # 输入图像尺寸
NUM_CLASSES = 2                  # 分类数量
CLASS_NAMES = ["健康", "患虫"]   # 类别名称
```

### 环境变量（.env）

```bash
# Qwen API 密钥
QWEN_API_KEY=sk-your-actual-api-key-here
```

## 🐛 常见问题

### 服务器相关

**Q: 移动端无法访问？**  
A: 
1. 确保 `HOST = "0.0.0.0"`（不是 `127.0.0.1`）
2. 检查防火墙设置，允许 5000 端口
3. 确认手机和电脑在同一 WiFi 网络

**Q: 端口 5000 已被占用？**  
A: 
```bash
# 查找占用进程
netstat -ano | findstr :5000
# 停止进程
Stop-Process -Id [PID] -Force
# 或修改 config.py 中的 PORT
```

**Q: 服务器启动后立即退出？**  
A: 
- 检查是否有错误日志
- 确认模型文件路径正确
- 验证 API 密钥配置

### 模型相关

**Q: 模型加载失败？**  
A: 
- 检查模型文件是否存在：`best_model.pth`
- 确认路径在 config.py 中正确配置
- 检查文件大小（约 324MB）

**Q: 检测时提示"too many indices for tensor"？**  
A: 
- 已修复，确保 detection_skill.py 中正确预处理图像
- 图像会被复制 5 份以匹配模型输入要求

**Q: GPU 内存不足？**  
A: 
- 在 config.py 中设置 `DEVICE = "cpu"`
- 或关闭其他占用 GPU 的程序

### API 相关

**Q: Qwen API 调用失败？**  
A: 
- 检查 `.env` 文件中的 `QWEN_API_KEY` 是否正确
- 确认 API 密钥有足够额度
- 检查网络连接

**Q: RAG 索引创建失败？**  
A: 
- 安装依赖：`pip install sentence-transformers faiss-cpu`
- 确保 `rag/` 目录存在且有文档

### 前端相关

**Q: 页面显示不正常？**  
A: 
- 清除浏览器缓存
- 使用现代浏览器（Chrome、Safari、Edge）
- 检查是否正确加载 CSS 文件

**Q: 上传图片后无法检测？**  
A: 
- 检查浏览器控制台（F12）是否有错误
- 确认服务器正在运行
- 验证图片格式（支持 JPG、PNG）

## 📝 开发笔记

### 前后端互通原理

```
前端 (index.html)          后端 (app.py)
     │                         │
     │  fetch('/api/...)       │
     ├────────────────────────►│  @app.route('/api/...')
     │  POST image data        │     ↓
     │                         │  skill.predict()
     │                         │     ↓
     │◄────────────────────────┤  return JSON
     │  JSON response          │
     │                         │
     ▼
更新页面显示
```

### 服务器启动流程

1. **加载配置**：读取 config.py 和 .env
2. **初始化模型**：加载 ConvNeXt-Tiny 权重
3. **创建路由**：注册所有 API 接口
4. **启动服务**：监听 0.0.0.0:5000
5. **等待请求**：处理浏览器发送的 HTTP 请求

### 关键代码位置

- **服务器入口**：`app.py` 第 254-278 行
- **检测 API**：`api/detection_api.py`
- **模型推理**：`skills/detection_skill.py`
- **前端页面**：`templates/index.html`

## 📚 学习资源

### Flask 入门
- [Flask 官方文档](https://flask.palletsprojects.com/)
- [MDN HTTP 指南](https://developer.mozilla.org/zh-CN/docs/Web/HTTP)

### 深度学习
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- [timm 库文档](https://github.com/huggingface/pytorch-image-models)

### 前端开发
- [MDN Web 文档](https://developer.mozilla.org/zh-CN/)
- [CSS  Tricks](https://css-tricks.com/)

## � 更新日志

### v1.1.0 (2026-04-11)
- ✨ 苹果风格毛玻璃 UI 全面升级
- 📱 支持移动端访问（0.0.0.0 监听）
- 🎨 动态渐变背景动画
- 🍊 完全参考设计稿优化
- 🔧 修复图像预处理维度问题
- 📚 更新文档和快速启动指南

### v1.0.0 (2026-04-10)
- ✨ 初始版本发布
- 🎨 高级感 UI 设计
- 🤖 集成 Qwen 多智能体
- 📸 单视角检测功能
- 📚 RAG 知识库系统

## 👨‍💻 开发团队

柑橘实蝇检测项目组

## 📄 许可证

本项目仅供科研使用。

## 📞 联系方式

如有问题，请联系项目组或查看 [QUICKSTART.md](QUICKSTART.md) 获取详细启动指南。
