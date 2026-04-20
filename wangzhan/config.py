"""
网站配置文件
包含 Qwen API 配置、模型路径、系统配置等
"""
import os
from dotenv import load_dotenv

# 加载.env 文件
load_dotenv()

# ==================== Qwen API 配置 ====================
QWEN_API_KEY = os.getenv("QWEN_API_KEY")  # 从环境变量读取
QWEN_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-plus" # 使用 qwen-plus，性价比高，响应快速

# ==================== 模型路径配置 ====================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SINGLE_VIEW_MODEL_PATH = r"C:\Users\18656\Desktop\daima\chongju\checkpoints\single_view\best_model.pth"
CHONGJU_PROJECT_ROOT = r"C:\Users\18656\Desktop\daima\chongju"

# ==================== RAG 配置 ====================
# 📝 任务二修改：RAG 知识库文档目录
RAG_DOCUMENTS_PATH = r"C:\Users\18656\Desktop\daima\chongju\wangzhan\RAG\wenjian"  # 知识库文档存放目录
RAG_INDEX_PATH = os.path.join(PROJECT_ROOT, "rag_index")  # 向量索引保存目录
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==================== 系统配置 ====================
DEVICE = "cuda"  # 或 "cpu"
IMAGE_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["健康", "患虫"]

# ==================== 网站配置 ====================
HOST = "0.0.0.0"  # 监听所有网络接口，支持移动端/局域网访问
PORT = 5000
DEBUG = True

# ==================== 公网访问配置 ====================
# 🌐 任务一新增：公网访问配置
# 说明：默认支持局域网访问，如需全网（公网）访问，请选择以下方案之一
# 方案 1：Ngrok（推荐，简单快速）
#   1. 注册：https://ngrok.com/
#   2. 下载：https://ngrok.com/download
#   3. 运行：ngrok http 5000
#   4. 获取公网 URL：https://xxxx.ngrok.io
ENABLE_PUBLIC_ACCESS = True # 是否启用公网访问（True/False）
PUBLIC_ACCESS_METHOD = "ngrok"  # 公网访问方式："ngrok", "frp", "cloudflare"
NGROK_AUTH_TOKEN = ""  # Ngrok 认证令牌（可选，提高限额）

# 方案 2：Frp 内网穿透（需要有自己的服务器）
#   1. 配置 frpc.ini
#   2. 运行 frpc -c frpc.ini
FRP_SERVER_ADDR = ""  # FRP 服务器地址
FRP_SERVER_PORT = 0  # FRP 服务器端口

# 方案 3：Cloudflare Tunnel（免费，安全）
#   1. 安装 cloudflared
#   2. 运行：cloudflared tunnel --url http://localhost:5000
CLOUDFLARE_ENABLED = False  # 是否启用 Cloudflare Tunnel

# ==================== 搜索 API 配置 ====================
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")  # 从环境变量读取（可选）
SEARCH_API_URL = "https://api.bing.microsoft.com/v7.0/search"  # Bing Search API
