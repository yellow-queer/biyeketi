"""
柑橘检测多智能体网站 - 后端服务
集成：
- 图像检测 API
- RAG 知识库 API
- Qwen 多智能体对话
- 联网搜索功能
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import json
from config import (
    QWEN_API_KEY, QWEN_API_BASE_URL, QWEN_MODEL,
    HOST, PORT, DEBUG
)
from api.detection_api import detection_bp
from api.rag_api import rag_bp
from api.search_api import search_bp
from api.image_quality import image_quality_bp

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 注册 Blueprint
app.register_blueprint(detection_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(search_bp)
app.register_blueprint(image_quality_bp)

# 对话历史存储（简单实现，生产环境应使用数据库）
conversation_histories = {}


# ==================== 页面路由 ====================

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


# ==================== Qwen 多智能体对话 API ====================

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    聊天接口 - 集成 Qwen 多智能体
    
    Request:
        JSON: {
            'message': str (用户消息),
            'conversation_id': str (对话 ID，可选),
            'mode': str (模式：'chat'/'search'/'rag', 默认'chat'),
            'image_result': dict (可选，图像检测结果)
        }
    
    Returns:
        JSON: {
            'success': bool,
            'response': str (AI 响应),
            'conversation_id': str
        }
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        conversation_id = data.get('conversation_id', 'default')
        mode = data.get('mode', 'chat')
        image_result = data.get('image_result', None)
        
        if not user_message:
            return jsonify({
                'success': False,
                'message': '消息不能为空'
            }), 400
        
        # 获取或创建对话历史
        if conversation_id not in conversation_histories:
            conversation_histories[conversation_id] = []
        
        history = conversation_histories[conversation_id]
        
        # 构建系统提示
        system_prompt = "你是一位专注于柑橘实蝇检测的科研助手，性格实惠、友好。你的主要任务是帮助用户了解柑橘检测技术、实蝇防治知识，并解读检测结果。"
        
        # 根据模式调整提示
        if mode == 'search':
            system_prompt += " 你可以使用联网搜索获取最新信息。"
        elif mode == 'rag':
            system_prompt += " 你可以参考本地知识库中的专业文档。"
        
        # 如果有图像检测结果，添加到上下文
        context_messages = []
        if image_result:
            detection_info = (
                f"用户上传了一张柑橘图像，检测结果显示：{image_result['class']}，"
                f"置信度为{image_result['confidence']*100:.2f}%。"
                f"健康概率：{image_result['probabilities']['健康']*100:.2f}%，"
                f"患虫概率：{image_result['probabilities']['患虫']*100:.2f}%。"
                f"请根据这个检测结果为用户提供专业建议。"
            )
            context_messages.append({"role": "system", "content": detection_info})
        
        # 添加用户消息到历史
        history.append({"role": "user", "content": user_message})
        
        # 构建 API 请求
        messages = [
            {"role": "system", "content": system_prompt}
        ] + context_messages + history
        
        # 调用 Qwen API
        headers = {
            "Authorization": f"Bearer {QWEN_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": QWEN_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(
            f"{QWEN_API_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Qwen API 调用失败：{response.status_code} - {response.text}")
        
        # 解析响应
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        
        # 添加到历史
        history.append({"role": "assistant", "content": assistant_message})
        
        # 限制历史长度（保留最近 20 条）
        if len(history) > 20:
            history = history[-20:]
            conversation_histories[conversation_id] = history
        
        return jsonify({
            'success': True,
            'response': assistant_message,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500


@app.route('/api/chat/clear', methods=['POST'])
def clear_history():
    """清除对话历史"""
    try:
        data = request.json
        conversation_id = data.get('conversation_id', 'default')
        
        if conversation_id in conversation_histories:
            conversation_histories[conversation_id] = []
        
        return jsonify({
            'success': True,
            'message': '对话历史已清除'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'清除失败：{str(e)}'
        }), 500


# ==================== 联网搜索 API ====================

@app.route('/api/search', methods=['POST'])
def web_search():
    """
    联网搜索接口
    
    Request:
        JSON: {
            'query': str (搜索关键词)
        }
    
    Returns:
        JSON: {
            'success': bool,
            'results': List[Dict] (搜索结果),
            'message': str
        }
    """
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({
                'success': False,
                'message': '搜索关键词不能为空'
            }), 400
        
        # 使用 Qwen 的联网搜索能力（通过工具调用）
        # 这里简化实现，直接返回提示
        search_tip = f"正在搜索\"{query}\"相关信息..."
        
        # 如果有搜索 API，可以在这里调用
        # 目前返回提示信息，实际搜索由 Qwen 处理
        
        return jsonify({
            'success': True,
            'results': [
                {
                    'title': '联网搜索',
                    'snippet': search_tip,
                    'link': ''
                }
            ],
            'message': '搜索请求已接收'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'搜索失败：{str(e)}'
        }), 500


# ==================== 系统状态 API ====================

@app.route('/api/status', methods=['GET'])
def system_status():
    """系统状态查询"""
    return jsonify({
        'success': True,
        'status': 'running',
        'services': {
            'detection': 'available',
            'rag': 'available',
            'chat': 'available',
            'search': 'available'
        }
    })


if __name__ == '__main__':
    import socket
    
    # 获取本机局域网 IP
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return '127.0.0.1'
    
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("🍊 柑橘检测多智能体网站启动中...")
    print("=" * 60)
    print(f"📍 本地访问：http://127.0.0.1:{PORT}")
    print(f"📱 移动端访问：http://{local_ip}:{PORT}")
    print(f"🔧 调试模式：{DEBUG}")
    print("=" * 60)
    
    app.run(host=HOST, port=PORT, debug=DEBUG)
