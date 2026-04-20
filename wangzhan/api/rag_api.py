"""
RAG 检索 API
提供知识库检索功能
"""
from flask import Blueprint, request, jsonify
from RAG.knowledge_base import get_knowledge_base

# 创建 Blueprint
rag_bp = Blueprint('rag', __name__, url_prefix='/api/rag')

# 全局知识库实例
kb_instance = None


@rag_bp.route('/search', methods=['POST'])
def search():
    """
    知识库检索接口
    
    Request:
        JSON: {
            'query': str (查询文本),
            'top_k': int (返回结果数量，默认 3)
        }
    
    Returns:
        JSON: {
            'success': bool,
            'results': List[Dict] (检索结果),
            'message': str
        }
    """
    global kb_instance
    
    try:
        # 获取知识库实例
        if kb_instance is None:
            kb_instance = get_knowledge_base()
        
        # 解析请求
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'message': '请求体为空'
            }), 400
        
        query = data.get('query', '')
        top_k = data.get('top_k', 3)
        
        if not query:
            return jsonify({
                'success': False,
                'message': '查询文本不能为空'
            }), 400
        
        # 检索
        results = kb_instance.search(query, top_k=top_k)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': '检索成功'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500


@rag_bp.route('/rebuild', methods=['POST'])
def rebuild_index():
    """
    重建知识库索引
    用于在文档更新后手动触发重建
    
    Returns:
        JSON: {
            'success': bool,
            'message': str
        }
    """
    global kb_instance
    
    try:
        # 重新创建实例以刷新
        kb_instance = None
        kb_instance = get_knowledge_base()
        
        return jsonify({
            'success': True,
            'message': '索引重建完成'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'重建失败：{str(e)}'
        }), 500


@rag_bp.route('/status', methods=['GET'])
def status():
    """
    知识库状态查询
    
    Returns:
        JSON: {
            'success': bool,
            'document_count': int,
            'index_loaded': bool
        }
    """
    global kb_instance
    
    try:
        if kb_instance is None:
            kb_instance = get_knowledge_base()
        
        return jsonify({
            'success': True,
            'document_count': len(kb_instance.documents) if kb_instance.documents else 0,
            'index_loaded': kb_instance.index is not None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'查询失败：{str(e)}'
        }), 500
