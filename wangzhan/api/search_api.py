"""
联网搜索 API
提供实时网络搜索功能
"""
from flask import Blueprint, request, jsonify
from api.web_search import get_web_search

# 创建 Blueprint
search_bp = Blueprint('search', __name__, url_prefix='/api/search')

# 全局搜索实例
search_instance = None


@search_bp.route('/web', methods=['POST'])
def web_search():
    """
    联网搜索接口
    
    Request:
        JSON: {
            'query': str (搜索关键词),
            'num_results': int (返回结果数量，默认 5)
        }
    
    Returns:
        JSON: {
            'success': bool,
            'results': List[Dict] (搜索结果),
            'message': str
        }
    """
    global search_instance
    
    try:
        # 获取搜索实例
        if search_instance is None:
            search_instance = get_web_search()
        
        # 解析请求
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'message': '请求体为空'
            }), 400
        
        query = data.get('query', '')
        num_results = data.get('num_results', 5)
        
        if not query:
            return jsonify({
                'success': False,
                'message': '搜索关键词不能为空'
            }), 400
        
        # 执行搜索
        results = search_instance.search(query, num_results=num_results)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': '搜索成功'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500


@search_bp.route('/news', methods=['POST'])
def news_search():
    """
    新闻搜索接口
    
    Request:
        JSON: {
            'query': str (搜索关键词),
            'num_results': int (返回结果数量，默认 3)
        }
    
    Returns:
        JSON: {
            'success': bool,
            'results': List[Dict] (新闻列表),
            'message': str
        }
    """
    global search_instance
    
    try:
        # 获取搜索实例
        if search_instance is None:
            search_instance = get_web_search()
        
        # 解析请求
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'message': '请求体为空'
            }), 400
        
        query = data.get('query', '')
        num_results = data.get('num_results', 3)
        
        if not query:
            return jsonify({
                'success': False,
                'message': '搜索关键词不能为空'
            }), 400
        
        # 执行新闻搜索
        results = search_instance.get_news(query, num_results=num_results)
        
        return jsonify({
            'success': True,
            'results': results,
            'message': '新闻搜索成功'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500


@search_bp.route('/status', methods=['GET'])
def status():
    """
    搜索服务状态查询
    
    Returns:
        JSON: {
            'success': bool,
            'engine': str (搜索引擎),
            'api_configured': bool (API 是否配置)
        }
    """
    global search_instance
    
    try:
        if search_instance is None:
            search_instance = get_web_search()
        
        return jsonify({
            'success': True,
            'engine': search_instance.engine,
            'api_configured': search_instance.api_key is not None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'查询失败：{str(e)}'
        }), 500
