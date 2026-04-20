"""
柑橘检测 API
提供 HTTP 接口用于图像检测
"""
from flask import Blueprint, request, jsonify
from skills.detection_skill import get_detection_skill
import base64
import re

# 创建 Blueprint
detection_bp = Blueprint('detection', __name__, url_prefix='/api/detection')

# 全局 Skill 实例
skill_instance = None


@detection_bp.route('/predict', methods=['POST'])
def predict():
    """
    图像检测接口
    
    支持的输入格式：
    1. multipart/form-data: 上传图像文件
    2. application/json: base64 编码的图像
    
    Returns:
        JSON: {
            'success': bool,
            'result': dict (预测结果),
            'message': str (错误信息)
        }
    """
    global skill_instance
    
    try:
        # 获取 Skill 实例
        if skill_instance is None:
            skill_instance = get_detection_skill()
        
        # 检查输入格式
        if request.files:
            # 处理文件上传
            if 'image' not in request.files:
                return jsonify({
                    'success': False,
                    'message': '未找到图像文件'
                }), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'message': '文件名为空'
                }), 400
            
            # 保存临时文件并预测
            import tempfile
            import os
            from PIL import Image
            
            temp_path = None
            try:
                # 读取文件到内存
                image_data = file.read()
                
                # 验证是否为有效图像
                from io import BytesIO
                img = Image.open(BytesIO(image_data))
                img.verify()
                
                # 重新加载（验证后会关闭）
                img = Image.open(BytesIO(image_data)).convert('RGB')
                
                # 预测
                result = skill_instance.predict(image_pil=img)
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'图像处理失败：{str(e)}'
                }), 400
            
        elif request.json:
            # 处理 base64 编码
            data = request.json
            
            if 'image_base64' not in data:
                return jsonify({
                    'success': False,
                    'message': '未提供 image_base64 字段'
                }), 400
            
            base64_str = data['image_base64']
            
            # 移除可能的 data:image 前缀
            if ',' in base64_str:
                base64_str = re.sub(r'^data:image/.+;base64,', '', base64_str)
            
            result = skill_instance.predict_from_base64(base64_str)
            
        else:
            return jsonify({
                'success': False,
                'message': '不支持的请求格式，请使用 multipart/form-data 或 application/json'
            }), 400
        
        # 返回结果
        return jsonify({
            'success': True,
            'result': result,
            'message': '检测成功'
        })
        
    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'message': f'模型文件未找到：{str(e)}'
        }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'服务器错误：{str(e)}'
        }), 500


@detection_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'skill_loaded': skill_instance is not None
    })
