"""
图像质量评估 API

提供图像质量评估功能，包括：
- PSNR (Peak Signal-to-Noise Ratio) - 峰值信噪比
- SSIM (Structural Similarity Index) - 结构相似性
- MSE (Mean Squared Error) - 均方误差
- 亮度评估
- 对比度评估
- 清晰度评估
"""

from flask import Blueprint, request, jsonify
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

image_quality_bp = Blueprint('image_quality', __name__, url_prefix='/api/image-quality')


def calculate_mse(image1, image2):
    """
    计算均方误差 (MSE)
    
    Args:
        image1: 参考图像
        image2: 比较图像
    
    Returns:
        float: MSE 值
    """
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err


def calculate_psnr(image1, image2):
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        image1: 参考图像
        image2: 比较图像
    
    Returns:
        float: PSNR 值 (dB)
    """
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def calculate_ssim(image1, image2):
    """
    计算结构相似性 (SSIM)
    
    Args:
        image1: 参考图像
        image2: 比较图像
    
    Returns:
        float: SSIM 值 (0-1)
    """
    # 转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # 计算 SSIM
    score, _ = ssim(gray1, gray2, full=True)
    return score


def assess_image_quality(image):
    """
    综合评估图像质量
    
    Args:
        image: OpenCV 图像 (BGR)
    
    Returns:
        dict: 包含各项质量指标
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 清晰度评估（拉普拉斯方差）
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. 亮度评估
    brightness = np.mean(gray)
    
    # 3. 对比度评估
    contrast = np.std(gray)
    
    # 4. 色彩丰富度（如果有颜色信息）
    if len(image.shape) == 3:
        color_std = np.std(image, axis=(0, 1)).mean()
    else:
        color_std = 0
    
    # 5. 噪声估计（使用局部方差）
    # 使用 3x3 邻域计算局部方差
    kernel = np.ones((3, 3), np.float32) / 9.0
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
    noise_estimate = np.mean(local_var)
    
    # 计算综合质量分数 (0-100)
    # 清晰度占 30 分
    sharpness_score = min(blur_score / 200.0, 1.0) * 30
    
    # 亮度占 20 分（理想亮度 128）
    brightness_score = (1 - abs(brightness - 128) / 128.0) * 20
    
    # 对比度占 20 分
    contrast_score = min(contrast / 50.0, 1.0) * 20
    
    # 色彩丰富度占 15 分
    color_score = min(color_std / 50.0, 1.0) * 15
    
    # 噪声占 15 分（噪声越低分越高）
    noise_score = (1 - min(noise_estimate / 1000.0, 1.0)) * 15
    
    overall_score = sharpness_score + brightness_score + contrast_score + color_score + noise_score
    
    return {
        'sharpness': {
            'score': round(blur_score, 2),
            'normalized': round(min(blur_score / 200.0, 1.0), 3),
            'level': '优秀' if blur_score > 200 else '良好' if blur_score > 100 else '一般' if blur_score > 50 else '较差'
        },
        'brightness': {
            'value': round(brightness, 2),
            'normalized': round(1 - abs(brightness - 128) / 128.0, 3),
            'level': '过暗' if brightness < 50 else '偏暗' if brightness < 100 else '适中' if brightness < 180 else '偏亮' if brightness < 220 else '过曝'
        },
        'contrast': {
            'value': round(contrast, 2),
            'normalized': round(min(contrast / 50.0, 1.0), 3),
            'level': '优秀' if contrast > 60 else '良好' if contrast > 40 else '一般' if contrast > 20 else '较差'
        },
        'color_richness': {
            'value': round(color_std, 2),
            'normalized': round(min(color_std / 50.0, 1.0), 3),
            'level': '丰富' if color_std > 60 else '适中' if color_std > 30 else '单调'
        },
        'noise': {
            'estimate': round(noise_estimate, 2),
            'normalized': round(1 - min(noise_estimate / 1000.0, 1.0), 3),
            'level': '低' if noise_estimate < 100 else '中' if noise_estimate < 500 else '高'
        },
        'overall_score': round(overall_score, 1),
        'quality_level': '优秀' if overall_score >= 80 else '良好' if overall_score >= 60 else '一般' if overall_score >= 40 else '较差'
    }


def get_quality_suggestions(quality_result):
    """
    根据质量评估结果生成改进建议
    
    Args:
        quality_result: assess_image_quality 返回的结果
    
    Returns:
        list: 改进建议列表
    """
    suggestions = []
    
    # 清晰度建议
    if quality_result['sharpness']['normalized'] < 0.5:
        suggestions.append({
            'aspect': '清晰度',
            'issue': f"图像模糊 (得分：{quality_result['sharpness']['score']})",
            'suggestion': '请重新拍摄，确保对焦清晰。可以使用三脚架或稳定设备。'
        })
    
    # 亮度建议
    brightness_level = quality_result['brightness']['level']
    if brightness_level in ['过暗', '偏暗']:
        suggestions.append({
            'aspect': '亮度',
            'issue': f"图像过暗 (亮度值：{quality_result['brightness']['value']})",
            'suggestion': '请增加光线或使用闪光灯。避免在昏暗环境下拍摄。'
        })
    elif brightness_level in ['偏亮', '过曝']:
        suggestions.append({
            'aspect': '亮度',
            'issue': f"图像过亮 (亮度值：{quality_result['brightness']['value']})",
            'suggestion': '请减少光线或调整拍摄角度。避免强光直射。'
        })
    
    # 对比度建议
    if quality_result['contrast']['normalized'] < 0.5:
        suggestions.append({
            'aspect': '对比度',
            'issue': f"对比度不足 (对比度：{quality_result['contrast']['value']})",
            'suggestion': '请调整拍摄角度，增强光影对比。可以尝试侧光拍摄。'
        })
    
    # 色彩建议
    if quality_result['color_richness']['normalized'] < 0.5:
        suggestions.append({
            'aspect': '色彩',
            'issue': f"色彩单调 (色彩丰富度：{quality_result['color_richness']['value']})",
            'suggestion': '请确保在自然光下拍摄，避免白平衡设置不当。'
        })
    
    # 噪声建议
    if quality_result['noise']['normalized'] < 0.5:
        suggestions.append({
            'aspect': '噪声',
            'issue': f"图像噪声较大 (噪声估计：{quality_result['noise']['estimate']})",
            'suggestion': '请使用较低 ISO 值拍摄，或后期进行降噪处理。'
        })
    
    return suggestions


@image_quality_bp.route('/assess', methods=['POST'])
def assess_quality():
    """
    图像质量评估接口
    
    支持的输入格式：
    1. multipart/form-data: 上传图像文件
    
    Returns:
        JSON: {
            'success': bool,
            'quality': dict (质量评估结果),
            'suggestions': list (改进建议),
            'message': str (错误信息)
        }
    """
    try:
        # 检查是否有上传文件
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': '请上传图像文件'
            }), 400
        
        file = request.files['image']
        
        # 检查文件是否为空
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        # 检查文件类型
        allowed_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({
                'success': False,
                'message': '不支持的文件格式，请上传 JPG、PNG、BMP 或 WebP 格式'
            }), 400
        
        # 读取图像
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': '无法读取图像文件'
            }), 400
        
        # 评估图像质量
        quality_result = assess_image_quality(image)
        
        # 生成改进建议
        suggestions = get_quality_suggestions(quality_result)
        
        # 返回结果
        return jsonify({
            'success': True,
            'quality': quality_result,
            'suggestions': suggestions,
            'image_info': {
                'filename': file.filename,
                'size': len(file.read()) if hasattr(file, 'read') else 0,
                'dimensions': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                }
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'质量评估失败：{str(e)}'
        }), 500


@image_quality_bp.route('/compare', methods=['POST'])
def compare_images():
    """
    图像质量对比接口（比较两张图像的差异）
    
    Returns:
        JSON: {
            'success': bool,
            'psnr': float,
            'ssim': float,
            'mse': float,
            'message': str
        }
    """
    try:
        # 检查是否有上传文件
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({
                'success': False,
                'message': '请上传两张图像文件'
            }), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        # 读取图像 1
        npimg1 = np.frombuffer(file1.read(), np.uint8)
        image1 = cv2.imdecode(npimg1, cv2.IMREAD_COLOR)
        
        # 读取图像 2
        npimg2 = np.frombuffer(file2.read(), np.uint8)
        image2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)
        
        if image1 is None or image2 is None:
            return jsonify({
                'success': False,
                'message': '无法读取图像文件'
            }), 400
        
        # 检查尺寸是否相同
        if image1.shape != image2.shape:
            return jsonify({
                'success': False,
                'message': f'两张图像尺寸不匹配：{image1.shape} vs {image2.shape}'
            }), 400
        
        # 计算各项指标
        mse_value = calculate_mse(image1, image2)
        psnr_value = calculate_psnr(image1, image2)
        ssim_value = calculate_ssim(image1, image2)
        
        return jsonify({
            'success': True,
            'metrics': {
                'mse': round(mse_value, 4),
                'psnr': round(psnr_value, 2) if psnr_value != float('inf') else 'inf',
                'ssim': round(ssim_value, 4)
            },
            'interpretation': {
                'psnr': '越高越好 (>30dB 表示质量很好)' if psnr_value != float('inf') else '完全相同',
                'ssim': '越接近 1 越好 (>0.9 表示非常相似)',
                'mse': '越低越好 (=0 表示完全相同)'
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'图像对比失败：{str(e)}'
        }), 500
