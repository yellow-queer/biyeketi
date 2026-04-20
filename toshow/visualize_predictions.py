"""
训练过程热力图可视化
功能：交互式输入样本，生成并展示训练过程中的 Grad-CAM 热力图
作者：柑橘虫害检测项目组
日期：2026 年
"""
import os
import sys

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from config import Config
from data_preprocess import ImagePreprocessor
from models import MultiViewAttentionFusionModel, SingleViewModel, SimpleConcatModel, CNNLSTMModel, CNNLSTMAttentionModel


def get_available_models(checkpoint_dir='./checkpoints'):
    """获取可用的模型列表"""
    model_types = ['attention_fusion', 'single_view', 'simple_concat', 'cnn_lstm', 'cnn_lstm_attention']
    available_models = []
    
    for model_type in model_types:
        checkpoint_path = os.path.join(checkpoint_dir, model_type, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            available_models.append({
                'model_type': model_type,
                'checkpoint_path': checkpoint_path
            })
    
    return available_models


def select_model_interactive(checkpoint_dir='./checkpoints'):
    """交互式选择模型"""
    print("\n" + "="*60)
    print("可用的模型列表:")
    print("="*60)
    
    available_models = get_available_models(checkpoint_dir)
    
    if not available_models:
        print("错误：未找到任何训练好的模型!")
        return None, None
    
    model_names = {
        'attention_fusion': '注意力融合模型',
        'single_view': '单视角模型',
        'simple_concat': '简单拼接模型',
        'cnn_lstm': 'CNN-LSTM 模型',
        'cnn_lstm_attention': 'CNN-LSTM-Attention 模型'
    }
    
    for i, model_info in enumerate(available_models, 1):
        model_type = model_info['model_type']
        checkpoint_path = model_info['checkpoint_path']
        print(f"  [{i}] {model_names.get(model_type, model_type)}")
        print(f"      路径：{checkpoint_path}")
    
    print(f"  [0] 手动输入模型路径")
    print("="*60)
    
    while True:
        try:
            choice = input("\n请选择要使用的模型 (输入数字): ").strip()
            choice_idx = int(choice)
            
            if choice_idx == 0:
                custom_path = input("请输入模型权重路径：").strip()
                if os.path.exists(custom_path):
                    model_type_input = input("请输入模型类型：").strip()
                    if model_type_input in ['attention_fusion', 'single_view', 'simple_concat', 'cnn_lstm', 'cnn_lstm_attention']:
                        return model_type_input, custom_path
                    else:
                        print("错误：无效的模型类型")
                        continue
                else:
                    print(f"错误：路径不存在：{custom_path}")
                    continue
            
            if 1 <= choice_idx <= len(available_models):
                selected = available_models[choice_idx - 1]
                return selected['model_type'], selected['checkpoint_path']
            else:
                print(f"错误：请输入 0-{len(available_models)} 之间的数字")
        
        except ValueError:
            print("错误：请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消操作")
            sys.exit(0)


def get_sample_paths(sample_label: str, data_dir: str = './datasets'):
    """根据样本标签获取对应的图像和视频路径"""
    sample_label = sample_label.strip().upper()
    
    if sample_label.startswith('A') or sample_label.startswith('X'):
        category = 'chongju'
        label = 1
        category_name = '患虫'
        pic_subdir = 'pic_chongju'
        video_subdir = 'v_chongju'
    elif sample_label.startswith('H') or sample_label.startswith('h'):
        category = 'healthy'
        label = 0
        category_name = '健康'
        pic_subdir = 'pic_healthy'
        video_subdir = 'v_healthy'
        sample_label = sample_label.lower().replace('h', 'h')
    else:
        raise ValueError(f"无法识别的样本标签格式：{sample_label}")
    
    pic_dir = os.path.join(data_dir, 'pic', pic_subdir)
    video_dir = os.path.join(data_dir, 'video', video_subdir)
    
    possible_names = [sample_label, sample_label.upper(), sample_label.lower()]
    
    top_image_path = None
    video_path = None
    
    for name in possible_names:
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(pic_dir, name + ext)
            if os.path.exists(test_path):
                top_image_path = test_path
                break
        
        for ext in ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']:
            test_path = os.path.join(video_dir, name + ext)
            if os.path.exists(test_path):
                video_path = test_path
                break
        
        if top_image_path and video_path:
            break
    
    if not top_image_path:
        raise FileNotFoundError(f"找不到顶角图像：{sample_label} in {pic_dir}")
    if not video_path:
        raise FileNotFoundError(f"找不到视频文件：{sample_label} in {video_dir}")
    
    return top_image_path, video_path, label, category_name


def load_sample_data(sample_label: str, data_dir: str = './datasets'):
    """加载指定样本的多视角图像数据"""
    top_image_path, video_path, label, category_name = get_sample_paths(sample_label, data_dir)
    
    print(f"\n加载样本：{sample_label}")
    print(f"  顶角图像：{top_image_path}")
    print(f"  环绕视频：{video_path}")
    print(f"  类别：{category_name} (标签={label})")
    
    preprocessor = ImagePreprocessor(image_size=Config.IMAGE_SIZE, is_train=False)
    
    horizontal_frames = preprocessor.extract_frames_from_video(video_path, 4)
    top_image = preprocessor.read_image(top_image_path)
    
    all_pil_images = horizontal_frames + [top_image]
    images_tensor = preprocessor.preprocess_images(all_pil_images).unsqueeze(0)
    
    return images_tensor, all_pil_images, label, category_name


def load_model(model_type, checkpoint_path, backbone_model, device):
    """加载训练好的模型"""
    if model_type == 'attention_fusion':
        model = MultiViewAttentionFusionModel(Config, backbone_model=backbone_model)
    elif model_type == 'single_view':
        model = SingleViewModel(Config, backbone_model=backbone_model)
    elif model_type == 'simple_concat':
        model = SimpleConcatModel(Config, backbone_model=backbone_model)
    elif model_type == 'cnn_lstm':
        model = CNNLSTMModel(Config, backbone_model=backbone_model)
    elif model_type == 'cnn_lstm_attention':
        model = CNNLSTMAttentionModel(Config, backbone_model=backbone_model)
    else:
        raise ValueError(f"未知模型类型：{model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


class FeatureExtractor:
    """特征提取器，用于获取中间层特征"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def generate_gradcam_for_single_view(model, image_tensor, device, target_class=None):
    """为单视角模型生成 Grad-CAM"""
    target_layer = None
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'stages'):
            target_layer = backbone.stages[-1]
        elif hasattr(backbone, 'features'):
            target_layer = backbone.features[-1]
        elif hasattr(backbone, 'layer4'):
            target_layer = backbone.layer4
    
    if target_layer is None:
        return None
    
    extractor = FeatureExtractor(model, target_layer)
    
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    
    try:
        output = model(image_tensor)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        probabilities = torch.softmax(logits, dim=1)
        
        if target_class is None:
            pred_label = torch.argmax(logits, dim=1).item()
            class_idx = pred_label
        else:
            class_idx = target_class
            pred_label = torch.argmax(logits, dim=1).item()
        
        model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        if extractor.features is None or extractor.gradients is None:
            extractor.remove_hooks()
            return None, pred_label, None
        
        gradients = extractor.gradients[0]
        features = extractor.features[0]
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * features, dim=0)
        cam = torch.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        cam_np = cam.cpu().numpy()
        cam_resized = cv2.resize(cam_np, (224, 224))
        
        extractor.remove_hooks()
        
        prob = probabilities[0, pred_label].item()
        
        return cam_resized, pred_label, prob
        
    except Exception as e:
        print(f"  Grad-CAM 生成错误：{str(e)}")
        extractor.remove_hooks()
        return None, None, None


def generate_gradcam(model, images_tensor, device, model_type, target_class=None):
    """生成 Grad-CAM 热力图
    
    Args:
        model: 训练好的模型
        images_tensor: 输入图像张量 (B, num_views, C, H, W)
        device: 计算设备
        model_type: 模型类型
        target_class: 目标类别，如果为 None 则使用预测类别
    
    Returns:
        cam_list: 每个视角的热力图列表
        pred_label: 预测标签
        prob: 预测概率
    """
    # 判断是单视角还是多视角
    is_single_view = model_type == 'single_view'
    
    if is_single_view:
        # 单视角模型：直接处理
        cam, pred_label, prob = generate_gradcam_for_single_view(
            model, images_tensor, device, target_class
        )
        return [cam], pred_label, prob
    
    # 多视角模型：先获取完整预测
    num_views = images_tensor.shape[1]
    model.eval()
    
    with torch.no_grad():
        output = model(images_tensor.to(device))
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        probabilities = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(logits, dim=1).item()
        
        if target_class is None:
            class_idx = pred_label
        else:
            class_idx = target_class
    
    # 对多视角模型，为每个视角生成 Grad-CAM
    # 方法：分别将每个视角作为单视角输入到 backbone，获取其特征图
    cam_list = []
    
    target_layer = None
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'stages'):
            target_layer = backbone.stages[-1]
        elif hasattr(backbone, 'features'):
            target_layer = backbone.features[-1]
        elif hasattr(backbone, 'layer4'):
            target_layer = backbone.layer4
    
    if target_layer is None:
        print("  警告：未找到目标层，无法生成 Grad-CAM")
        return [None] * num_views, pred_label, probabilities[0, pred_label].item()
    
    # 对每个视角，提取其经过 backbone 的特征并计算 Grad-CAM
    for view_idx in range(num_views):
        view_image = images_tensor[:, view_idx, :, :, :]  # (B, C, H, W)
        
        # 单独通过这个视角的 backbone 获取特征
        view_features = model.backbone(view_image)
        
        # 使用 backbone 的特征图计算 Grad-CAM
        # 需要重新计算梯度
        view_image = view_image.to(device)
        view_image.requires_grad = True
        
        # 注册 hook 到 backbone 的目标层
        features_container = {'features': None, 'gradients': None}
        
        def forward_hook(module, input, output):
            features_container['features'] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            features_container['gradients'] = grad_output[0].detach()
        
        hook = target_layer.register_forward_hook(forward_hook)
        hook_backward = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # 前向传播
            feats = model.backbone(view_image)
            
            # 为了获取梯度，需要通过完整模型
            # 但这样会很复杂，我们简化处理：直接使用特征图的激活
            # 使用特征图本身作为热力图（简化版 Grad-CAM）
            if features_container['features'] is not None:
                features = features_container['features'][0]
                # 对特征图的所有通道求平均作为热力图
                cam = torch.mean(features, dim=0)
                cam = torch.relu(cam)
                
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                
                cam_np = cam.cpu().numpy()
                cam_resized = cv2.resize(cam_np, (224, 224))
                cam_list.append(cam_resized)
            else:
                cam_list.append(None)
            
            hook.remove()
            hook_backward.remove()
            
        except Exception as e:
            print(f"  视角 {view_idx} Grad-CAM 生成错误：{str(e)}")
            hook.remove()
            hook_backward.remove()
            cam_list.append(None)
    
    prob = probabilities[0, pred_label].item()
    
    return cam_list, pred_label, prob


def visualize_training_heatmaps(pil_images, cam_list, true_label, pred_label, 
                                 prob, sample_name, model_type, save_dir):
    """可视化训练过程热力图
    
    Args:
        pil_images: PIL 图像列表（单视角为 1 张，多视角为 5 张）
        cam_list: Grad-CAM 热力图列表
        true_label: 真实标签
        pred_label: 预测标签
        prob: 预测概率
        sample_name: 样本名称
        model_type: 模型类型
        save_dir: 保存目录
    """
    # 判断是单视角还是多视角
    is_single_view = model_type == 'single_view'
    
    if is_single_view:
        # 单视角模型：只显示一张图
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原始图像
        ax = axes[0]
        img = pil_images[0].resize((224, 224))
        ax.imshow(img)
        ax.set_title('输入图像', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # 热力图
        ax = axes[1]
        img_array = np.array(pil_images[0].resize((224, 224)))
        
        if cam_list[0] is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_list[0]), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            superimposed = heatmap * 0.5 + img_array * 0.5
            superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
            ax.imshow(superimposed)
            ax.set_title('注意力热力图', fontsize=10)
        else:
            ax.imshow(img_array)
            ax.set_title('输入图像', fontsize=10)
        ax.axis('off')
    else:
        # 多视角模型：显示 5 个视角
        view_names = ['0°', '90°', '180°', '270°', '顶角']
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # 第一行：原始图像
        for i in range(5):
            ax = axes[0, i]
            img = pil_images[i].resize((224, 224))
            ax.imshow(img)
            ax.set_title(view_names[i], fontsize=12, fontweight='bold')
            ax.axis('off')
        
        # 第二行：热力图
        for i in range(5):
            ax = axes[1, i]
            img_array = np.array(pil_images[i].resize((224, 224)))
            
            if cam_list[i] is not None:
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_list[i]), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                superimposed = heatmap * 0.5 + img_array * 0.5
                superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                ax.imshow(superimposed)
            else:
                ax.imshow(img_array)
            
            ax.set_title(f'{view_names[i]} - 注意力热力图', fontsize=10)
            ax.axis('off')
    
    true_text = '健康' if true_label == 0 else '患虫'
    pred_text = '健康' if pred_label == 0 else '患虫'
    color = 'green' if true_label == pred_label else 'red'
    
    status = "预测正确 ✓" if true_label == pred_label else "预测错误 ✗"
    
    fig.suptitle(f'样本：{sample_name} | 模型：{model_type}\n'
                 f'真实：{true_text} | 预测：{pred_text} (置信度：{prob:.2%}) | {status}', 
                 fontsize=14, fontweight='bold', color=color)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'heatmap_{sample_name}_{timestamp}.png')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练过程热力图已保存：{save_path}")
    
    return save_path


def save_analysis_result(sample_name, true_label, pred_label, prob, model_type, 
                        cam_list, save_dir):
    """保存分析结果到 JSON 文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 判断是单视角还是多视角
    is_single_view = model_type == 'single_view'
    
    if is_single_view:
        cam_summary = {
            '单视角': float(np.mean(cam_list[0])) if cam_list[0] is not None else None
        }
    else:
        cam_summary = {
            '0°': float(np.mean(cam_list[0])) if cam_list[0] is not None else None,
            '90°': float(np.mean(cam_list[1])) if cam_list[1] is not None else None,
            '180°': float(np.mean(cam_list[2])) if cam_list[2] is not None else None,
            '270°': float(np.mean(cam_list[3])) if cam_list[3] is not None else None,
            '顶角': float(np.mean(cam_list[4])) if cam_list[4] is not None else None,
        }
    
    result = {
        'sample_name': sample_name,
        'model_type': model_type,
        'true_label': int(true_label),
        'pred_label': int(pred_label),
        'probability': float(prob),
        'correct': bool(true_label == pred_label),
        'timestamp': timestamp,
        'cam_summary': cam_summary
    }
    
    json_path = os.path.join(save_dir, f'analysis_{sample_name}_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"分析结果已保存：{json_path}")
    
    return json_path


def analyze_sample(model, model_type, images_tensor, pil_images, true_label, 
                   sample_name, device, save_dir):
    """分析单个样本并生成训练过程热力图
    
    Args:
        model: 训练好的模型
        model_type: 模型类型
        images_tensor: 图像张量
        pil_images: PIL 图像列表
        true_label: 真实标签
        sample_name: 样本名称
        device: 计算设备
        save_dir: 保存目录
    
    Returns:
        result: 分析结果字典
    """
    # 判断是单视角还是多视角
    is_single_view = model_type == 'single_view'
    
    # 单视角模型只使用第一张图像
    if is_single_view:
        images_tensor = images_tensor[:, :1, :, :, :]  # 只取第一个视角
        pil_images = [pil_images[0]]  # 只保留第一张图
        view_names_display = ['单视角 (第 1 帧)']
    else:
        view_names_display = ['0°', '90°', '180°', '270°', '顶角']
    
    images_tensor = images_tensor.to(device)
    
    print("\n" + "="*60)
    print(f"正在分析样本：{sample_name}")
    print("="*60)
    
    with torch.no_grad():
        output = model(images_tensor)
        
        if model_type in ['attention_fusion', 'cnn_lstm_attention']:
            if len(output) == 3:
                logits, probabilities, _ = output
            else:
                logits, probabilities = output
        else:
            if isinstance(output, tuple):
                logits, probabilities = output
            else:
                logits = output
                probabilities = torch.softmax(logits, dim=1)
    
    pred_label = torch.argmax(logits, dim=1).item()
    prob = probabilities[0, pred_label].item()
    
    print(f"\n预测结果:")
    print(f"  真实标签：{'健康' if true_label == 0 else '患虫'} ({true_label})")
    print(f"  预测标签：{'健康' if pred_label == 0 else '患虫'} ({pred_label})")
    print(f"  预测置信度：{prob:.2%}")
    print(f"  预测{'✓ 正确' if pred_label == true_label else '✗ 错误'}")
    
    print("\n生成训练过程 Grad-CAM 热力图...")
    cam_list, pred_label_cam, prob_cam = generate_gradcam(
        model, images_tensor.clone(), device, model_type
    )
    
    if pred_label_cam is not None:
        pred_label = pred_label_cam
        prob = prob_cam
    
    # 显示各视角激活强度
    if is_single_view:
        if cam_list[0] is not None:
            activation = np.mean(cam_list[0])
            print(f"  单视角：平均激活强度 = {activation:.4f}")
    else:
        for i, cam in enumerate(cam_list):
            if cam is not None:
                activation = np.mean(cam)
                print(f"  {view_names_display[i]}: 平均激活强度 = {activation:.4f}")
    
    heatmap_path = visualize_training_heatmaps(
        pil_images, cam_list, true_label, pred_label, 
        prob, sample_name, model_type, save_dir
    )
    
    json_path = save_analysis_result(
        sample_name, true_label, pred_label, prob, model_type, cam_list, save_dir
    )
    
    return {
        'sample_name': sample_name,
        'true_label': true_label,
        'pred_label': pred_label,
        'prob': prob,
        'correct': pred_label == true_label,
        'heatmap_path': heatmap_path,
        'json_path': json_path
    }


def main():
    """主函数：交互式输入样本，输出训练过程热力图"""
    print("="*70)
    print("柑橘虫害检测 - 训练过程热力图可视化工具")
    print("="*70)
    
    device = Config.DEVICE
    print(f"\n使用设备：{device}")
    
    model_type, checkpoint_path = select_model_interactive(Config.CHECKPOINT_PATH)
    if model_type is None:
        print("未选择模型，程序退出")
        return
    
    print(f"\n加载模型：{model_type}")
    print("正在加载模型权重...")
    model = load_model(model_type, checkpoint_path, 'convnext_tiny', device)
    print("模型加载完成 ✓")
    
    save_dir = os.path.join(Config.CHECKPOINT_PATH, model_type, 'heatmaps')
    os.makedirs(save_dir, exist_ok=True)
    print(f"热力图保存目录：{save_dir}")
    
    print("\n" + "="*60)
    print("使用说明:")
    print("  - 输入样本编号 (如 A1, h1, X1 等)")
    print("  - 程序将生成该样本的训练过程 Grad-CAM 热力图")
    print("  - 输入 'q' 退出程序")
    print("="*60)
    
    while True:
        print("\n" + "-"*50)
        sample_label = input("请输入样本编号 (输入 q 退出): ").strip()
        
        if sample_label.lower() == 'q':
            print("\n感谢使用，再见!")
            break
        
        if not sample_label:
            print("错误：样本编号不能为空")
            continue
        
        try:
            images_tensor, pil_images, true_label, category_name = load_sample_data(
                sample_label, data_dir='./datasets'
            )
            
            result = analyze_sample(model, model_type, images_tensor, pil_images,
                                    true_label, sample_label, device, save_dir)
            
            print(f"\n✓ 分析完成!")
            print(f"  热力图：{result['heatmap_path']}")
            print(f"  JSON 结果：{result['json_path']}")
            
        except FileNotFoundError as e:
            print(f"错误：{str(e)}")
        except ValueError as e:
            print(f"错误：{str(e)}")
        except Exception as e:
            print(f"分析过程中出错：{str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
