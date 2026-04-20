"""
模型可视化分析工具
功能：加载训练好的模型，生成视角权重分布柱状图和Grad-CAM热力图
日期：2026年
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from config import Config
from data_preprocess import ImagePreprocessor
from models import MultiViewAttentionFusionModel, SingleViewModel, SimpleConcatModel, CNNLSTMModel, CNNLSTMAttentionModel


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GradCAM:
    """Grad-CAM实现类"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        cam = torch.sum(weights * activations, dim=0)
        cam = torch.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def get_sample_paths(sample_label: str, data_dir: str = './datasets'):
    """
    根据样本标签获取对应的图像和视频路径
    
    参数:
        sample_label: 样本标签，如 'A1', 'h1', 'X1' 等
        data_dir: 数据集目录
    
    返回:
        (top_image_path, video_path, label, category_name)
    """
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
        sample_label = sample_label.upper().replace('H', 'h')
    else:
        raise ValueError(f"无法识别的样本标签格式: {sample_label}")
    
    pic_dir = os.path.join(data_dir, 'pic', pic_subdir)
    video_dir = os.path.join(data_dir, 'video', video_subdir)
    
    possible_names = [
        sample_label,
        sample_label.upper(),
        sample_label.lower(),
        sample_label.replace('H', 'h') if sample_label.startswith('H') else sample_label
    ]
    
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
        raise FileNotFoundError(f"找不到顶角图像: {sample_label} in {pic_dir}")
    if not video_path:
        raise FileNotFoundError(f"找不到视频文件: {sample_label} in {video_dir}")
    
    return top_image_path, video_path, label, category_name


def load_sample_data(sample_label: str, data_dir: str = './datasets'):
    """
    加载指定样本的多视角图像数据
    
    返回:
        images_tensor: (1, 5, 3, 224, 224) 的张量
        pil_images: 5张PIL图像列表
        label: 标签
        category_name: 类别名称
    """
    top_image_path, video_path, label, category_name = get_sample_paths(sample_label, data_dir)
    
    print(f"加载样本: {sample_label}")
    print(f"  顶角图像: {top_image_path}")
    print(f"  环绕视频: {video_path}")
    print(f"  类别: {category_name} (标签={label})")
    
    preprocessor = ImagePreprocessor(image_size=Config.IMAGE_SIZE, is_train=False)
    
    horizontal_frames = preprocessor.extract_frames_from_video(video_path, 4)
    top_image = preprocessor.read_image(top_image_path)
    
    all_pil_images = horizontal_frames + [top_image]
    
    images_tensor = preprocessor.preprocess_images(all_pil_images).unsqueeze(0)
    
    return images_tensor, all_pil_images, label, category_name


def load_model(model_type: str, checkpoint_path: str, device: torch.device):
    """
    加载指定类型的模型
    
    参数:
        model_type: 模型类型 ('attention_fusion', 'single_view', 'simple_concat', 'cnn_lstm', 'cnn_lstm_attention')
        checkpoint_path: 模型检查点路径
        device: 设备
    
    返回:
        加载好的模型
    """
    if model_type == 'attention_fusion':
        model = MultiViewAttentionFusionModel(Config)
    elif model_type == 'single_view':
        model = SingleViewModel(Config)
    elif model_type == 'simple_concat':
        model = SimpleConcatModel(Config)
    elif model_type == 'cnn_lstm':
        model = CNNLSTMModel(Config)
    elif model_type == 'cnn_lstm_attention':
        model = CNNLSTMAttentionModel(Config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def plot_attention_weights(attn_weights: np.ndarray, save_path: str, sample_label: str):
    """
    绘制视角权重分布柱状图
    
    参数:
        attn_weights: 注意力权重，形状为 (num_heads, 5, 5) 或 (5, 5)
        save_path: 保存路径
        sample_label: 样本标签
    """
    if len(attn_weights.shape) == 3:
        attn_weights = np.mean(attn_weights, axis=0)
    
    view_labels = ['0°', '90°', '180°', '270°', '顶角']
    
    avg_weights = np.mean(attn_weights, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    bars = ax.bar(view_labels, avg_weights, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, weight in zip(bars, avg_weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{weight:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('视角', fontsize=12)
    ax.set_ylabel('平均注意力权重', fontsize=12)
    ax.set_title(f'多视角注意力权重分布 - 样本 {sample_label}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(avg_weights) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"视角权重分布图已保存至: {save_path}")


def generate_gradcam_for_view(model, input_tensor, view_idx, device, target_layer=None):
    """
    为单个视角生成Grad-CAM热力图
    
    参数:
        model: 模型
        input_tensor: 输入张量 (1, 5, 3, 224, 224)
        view_idx: 视角索引 (0-4)
        device: 设备
        target_layer: 目标层
    
    返回:
        cam: Grad-CAM热力图
    """
    if target_layer is None:
        if hasattr(model, 'backbone'):
            target_layer = model.backbone
        else:
            raise ValueError("无法找到目标层")
    
    grad_cam = GradCAM(model, target_layer)
    
    def forward_with_view(module, input, output):
        view_input = input[0][:, view_idx, :, :, :]
        return module(view_input)
    
    single_view_input = input_tensor[:, view_idx:view_idx+1, :, :, :].squeeze(1)
    
    cam = grad_cam.generate_cam(single_view_input)
    grad_cam.remove_hooks()
    
    return cam


def apply_cam_on_image(img: Image.Image, cam: np.ndarray, alpha: float = 0.4):
    """
    将Grad-CAM热力图叠加到原始图像上
    
    参数:
        img: PIL图像
        cam: Grad-CAM热力图
        alpha: 透明度
    
    返回:
        叠加后的图像
    """
    img_array = np.array(img.resize((224, 224)))
    
    cam_resized = cv2.resize(cam, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed = heatmap * alpha + img_array * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed)


def plot_gradcam_grid(pil_images, cam_list, save_path, sample_label, view_labels=None):
    """
    绘制Grad-CAM热力图网格
    
    参数:
        pil_images: 5张PIL图像列表
        cam_list: 5个Grad-CAM热力图列表
        save_path: 保存路径
        sample_label: 样本标签
        view_labels: 视角标签列表
    """
    if view_labels is None:
        view_labels = ['0°', '90°', '180°', '270°', '顶角']
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, (img, cam, view_label) in enumerate(zip(pil_images, cam_list, view_labels)):
        axes[0, i].imshow(img.resize((224, 224)))
        axes[0, i].set_title(f'{view_label}\n原始图像', fontsize=11)
        axes[0, i].axis('off')
        
        superimposed = apply_cam_on_image(img, cam)
        axes[1, i].imshow(superimposed)
        axes[1, i].set_title(f'{view_label}\nGrad-CAM', fontsize=11)
        axes[1, i].axis('off')
    
    fig.suptitle(f'Grad-CAM 空间热力图分析 - 样本 {sample_label}', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM热力图已保存至: {save_path}")


def analyze_sample_with_model(model_type: str, sample_label: str, data_dir: str = './datasets', 
                               checkpoint_dir: str = './checkpoints'):
    """
    使用指定模型分析样本
    
    参数:
        model_type: 模型类型
        sample_label: 样本标签
        data_dir: 数据集目录
        checkpoint_dir: 检查点目录
    """
    device = Config.DEVICE
    print(f"\n{'='*60}")
    print(f"模型类型: {model_type}")
    print(f"使用设备: {device}")
    print('='*60)
    
    checkpoint_path = os.path.join(checkpoint_dir, model_type, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"警告: 找不到模型检查点 {checkpoint_path}")
        return None
    
    images_tensor, pil_images, true_label, category_name = load_sample_data(sample_label, data_dir)
    images_tensor = images_tensor.to(device)
    
    model = load_model(model_type, checkpoint_path, device)
    
    with torch.no_grad():
        output = model(images_tensor)
        
        if isinstance(output, tuple):
            if len(output) == 3:
                logits, probabilities, attn_weights = output
            else:
                logits, probabilities = output
                attn_weights = None
        else:
            logits = output
            probabilities = torch.softmax(logits, dim=1)
            attn_weights = None
    
    pred_class = torch.argmax(probabilities, dim=1).item()
    pred_prob = probabilities[0, pred_class].item()
    
    class_names = ['健康', '患虫']
    
    print(f"\n预测结果:")
    print(f"  真实标签: {category_name} ({true_label})")
    print(f"  预测标签: {class_names[pred_class]} ({pred_class})")
    print(f"  预测概率: {pred_prob:.4f}")
    print(f"  预测正确: {'✓' if pred_class == true_label else '✗'}")
    
    output_dir = os.path.join(checkpoint_dir, model_type, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    if attn_weights is not None:
        attn_weights_np = attn_weights.cpu().numpy()[0]
        attention_save_path = os.path.join(output_dir, f'attention_weights_{sample_label}.png')
        plot_attention_weights(attn_weights_np, attention_save_path, sample_label)
    
    try:
        cam_list = []
        for view_idx in range(5):
            cam = generate_gradcam_for_view(model, images_tensor, view_idx, device)
            cam_list.append(cam)
        
        gradcam_save_path = os.path.join(output_dir, f'gradcam_{sample_label}.png')
        plot_gradcam_grid(pil_images, cam_list, gradcam_save_path, sample_label)
    except Exception as e:
        print(f"生成Grad-CAM时出错: {str(e)}")
    
    return {
        'model_type': model_type,
        'sample_label': sample_label,
        'true_label': true_label,
        'pred_class': pred_class,
        'pred_prob': pred_prob,
        'correct': pred_class == true_label
    }


def main():
    print("="*70)
    print("柑橘虫害检测模型可视化分析工具")
    print("="*70)
    
    sample_label = input("\n请输入样本标签 (例如 A1, h1, X1 等): ").strip()
    
    if not sample_label:
        print("错误: 样本标签不能为空")
        return
    
    data_dir = './datasets'
    checkpoint_dir = './checkpoints'
    
    model_types = ['attention_fusion', 'single_view', 'simple_concat', 'cnn_lstm', 'cnn_lstm_attention']
    
    results = []
    for model_type in model_types:
        try:
            result = analyze_sample_with_model(model_type, sample_label, data_dir, checkpoint_dir)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n分析模型 {model_type} 时出错: {str(e)}")
            continue
    
    print("\n" + "="*70)
    print("分析结果汇总")
    print("="*70)
    
    class_names = ['健康', '患虫']
    for result in results:
        status = "✓ 正确" if result['correct'] else "✗ 错误"
        print(f"{result['model_type']:25s} | 预测: {class_names[result['pred_class']]:4s} | "
              f"概率: {result['pred_prob']:.4f} | {status}")
    
    print("\n" + "="*70)
    print("可视化结果保存位置:")
    for model_type in model_types:
        output_dir = os.path.join(checkpoint_dir, model_type, 'analysis')
        if os.path.exists(output_dir):
            print(f"  - {model_type}: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
