"""
Grad-CAM 可视化工具
功能：实现梯度加权类激活映射，用于可视化模型关注的图像区域
日期：2026 年
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GradCAM:
    """
    Grad-CAM 实现类
    用于生成类激活热力图，可视化模型决策依据
    """
    
    def __init__(self, model, target_layer, device='cuda'):
        """
        参数:
            model: 训练好的模型
            target_layer: 目标特征提取层（通常是最后一个卷积层）
            device: 计算设备
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和后向钩子"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        # 使用 register_full_backward_hook 替代已弃用的 register_backward_hook
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 热力图
        
        参数:
            input_tensor: 输入图像张量 (B, C, H, W)
            target_class: 目标类别索引，None 则使用预测类别
        
        返回:
            cam: 类激活图 (H, W)
            prediction: 模型预测结果
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # 前向传播
        outputs = self.model(input_tensor)
        if len(outputs) == 3:
            logits, probabilities, _ = outputs
        else:
            logits, probabilities = outputs
        
        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # 获取目标类别的 logits
        target_logits = logits[0, target_class]
        
        # 反向传播
        self.model.zero_grad()
        target_logits.backward(retain_graph=True)
        
        # 获取梯度和激活值
        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()
        
        # 计算权重
        pooled_gradients = np.mean(gradients, axis=(2, 3), keepdims=True)
        
        # 加权求和
        cam = np.sum(activations * pooled_gradients, axis=1)[0]
        
        # ReLU 和非线性归一化
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        
        return cam, probabilities.cpu().detach().numpy()
    
    def generate_cam_for_multiview(self, input_tensor, target_class=None, view_idx=0):
        """
        为多视角模型生成 Grad-CAM
        
        参数:
            input_tensor: 多视角输入张量 (B, 5, C, H, W)
            target_class: 目标类别
            view_idx: 要可视化的视角索引 (0-4)
        
        返回:
            cam: 类激活图
            prediction: 预测结果
            view_image: 对应视角的原始图像
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        # 提取指定视角的图像
        batch_size = input_tensor.shape[0]
        view_tensor = input_tensor[:, view_idx, :, :, :]
        
        # 前向传播
        outputs = self.model(input_tensor)
        if len(outputs) == 3:
            logits, probabilities, attn_weights = outputs
        else:
            logits, probabilities = outputs
        
        # 确定目标类别
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # 获取目标类别的 logits
        target_logits = logits[0, target_class]
        
        # 反向传播
        self.model.zero_grad()
        target_logits.backward(retain_graph=True)
        
        # 获取梯度和激活值
        if self.gradients is None or self.activations is None:
            raise RuntimeError("未能获取梯度或激活值，请检查目标层是否正确注册")
        
        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()
        
        # 计算权重
        pooled_gradients = np.mean(gradients, axis=(2, 3), keepdims=True)
        
        # 加权求和
        cam = np.sum(activations * pooled_gradients, axis=1)[0]
        
        # ReLU 和非线性归一化
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        
        # 获取原始图像
        view_image = view_tensor[0].cpu().detach().numpy()
        view_image = np.transpose(view_image, (1, 2, 0))
        view_image = (view_image - view_image.min()) / (view_image.max() - view_image.min())
        
        return cam, probabilities.cpu().detach().numpy(), view_image


def apply_cam_to_image(image, cam, alpha=0.5, colormap='jet'):
    """
    将 CAM 热力图叠加到原始图像上
    
    参数:
        image: 原始图像 (numpy array, HxWx3, 0-1 范围)
        cam: CAM 热力图 (HxW, 0-1 范围)
        alpha: 叠加透明度
        colormap: 颜色映射
    
    返回:
        overlay: 叠加后的图像
    """
    # 调整 CAM 大小以匹配图像
    if image.shape[:2] != cam.shape:
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    else:
        cam_resized = cam
    
    # 应用颜色映射
    cmap = plt.get_cmap(colormap)
    cam_colored = (cmap(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    
    # 转换为 BGR (OpenCV 格式)
    cam_colored_bgr = cv2.cvtColor(cam_colored, cv2.COLOR_RGB2BGR)
    
    # 图像转换为 0-255 范围
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)
    
    if image_uint8.shape[2] == 1:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    elif image_uint8.shape[2] == 4:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_BGRA2BGR)
    elif image_uint8.shape[2] == 3:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    
    # 加权叠加
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, cam_colored_bgr, alpha, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    return overlay


def save_grad_cam_visualization(image, cam, save_path, title='Grad-CAM Visualization', 
                                class_name='Predicted Class', confidence=None):
    """
    保存 Grad-CAM 可视化结果
    
    参数:
        image: 原始图像
        cam: CAM 热力图
        save_path: 保存路径
        title: 图像标题
        class_name: 类别名称
        confidence: 置信度
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    if image.max() <= 1.0:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # CAM 热力图
    im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 叠加图像
    overlay = apply_cam_to_image(image, cam)
    axes[2].imshow(overlay)
    title_text = f'{title}\n{class_name}'
    if confidence is not None:
        title_text += f' ({confidence:.2%})'
    axes[2].set_title(title_text)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM 可视化已保存至 {save_path}")


def get_backbone_last_layer(model, backbone_name='convnext_tiny'):
    """
    获取骨干网络的最后一层卷积层
    
    参数:
        model: 模型对象
        backbone_name: 骨干网络名称
    
    返回:
        target_layer: 目标卷积层
    """
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        
        # 通用方法：获取最后一个卷积层
        # 遍历所有模块，找到最后一个 Conv2d 层
        conv_layers = []
        for name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if conv_layers:
            # 返回最后一个卷积层
            last_layer_name, target_layer = conv_layers[-1]
            print(f"Grad-CAM 目标层：{last_layer_name} ({type(target_layer).__name__})")
            return target_layer
        else:
            raise RuntimeError("在 backbone 中未找到 Conv2d 层")
    else:
        raise RuntimeError("模型没有 backbone 属性")
