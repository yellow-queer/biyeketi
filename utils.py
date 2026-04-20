"""
工具函数模块
功能：包含数据集类、评估指标计算、模型保存加载、可视化、训练评估等工具函数
日期：2026年
"""
import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from data_preprocess import ImagePreprocessor


def set_seed(seed: int):
    """
    设置全局随机种子，保证可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CitrusPestDataset(Dataset):
    """
    柑橘虫害数据集类
    支持读取多视角图像数据和标签
    """
    
    def __init__(self, data_list: List[Dict], image_preprocessor: ImagePreprocessor, is_train: bool = True):
        """
        参数:
            data_list: 数据列表，每个元素包含 'images', 'label'
            image_preprocessor: 图像预处理器
            is_train: 是否为训练集
        """
        self.data_list = data_list
        self.image_preprocessor = image_preprocessor
        self.image_preprocessor.is_train = is_train
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_list[idx]
        
        images_tensor = self.image_preprocessor.preprocess_images(item['images'])
        
        label = torch.LongTensor([item['label']])
        
        return {
            'images': images_tensor,
            'label': label.squeeze(0)
        }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评价指标：准确率、精确率、召回率、F1 值
    """
    accuracy = accuracy_score(y_true, y_pred)
    # 使用 zero_division=0 参数避免警告，当没有预测样本时返回 0.0
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                    optimizer: optim.Optimizer, device: torch.device, logger) -> Tuple[float, Dict[str, float]]:
    """
    训练一个epoch
    
    参数:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        logger: 日志记录器
    
    返回:
        avg_loss: 平均损失
        metrics: 评估指标字典
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        images = batch['images'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        if len(outputs) == 3:
            logits, probabilities, _ = outputs
        else:
            logits, probabilities = outputs
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    logger.info(f"训练 - Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}")
    
    return avg_loss, metrics


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device, logger, dataset_name: str = "验证集") -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """
    评估模型
    
    参数:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        logger: 日志记录器
        dataset_name: 数据集名称
    
    返回:
        avg_loss: 平均损失
        metrics: 评估指标字典
        all_labels: 所有真实标签
        all_preds: 所有预测标签
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Evaluating {dataset_name}')
        for batch in progress_bar:
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            if len(outputs) == 3:
                logits, probabilities, _ = outputs
            else:
                logits, probabilities = outputs
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    
    logger.info(f"{dataset_name} - Loss: {avg_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}")
    
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


def plot_training_curves(train_losses: List[float], train_accs: List[float], 
                         val_losses: List[float], val_accs: List[float], 
                         save_path: str = './checkpoints/training_curves.png'):
    """
    绘制训练曲线并保存
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(val_accs, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线已保存至 {save_path}")


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, metrics: Dict[str, float], save_path: str, 
                    scheduler=None):
    """
    保存模型检查点
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, save_path)
    print(f"模型检查点已保存至 {save_path}")


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, 
                    load_path: str = './checkpoints/best_model.pth', device: str = 'cpu'):
    """
    加载模型检查点
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"模型检查点文件不存在: {load_path}")
        
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    additional_info = checkpoint.copy()
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    print(f"模型检查点已从 {load_path} 加载")
    return model, optimizer, checkpoint.get('epoch', 0), additional_info


def setup_logger(log_file: str = './checkpoints/training.log'):
    """
    配置日志记录器
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('CitrusPestDetection')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def generate_dummy_images(num_samples: int = 5) -> List[Image.Image]:
    """
    生成模拟图像数据（当没有真实图像时使用）
    """
    dummy_images = []
    for _ in range(num_samples):
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        dummy_images.append(Image.fromarray(img_array))
    return dummy_images


def prepare_data_list_for_multiview(num_samples: int = 100) -> List[Dict]:
    """
    为多视角模型准备模拟数据
    """
    data_list = []
    for i in range(num_samples):
        label = 0 if i < num_samples // 2 else 1
        data_list.append({
            'images': generate_dummy_images(5),
            'label': label
        })
    return data_list


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str] = ['健康', '患虫'],
                           save_path: str = './checkpoints/confusion_matrix.png'):
    """
    绘制混淆矩阵
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至 {save_path}")
    return cm


def save_evaluation_report(metrics: Dict[str, float], confusion_mat: np.ndarray, 
                            config, model_type: str = 'attention_fusion',
                            save_path: str = './checkpoints/evaluation_report.txt'):
    """
    保存评估报告
    
    参数:
        metrics: 评估指标字典
        confusion_mat: 混淆矩阵
        config: 配置对象
        model_type: 模型类型 ('single_view', 'simple_concat', 'attention_fusion')
        save_path: 保存路径
    """
    model_names = {
        'single_view': '单视角模型',
        'simple_concat': '多视角简单拼接模型',
        'attention_fusion': '多视角注意力融合模型'
    }
    
    fusion_methods = {
        'single_view': '仅使用单张水平视角图像',
        'simple_concat': '五张视角图像特征直接拼接',
        'attention_fusion': '五张视角图像特征通过多头注意力机制自适应融合'
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{model_names.get(model_type, '柑橘虫害检测模型')}评估报告\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【模型与融合方法】\n")
        f.write(f"模型类型: {model_names.get(model_type, model_type)}\n")
        f.write(f"融合方法: {fusion_methods.get(model_type, model_type)}\n\n")
        
        f.write("【配置参数】\n")
        f.write(f"骨干网络 (BACKBONE_MODEL): {config.BACKBONE_MODEL}\n")
        f.write(f"图像特征维度 (IMAGE_FEATURE_DIM): {getattr(config, 'IMAGE_FEATURE_DIM', 'N/A')}\n")
        f.write(f"图像尺寸 (IMAGE_SIZE): {config.IMAGE_SIZE}\n")
        f.write(f"注意力头数 (NUM_ATTENTION_HEADS): {getattr(config, 'NUM_ATTENTION_HEADS', 'N/A')}\n")
        f.write(f"融合层隐藏维度 (FUSION_HIDDEN_DIM): {getattr(config, 'FUSION_HIDDEN_DIM', 'N/A')}\n")
        f.write(f"随机种子 (SEED): {config.SEED}\n")
        f.write(f"是否冻结骨干网络 (FREEZE_CONVNEXT): {config.FREEZE_CONVNEXT}\n")
        f.write(f"Dropout率 (DROPOUT_RATE): {config.DROPOUT_RATE}\n\n")
        
        f.write("【评估指标】\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
        f.write(f"F1 值 (F1-Score): {metrics['f1']:.4f}\n\n")
        
        f.write("【混淆矩阵】\n")
        f.write(str(confusion_mat))
        f.write("\n")
        
    print(f"评估报告已保存至 {save_path}")


def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                          save_path: str = './checkpoints/model_comparison.png'):
    """
    绘制多个模型的性能对比图
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_names = ['准确率', '精确率', '召回率', 'F1值']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results[model][metric] for model in model_names]
        ax.bar(x + i * width, values, width, label=metric_name)
    
    ax.set_xlabel('模型')
    ax.set_ylabel('分数')
    ax.set_title('不同模型性能对比')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"模型对比图已保存至 {save_path}")


def natural_sort_key(s: str) -> tuple:
    """
    自然排序键函数，用于正确排序包含数字的文件名
    将字符串中的数字部分转换为整数，确保 A2 在 A10 前面
    """
    import re
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def load_dataset_from_videos(data_dir: str, image_preprocessor: ImagePreprocessor) -> List[Dict]:
    """
    从 datasets 文件夹加载视频和顶角图像数据
    
    数据集目录结构：
    datasets/
    ├── pic/
    │   ├── pic_healthy/      # 健康柑橘顶角照片 (h1.jpg, h2.jpg...)
    │   └── pic_chongju/      # 患虫柑橘顶角照片 (A1.jpg, A2.jpg...)
    └── video/
        ├── v_healthy/        # 健康柑橘环绕视频 (h1.mp4, h2.mp4...)
        └── v_chongju/        # 患虫柑橘环绕视频 (A1.mp4, A2.mp4...)
    
    参数:
        data_dir: 数据集目录路径
        image_preprocessor: 图像预处理器
    
    返回:
        数据列表，每个元素包含 'images' (5张PIL图像) 和 'label'
    """
    data_list = []
    
    pic_dir = os.path.join(data_dir, 'pic')
    video_dir = os.path.join(data_dir, 'video')
    
    if not os.path.exists(pic_dir) or not os.path.exists(video_dir):
        print(f"警告: 数据集目录结构不正确，将使用模拟数据")
        return prepare_data_list_for_multiview()
    
    categories = [
        ('pic_healthy', 'v_healthy', 0, '健康'),
        ('pic_chongju', 'v_chongju', 1, '患虫')
    ]
    
    for pic_subdir, video_subdir, label, category_name in categories:
        pic_path = os.path.join(pic_dir, pic_subdir)
        video_path = os.path.join(video_dir, video_subdir)
        
        if not os.path.exists(pic_path) or not os.path.exists(video_path):
            print(f"警告: 类别目录 {category_name} 不完整，跳过")
            continue
        
        # 对文件列表进行自然排序，确保可复现性（A2 在 A10 前面）
        pic_files = sorted([f for f in os.listdir(pic_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))], 
                          key=natural_sort_key)
        video_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))], 
                            key=natural_sort_key)
        
        pic_names = {os.path.splitext(f)[0].lower() for f in pic_files}
        video_names = {os.path.splitext(f)[0].lower() for f in video_files}
        
        common_names = pic_names.intersection(video_names)
        print(f"类别 {category_name}: 找到 {len(common_names)} 个匹配样本")
        
        # 对样本名称进行自然排序，确保遍历顺序一致（A2 在 A10 前面）
        for sample_name in sorted(common_names, key=natural_sort_key):
            try:
                pic_file = next(f for f in pic_files if os.path.splitext(f)[0].lower() == sample_name)
                video_file = next(f for f in video_files if os.path.splitext(f)[0].lower() == sample_name)
                
                top_image_path = os.path.join(pic_path, pic_file)
                current_video_path = os.path.join(video_path, video_file)
                
                # 使用 image_preprocessor 的 preprocess 方法进行完整预处理
                # 该方法会自动从视频提取 4 帧并加上顶角图像
                all_images = [
                    image_preprocessor.read_image(top_image_path),  # 顶角图像
                    current_video_path  # 视频路径，会在 preprocess 中提取帧
                ]
                
                # 读取顶角图像
                top_image = image_preprocessor.read_image(top_image_path)
                # 从视频提取 4 帧
                horizontal_frames = image_preprocessor.extract_frames_from_video(current_video_path, 4)
                # 组合：4 张水平 + 1 张顶角
                all_images = horizontal_frames + [top_image]
                
                data_list.append({
                    'images': all_images,
                    'label': label,
                    'sample_name': f"{category_name}/{sample_name}"
                })
                
            except Exception as e:
                print(f"加载样本 {sample_name} 失败：{str(e)}")
                continue
    
    if len(data_list) == 0:
        print("未找到有效的视频数据，将使用模拟数据")
        return prepare_data_list_for_multiview()
    
    print(f"共加载 {len(data_list)} 个样本")
    return data_list
