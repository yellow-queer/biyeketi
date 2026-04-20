"""
骨干网络对比实验程序
功能：同时训练ConvNeXt-Tiny和ResNet-18版本的注意力融合模型，进行性能对比
作者：柑橘虫害检测项目组
日期：2026年
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse

from config import Config
from data_preprocess import ImagePreprocessor
from models import MultiViewAttentionFusionModel
from utils import (
    set_seed, CitrusPestDataset, calculate_metrics,
    save_checkpoint, load_checkpoint, setup_logger,
    load_dataset_from_videos, plot_model_comparison, train_one_epoch, evaluate
)


def train_and_evaluate_model(model_name: str, model: nn.Module, 
                              train_loader: DataLoader, val_loader: DataLoader, 
                              test_loader: DataLoader, criterion: nn.Module, 
                              optimizer: optim.Optimizer, device: torch.device, 
                              logger, save_dir: str, num_epochs: int):
    """
    训练和评估单个模型
    """
    os.makedirs(save_dir, exist_ok=True)
    best_f1 = 0.0
    best_metrics = None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练模型: {model_name}")
    logger.info(f"{'='*60}")
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 50)
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger
        )
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, device, logger, "验证集"
        )
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics,
                os.path.join(save_dir, f'{model_name}_best.pth')
            )
            logger.info(f"{model_name} 最佳模型已保存 (F1: {best_f1:.4f})")
    
    logger.info(f"\n{model_name} 训练完成!")
    logger.info(f"最佳验证 F1: {best_f1:.4f}")
    
    logger.info(f"\n在测试集上评估 {model_name}...")
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
    model, _, _, _ = load_checkpoint(model, load_path=best_model_path, device=device)
    
    test_loss, test_metrics, _, _ = evaluate(
        model, test_loader, criterion, device, logger, "测试集"
    )
    
    logger.info(f"\n{model_name} 测试集结果:")
    logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"  精确率: {test_metrics['precision']:.4f}")
    logger.info(f"  召回率: {test_metrics['recall']:.4f}")
    logger.info(f"  F1值: {test_metrics['f1']:.4f}")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='柑橘虫害骨干网络对比实验 (ConvNeXt-Tiny vs ResNet-18)')
    parser.add_argument('--data_dir', type=str, default='./datasets', 
                        help='数据集目录路径')
    parser.add_argument('--epochs', type=int, default=30,
                        help='每个模型的训练轮次')
    args = parser.parse_args()
    
    Config.create_directories()
    set_seed(Config.SEED)
    
    logger = setup_logger(os.path.join(Config.CHECKPOINT_PATH, 'backbone_comparison.log'))
    logger.info("=" * 60)
    logger.info("柑橘虫害骨干网络对比实验 (ConvNeXt-Tiny vs ResNet-18)")
    logger.info("=" * 60)
    
    device = Config.DEVICE
    logger.info(f"使用设备: {device}")
    
    logger.info("初始化预处理器...")
    image_preprocessor = ImagePreprocessor(image_size=Config.IMAGE_SIZE, is_train=True)
    
    logger.info("加载数据集...")
    data_list = load_dataset_from_videos(args.data_dir, image_preprocessor)
    
    if len(data_list) == 0:
        logger.error("未找到任何数据！")
        return
        
    logger.info(f"共加载 {len(data_list)} 个样本")
    
    healthy_count = sum(1 for item in data_list if item['label'] == 0)
    pest_count = sum(1 for item in data_list if item['label'] == 1)
    logger.info(f"样本分布: 健康={healthy_count}, 患虫={pest_count}")
    
    train_size = int(0.7 * len(data_list))
    val_size = int(0.15 * len(data_list))
    test_size = len(data_list) - train_size - val_size
    
    train_data, val_data, test_data = random_split(data_list, [train_size, val_size, test_size])
    logger.info(f"数据集划分: 训练集={len(train_data)}, 验证集={len(val_data)}, 测试集={len(test_data)}")
    
    train_dataset = CitrusPestDataset(train_data, image_preprocessor, is_train=True)
    val_dataset = CitrusPestDataset(val_data, image_preprocessor, is_train=False)
    test_dataset = CitrusPestDataset(test_data, image_preprocessor, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    backbones = ['convnext_tiny', 'resnet18']
    results = {}
    save_dir = os.path.join(Config.CHECKPOINT_PATH, 'backbone_comparison')
    
    for backbone in backbones:
        model_name = f'注意力融合_{backbone}'
        model = MultiViewAttentionFusionModel(Config, backbone_model=backbone).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        test_metrics = train_and_evaluate_model(
            model_name, model, train_loader, val_loader, test_loader,
            criterion, optimizer, device, logger, save_dir, args.epochs
        )
        
        results[model_name] = test_metrics
    
    logger.info("\n" + "=" * 60)
    logger.info("所有骨干网络对比实验完成!")
    logger.info("=" * 60)
    
    logger.info("\n骨干网络性能汇总:")
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision']:.4f}")
        logger.info(f"  召回率: {metrics['recall']:.4f}")
        logger.info(f"  F1值: {metrics['f1']:.4f}")
    
    logger.info("\n生成骨干网络对比图...")
    plot_model_comparison(
        results,
        save_path=os.path.join(Config.CHECKPOINT_PATH, 'backbone_comparison.png')
    )
    
    logger.info("\n保存对比实验结果...")
    with open(os.path.join(Config.CHECKPOINT_PATH, 'backbone_comparison_results.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("柑橘虫害骨干网络对比实验结果 (ConvNeXt-Tiny vs ResNet-18)\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"  精确率: {metrics['precision']:.4f}\n")
            f.write(f"  召回率: {metrics['recall']:.4f}\n")
            f.write(f"  F1值: {metrics['f1']:.4f}\n\n")
    
    logger.info("\n骨干网络对比实验完成!")


if __name__ == '__main__':
    main()
