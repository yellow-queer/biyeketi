"""
对比实验训练程序
功能：训练单视角模型、简单拼接模型、CNN-LSTM模型、CNN-LSTM-Attention模型等对比实验模型
日期：2026年
"""
import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from config import Config
from data_preprocess import ImagePreprocessor
from models import SingleViewModel, SimpleConcatModel, CNNLSTMModel, CNNLSTMAttentionModel
from utils import (
    set_seed, CitrusPestDataset, calculate_metrics,
    plot_training_curves, save_checkpoint, load_checkpoint,
    setup_logger, prepare_data_list_for_multiview,
    plot_confusion_matrix, save_evaluation_report,
    load_dataset_from_videos, train_one_epoch, evaluate
)


def main():
    parser = argparse.ArgumentParser(description='柑橘虫害对比实验模型训练程序')
    parser.add_argument('--data_dir', type=str, default='./datasets', 
                        help='数据集目录路径')
    parser.add_argument('--model_type', type=str, default='single_view',
                        choices=['single_view', 'simple_concat', 'cnn_lstm', 'cnn_lstm_attention'],
                        help='模型类型 (single_view, simple_concat, cnn_lstm, cnn_lstm_attention)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮次（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置文件）')
    parser.add_argument('--backbone', type=str, default=None,
                        help='骨干网络类型 (convnext_tiny, resnet18等)')
    args = parser.parse_args()
    
    Config.create_directories()
    set_seed(Config.SEED)
    
    model_names = {
        'single_view': '单视角模型',
        'simple_concat': '多视角简单拼接模型',
        'cnn_lstm': 'CNN-LSTM模型',
        'cnn_lstm_attention': 'CNN-LSTM-Attention模型'
    }
    
    checkpoint_dir = os.path.join(Config.CHECKPOINT_PATH, args.model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(checkpoint_dir, 'training.log'))
    logger.info("=" * 70)
    logger.info(f"{model_names[args.model_type]}训练程序启动")
    logger.info(f"保存目录: {checkpoint_dir}")
    logger.info("=" * 70)
    
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
        
    logger.info(f"配置参数: 模型类型={args.model_type}, 批次大小={Config.BATCH_SIZE}, "
                f"学习率={Config.LEARNING_RATE}, 训练轮次={Config.NUM_EPOCHS}")
    
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
    
    logger.info("初始化模型...")
    if args.model_type == 'single_view':
        model = SingleViewModel(Config, backbone_model=args.backbone).to(device)
    elif args.model_type == 'simple_concat':
        model = SimpleConcatModel(Config, backbone_model=args.backbone).to(device)
    elif args.model_type == 'cnn_lstm':
        model = CNNLSTMModel(Config, backbone_model=args.backbone).to(device)
    elif args.model_type == 'cnn_lstm_attention':
        model = CNNLSTMAttentionModel(Config, backbone_model=args.backbone).to(device)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    scheduler = None
    if Config.USE_LR_SCHEDULER:
        if Config.LR_SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=Config.NUM_EPOCHS,
                eta_min=Config.MIN_LEARNING_RATE
            )
            logger.info("使用 CosineAnnealingLR 学习率调度器")
    
    start_epoch = 0
    best_f1 = 0.0
    best_val_loss = float('inf')
    
    if args.resume is not None and os.path.exists(args.resume):
        logger.info(f"从检查点 {args.resume} 恢复训练...")
        model, optimizer, start_epoch, _ = load_checkpoint(
            model, optimizer, load_path=args.resume, device=device
        )
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    logger.info("开始训练...")
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        logger.info("-" * 60)
        logger.info(f"当前学习率: {current_lr:.8f}")
        
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device, logger, "验证集")
        
        train_losses.append(train_loss)
        train_accs.append(train_metrics['accuracy'])
        val_losses.append(val_loss)
        val_accs.append(val_metrics['accuracy'])
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics,
                os.path.join(checkpoint_dir, 'best_model.pth'),
                scheduler
            )
            logger.info(f"最佳模型已保存 (F1: {best_f1:.4f}, Val Loss: {val_loss:.4f})")
        elif val_metrics['f1'] == best_f1 and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch + 1, val_metrics,
                os.path.join(checkpoint_dir, 'best_model.pth'),
                scheduler
            )
            logger.info(f"最佳模型已更新 (相同F1，更小Val Loss: {val_loss:.4f})")
        
        if scheduler is not None:
            scheduler.step()
            
    logger.info("\n训练完成!")
    logger.info(f"最佳验证 F1 值: {best_f1:.4f}")
    
    plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        save_path=os.path.join(checkpoint_dir, 'training_curves.png')
    )
    
    logger.info("\n在测试集上评估最佳模型...")
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    model, _, _, _ = load_checkpoint(model, load_path=best_model_path, device=device)
    
    test_loss, test_metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device, logger, "测试集")
    
    logger.info("\n生成混淆矩阵...")
    confusion_mat = plot_confusion_matrix(
        y_true, y_pred, 
        save_path=os.path.join(checkpoint_dir, 'confusion_matrix.png')
    )
    
    logger.info("保存评估报告...")
    save_evaluation_report(
        test_metrics, confusion_mat,
        Config,
        model_type=args.model_type,
        save_path=os.path.join(checkpoint_dir, 'evaluation_report.txt')
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("所有流程执行完毕!")
    logger.info(f"测试集最终指标:")
    logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"  精确率: {test_metrics['precision']:.4f}")
    logger.info(f"  召回率: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 值: {test_metrics['f1']:.4f}")
    logger.info("=" * 70)
    
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU内存已释放")


if __name__ == '__main__':
    main()
