"""
柑橘虫害多视角检测主训练程序
功能：训练多视角注意力融合模型，进行柑橘虫害识别
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
from models import MultiViewAttentionFusionModel
from utils import (
    set_seed, CitrusPestDataset, calculate_metrics,
    plot_training_curves, save_checkpoint, load_checkpoint,
    setup_logger, prepare_data_list_for_multiview,
    plot_confusion_matrix, save_evaluation_report,
    load_dataset_from_videos, train_one_epoch, evaluate
)
# from grad_cam import GradCAM, get_backbone_last_layer, save_grad_cam_visualization


def main():
    parser = argparse.ArgumentParser(description='柑橘虫害多视角检测训练主程序')
    parser.add_argument('--data_dir', type=str, default='./datasets', 
                        help='数据集目录路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮次（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小（覆盖配置文件）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（覆盖配置文件）')
    parser.add_argument('--backbone', type=str, default=None,
                        help='骨干网络类型 (convnext_tiny, resnet18)')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=None,
                        help='是否使用学习率调度器（覆盖配置文件）')
    parser.add_argument('--lr_scheduler_type', type=str, default=None,
                        help='学习率调度器类型：cosine（覆盖配置文件）')
    parser.add_argument('--min_lr', type=float, default=None,
                        help='最小学习率（用于调度器，覆盖配置文件）')
    parser.add_argument('--model_name', type=str, default='attention_fusion',
                        help='模型名称，用于创建保存文件夹')
    parser.add_argument('--grad_cam_interval', type=int, default=None,
                        help='Grad-CAM 可视化间隔轮次（覆盖配置文件）')
    args = parser.parse_args()
    
    Config.create_directories()
    set_seed(Config.SEED)
    
    # 创建模型特定的保存目录
    checkpoint_dir = os.path.join(Config.CHECKPOINT_PATH, args.model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = setup_logger(os.path.join(checkpoint_dir, 'training.log'))
    logger.info("=" * 60)
    logger.info("柑橘虫害多视角检测训练程序启动")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"保存目录: {checkpoint_dir}")
    logger.info("=" * 60)
    
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        Config.LEARNING_RATE = args.lr
    if args.use_lr_scheduler is not None:
        Config.USE_LR_SCHEDULER = args.use_lr_scheduler
    if args.lr_scheduler_type is not None:
        Config.LR_SCHEDULER_TYPE = args.lr_scheduler_type
    if args.min_lr is not None:
        Config.MIN_LEARNING_RATE = args.min_lr
    if args.grad_cam_interval is not None:
        Config.GRAD_CAM_EPOCH_INTERVAL = args.grad_cam_interval
        
    logger.info(f"配置参数：批次大小={Config.BATCH_SIZE}, 学习率={Config.LEARNING_RATE}, 训练轮次={Config.NUM_EPOCHS}")
    logger.info(f"学习率调度器：使用={Config.USE_LR_SCHEDULER}, 类型={Config.LR_SCHEDULER_TYPE}, 最小 LR={Config.MIN_LEARNING_RATE}")
    # logger.info(f"Grad-CAM: 启用={Config.GRAD_CAM_ENABLED}, 间隔={Config.GRAD_CAM_EPOCH_INTERVAL} epochs, 视角={Config.GRAD_CAM_VIEWS}")
    
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
    model = MultiViewAttentionFusionModel(Config, backbone_model=args.backbone).to(device)
    
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
    best_val_loss = float('inf')  # 用于跟踪最佳验证集loss
    
    if args.resume is not None and os.path.exists(args.resume):
        logger.info(f"从检查点 {args.resume} 恢复训练...")
        model, optimizer, start_epoch, additional_info = load_checkpoint(
            model, optimizer, load_path=args.resume, device=device
        )
        if scheduler is not None and 'scheduler_state_dict' in additional_info:
            scheduler.load_state_dict(additional_info['scheduler_state_dict'])
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # # 初始化 Grad-CAM 工具（只在验证集上生成）
    # grad_cam_visualization_dir = os.path.join(checkpoint_dir, 'grad_cam_visualizations')
    # os.makedirs(grad_cam_visualization_dir, exist_ok=True)
    
    # grad_cam = None
    # grad_cam_samples = None
    
    # if Config.GRAD_CAM_ENABLED:
    #     # 从验证集获取一些样本用于 Grad-CAM 可视化
    #     for batch in val_loader:
    #         grad_cam_samples = batch
    #         break
        
    #     if grad_cam_samples is not None:
    #         logger.info("初始化 Grad-CAM 可视化工具...")
    #         try:
    #             target_layer = get_backbone_last_layer(model, args.backbone or Config.BACKBONE_MODEL)
    #             grad_cam = GradCAM(model, target_layer, device)
    #             logger.info(f"Grad-CAM 目标层：{target_layer}")
    #             logger.info(f"Grad-CAM 配置：每 {Config.GRAD_CAM_EPOCH_INTERVAL} 个 epoch 生成一次，视角={Config.GRAD_CAM_VIEWS}")
    #         except Exception as e:
    #             logger.warning(f"Grad-CAM 初始化失败：{str(e)}")
    #             grad_cam = None
    #     else:
    #         logger.warning("无法从验证集获取 Grad-CAM 样本")
    #         grad_cam = None
    
    logger.info("开始训练...")
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        logger.info("-" * 50)
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
        
        # # 生成 Grad-CAM 可视化（只在验证集上）
        # if grad_cam is not None and grad_cam_samples is not None and (epoch + 1) % Config.GRAD_CAM_EPOCH_INTERVAL == 0:
        #     logger.info("生成 Grad-CAM 可视化...")
        #     try:
        #         grad_cam_dir = os.path.join(grad_cam_visualization_dir, f'epoch_{epoch + 1}')
        #         os.makedirs(grad_cam_dir, exist_ok=True)
                
        #         # 选择样本进行可视化
        #         num_samples = min(Config.GRAD_CAM_NUM_SAMPLES, len(grad_cam_samples['images']))
                
        #         for sample_idx in range(num_samples):
        #             # 获取单个样本
        #             sample_images = grad_cam_samples['images'][sample_idx:sample_idx + 1].to(device)
        #             true_label = grad_cam_samples['label'][sample_idx].item()
                    
        #             # 为每个指定视角生成 Grad-CAM
        #             for view_idx in Config.GRAD_CAM_VIEWS:
        #                 if view_idx >= Config.NUM_VIEWS:
        #                     continue
                        
        #                 cam, probs, view_image = grad_cam.generate_cam_for_multiview(
        #                     sample_images, target_class=true_label, view_idx=view_idx
        #                 )
                        
        #                 predicted_class = np.argmax(probs[0])
        #                 confidence = probs[0][predicted_class]
                        
        #                 class_names = ['健康', '患虫']
        #                 title = f'True: {class_names[true_label]}, Pred: {class_names[predicted_class]}'
                        
        #                 save_path = os.path.join(
        #                     grad_cam_dir, 
        #                     f'sample_{sample_idx}_view_{view_idx}_class_{class_names[predicted_class]}.png'
        #                 )
                        
        #                 save_grad_cam_visualization(
        #                     view_image, cam, save_path,
        #                     title=title,
        #                     class_name=class_names[predicted_class],
        #                     confidence=confidence
        #                 )
                
        #         logger.info(f"Grad-CAM 可视化已保存至 {grad_cam_dir}")
        #     except Exception as e:
        #         logger.error(f"Grad-CAM 可视化失败：{str(e)}")
            
    logger.info("\n训练完成!")
    logger.info(f"最佳验证 F1 值: {best_f1:.4f}")
    
    # # 保存最后训练得到的模型
    # last_model_path = os.path.join(Config.CHECKPOINT_PATH, 'last_model.pth')
    # save_checkpoint(
    #     model, optimizer, Config.NUM_EPOCHS, val_metrics,
    #     last_model_path, scheduler
    # )
    # logger.info(f"最后训练的模型已保存至 {last_model_path}")
    
    plot_training_curves(train_losses, train_accs, val_losses, val_accs,
                        save_path=os.path.join(checkpoint_dir, 'training_curves.png'))
    
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
        model_type=args.model_name,
        save_path=os.path.join(checkpoint_dir, 'evaluation_report.txt')
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("所有流程执行完毕!")
    logger.info(f"测试集最终指标:")
    logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"  精确率: {test_metrics['precision']:.4f}")
    logger.info(f"  召回率: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 值: {test_metrics['f1']:.4f}")
    logger.info("=" * 60)
    
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU内存已释放")


if __name__ == '__main__':
    main()
