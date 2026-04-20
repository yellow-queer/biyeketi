"""
项目配置文件
功能：定义项目全局配置参数，包括数据路径、模型参数、训练超参数等
日期：2026年
"""
import torch
import os

class Config:
    # 随机种子设置，保证可复现性
    SEED = 2004
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据配置
    DATASET_PATH = './datasets'
    CHECKPOINT_PATH = './checkpoints'
    
    # 图像数据配置
    IMAGE_SIZE = 224  # 图像尺寸 (224x224)
    NUM_VIEWS = 5  # 每个样本的视角数量 (4张水平+1张顶角)
    VIEW_NAMES = ['0°', '90°', '180°', '270°', '顶角']
    
    # 骨干网络配置（3/14号，我进行了修改：出现报错下载预训练权重为1024，我将其修改为768对应的模型convnext_tiny）
    # ConvNeXt系列模型及其特征维度：
    # - convnext_tiny: 768 维（轻量级，推荐使用）
    # - convnext_small: 768 维
    # - convnext_base: 1024 维
    # - convnext_large: 1536 维
    # - resnet18: 512 维
    BACKBONE_MODEL = 'convnext_tiny'  # 骨干网络类型
    IMAGE_FEATURE_DIM = 768  # 单张图像的特征维度（已自动适配，无需手动修改）
    
    # 多视角融合网络配置
    POSITIONAL_ENCODING_DIM = 768  # 位置编码维度
    NUM_ATTENTION_HEADS = 8  # 多头注意力头数
    FUSION_HIDDEN_DIM = 512  # 融合层隐藏维度
    
    # 训练配置
    # Batch Size=8 太小意味着模型每步只看到极少数的图像组合，产生的梯度噪声极大（这就是曲线乱跳的原因）。
    # 学习率太大可能会跳出最优解，导致模型训练不稳定。1e-4-》5e-5
    BATCH_SIZE = 12
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 50#50
    TRAIN_TEST_RATIO = 0.7  # 训练集测试集划分比例 7:3
    
    # 学习率调度器配置
    USE_LR_SCHEDULER = True  # 是否使用学习率调度器
    LR_SCHEDULER_TYPE = 'cosine'  # 学习率调度器类型: 'cosine' (CosineAnnealingLR)
    MIN_LEARNING_RATE = 1e-7  # 最小学习率（用于调度器）
    
    # 模型配置
    FREEZE_CONVNEXT = False  # 是否冻结 ConvNeXt 预训练权重
    DROPOUT_RATE = 0.3
    
    # 类别配置
    NUM_CLASSES = 2  # 二分类：健康/患虫
    
    # # Grad-CAM 可视化配置
    # GRAD_CAM_ENABLED = True  # 是否启用 Grad-CAM 可视化
    # GRAD_CAM_EPOCH_INTERVAL = 10  # Grad-CAM 可视化间隔轮次（每 N 个 epoch 生成一次）
    # GRAD_CAM_VIEWS = [0, 2, 4]  # 要生成 Grad-CAM 的视角索引列表 [0, 2, 4] 表示水平 0°、180°和顶角视角
    # GRAD_CAM_NUM_SAMPLES = 3  # 每次生成 Grad-CAM 的样本数量
    
    @classmethod
    def create_directories(cls):
        if not os.path.exists(cls.CHECKPOINT_PATH):
            os.makedirs(cls.CHECKPOINT_PATH)
        if not os.path.exists(cls.DATASET_PATH):
            os.makedirs(cls.DATASET_PATH)
