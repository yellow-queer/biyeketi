"""
多视角注意力融合模型
功能：实现基于多视角图像注意力融合的柑橘虫害检测核心模型
作者：柑橘虫害检测项目组
日期：2026年
"""
import torch
import torch.nn as nn
import timm


class MultiViewAttentionFusionModel(nn.Module):
    """
    多视角注意力融合柑橘虫害分类模型
    架构：
    1. 骨干网络特征提取 (ConvNeXt/ResNet-18)
    2. 视角位置编码 (可学习编码)
    3. 多头注意力池化 (自适应融合)
    4. 全连接层+Softmax
    """
    
    def __init__(self, config, backbone_model: str = None):
        super(MultiViewAttentionFusionModel, self).__init__()
        
        self.backbone_model = backbone_model if backbone_model is not None else config.BACKBONE_MODEL
        
        # 动态确定特征维度
        if self.backbone_model == 'resnet18':
            self.image_feature_dim = 512
        elif self.backbone_model == 'convnext_tiny':
            self.image_feature_dim = 768
        elif self.backbone_model == 'convnext_small':
            self.image_feature_dim = 768
        elif self.backbone_model == 'convnext_base':
            self.image_feature_dim = 1024
        elif self.backbone_model == 'convnext_large':
            self.image_feature_dim = 1536
        else:
            self.image_feature_dim = 768  # 默认值
            
        self.num_views = config.NUM_VIEWS
        self.num_classes = config.NUM_CLASSES
        self.num_attention_heads = config.NUM_ATTENTION_HEADS
        self.positional_encoding_dim = self.image_feature_dim
        self.fusion_hidden_dim = config.FUSION_HIDDEN_DIM
        self.dropout_rate = config.DROPOUT_RATE
        
        try:
            self.backbone = timm.create_model(
                self.backbone_model,
                pretrained=True,
                num_classes=0
            )
            print(f"成功加载 {self.backbone_model} 预训练权重")
        except Exception as e:
            print(f"警告: 无法下载预训练权重，使用随机初始化: {str(e)}")
            self.backbone = timm.create_model(
                self.backbone_model,
                pretrained=False,
                num_classes=0
            )
        
        if config.FREEZE_CONVNEXT:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.positional_encoding = nn.Parameter(
            torch.randn(self.num_views, self.positional_encoding_dim)
        )
        
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.image_feature_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(self.image_feature_dim)
        self.layer_norm2 = nn.LayerNorm(self.image_feature_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.image_feature_dim, self.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fusion_hidden_dim, self.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.classifier = nn.Linear(self.fusion_hidden_dim // 2, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def extract_features(self, x):
        batch_size = x.shape[0]
        features_list = []
        
        for i in range(self.num_views):
            img = x[:, i, :, :, :]
            feat = self.backbone(img)
            features_list.append(feat)
        
        features = torch.stack(features_list, dim=1)
        return features
    
    def forward(self, x):
        features = self.extract_features(x)
        
        features = features + self.positional_encoding.unsqueeze(0)
        
        attn_output, attn_weights = self.multi_head_attention(
            features, features, features
        )
        
        features = self.layer_norm1(features + attn_output)
        
        fused_features = torch.mean(features, dim=1)
        
        fused_features = self.layer_norm2(fused_features)
        
        fc_features = self.fc_layers(fused_features)
        logits = self.classifier(fc_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities, attn_weights
