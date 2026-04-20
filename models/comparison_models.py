"""
对比实验模型模块
功能：实现单视角模型、简单拼接模型等对比实验模型
作者：柑橘虫害检测项目组
日期：2026年
"""
import torch
import torch.nn as nn
import timm


class SingleViewModel(nn.Module):
    """
    单视角模型 - 只使用一张图像进行分类 (使用第一张图像)
    """
    
    def __init__(self, config, backbone_model: str = None):
        super(SingleViewModel, self).__init__()
        
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
            
        self.num_classes = config.NUM_CLASSES
        self.dropout_rate = config.DROPOUT_RATE
        
        try:
            self.backbone = timm.create_model(
                self.backbone_model,
                pretrained=False,
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
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.image_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.classifier = nn.Linear(256, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        img = x[:, 0, :, :, :]
        features = self.backbone(img)
        
        fc_features = self.fc_layers(features)
        logits = self.classifier(fc_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities


class SimpleConcatModel(nn.Module):
    """
    简单拼接模型 - 将5张图像的特征简单拼接后分类
    """
    
    def __init__(self, config, backbone_model: str = None):
        super(SimpleConcatModel, self).__init__()
        
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
        self.dropout_rate = config.DROPOUT_RATE
        
        try:
            self.backbone = timm.create_model(
                self.backbone_model,
                pretrained=False,
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
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.image_feature_dim * self.num_views, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.classifier = nn.Linear(256, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        features_list = []
        
        for i in range(self.num_views):
            img = x[:, i, :, :, :]
            feat = self.backbone(img)
            features_list.append(feat)
        
        features = torch.cat(features_list, dim=1)
        
        fc_features = self.fc_layers(features)
        logits = self.classifier(fc_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM模型 - 使用CNN提取特征后用LSTM处理序列
    """
    
    def __init__(self, config, backbone_model: str = None):
        super(CNNLSTMModel, self).__init__()
        
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
        self.dropout_rate = config.DROPOUT_RATE
        self.hidden_dim = 256
        self.num_layers = 2
        
        try:
            self.backbone = timm.create_model(
                self.backbone_model,
                pretrained=False,
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
        
        self.lstm = nn.LSTM(
            input_size=self.image_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.classifier = nn.Linear(128, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        features_list = []
        
        for i in range(self.num_views):
            img = x[:, i, :, :, :]
            feat = self.backbone(img)
            features_list.append(feat)
        
        features = torch.stack(features_list, dim=1)
        
        lstm_out, (hidden, cell) = self.lstm(features)
        
        hidden_concat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        fc_features = self.fc_layers(hidden_concat)
        logits = self.classifier(fc_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities


class CNNLSTMAttentionModel(nn.Module):
    """
    CNN-LSTM-Attention模型 - CNN提取特征 + LSTM处理序列 + 注意力机制融合
    """
    
    def __init__(self, config, backbone_model: str = None):
        super(CNNLSTMAttentionModel, self).__init__()
        
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
        self.dropout_rate = config.DROPOUT_RATE
        self.hidden_dim = 256
        self.num_layers = 2
        self.num_attention_heads = 4
        
        try:
            self.backbone = timm.create_model(
                self.backbone_model,
                pretrained=False,
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
        
        self.lstm = nn.LSTM(
            input_size=self.image_feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=self.num_attention_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        self.classifier = nn.Linear(128, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        features_list = []
        
        for i in range(self.num_views):
            img = x[:, i, :, :, :]
            feat = self.backbone(img)
            features_list.append(feat)
        
        features = torch.stack(features_list, dim=1)
        
        lstm_out, (hidden, cell) = self.lstm(features)
        
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        attn_output = self.layer_norm(lstm_out + attn_output)
        
        fused_features = torch.mean(attn_output, dim=1)
        
        fc_features = self.fc_layers(fused_features)
        logits = self.classifier(fc_features)
        probabilities = self.softmax(logits)
        
        return logits, probabilities, attn_weights
