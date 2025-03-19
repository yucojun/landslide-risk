import torch
import torch.nn as nn
import timm

class AttentionBlock(nn.Module):
    """注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention = self.attention(x)
        return x * attention

class LandslideNet(nn.Module):
    """滑坡风险预测网络"""
    def __init__(self, input_channels, num_classes=1):
        """
        初始化网络
        
        Args:
            input_channels (int): 输入通道数
            num_classes (int): 输出类别数
        """
        super().__init__()
        
        # 使用预训练的EfficientNet作为编码器
        self.encoder = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            in_chans=input_channels,
            num_classes=0
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            AttentionBlock(512),
            
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            AttentionBlock(256),
            
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            AttentionBlock(128),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            AttentionBlock(64),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            AttentionBlock(32),
            
            nn.Conv2d(32, num_classes, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量
            
        Returns:
            torch.Tensor: 预测结果
        """
        # 编码器
        features = self.encoder(x)
        
        # 重塑特征图
        features = features.view(features.size(0), -1, 8, 8)
        
        # 解码器
        output = self.decoder(features)
        
        return output

class MultiScaleLandslideNet(nn.Module):
    """多尺度滑坡风险预测网络"""
    def __init__(self, input_channels, num_classes=1):
        """
        初始化网络
        
        Args:
            input_channels (int): 输入通道数
            num_classes (int): 输出类别数
        """
        super().__init__()
        
        # 主干网络
        self.backbone = LandslideNet(input_channels, num_classes)
        
        # 多尺度特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(num_classes * 3, num_classes, 1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量
            
        Returns:
            torch.Tensor: 预测结果
        """
        # 主干网络预测
        pred = self.backbone(x)
        
        # 多尺度特征
        pred_1 = nn.functional.interpolate(pred, scale_factor=0.5, mode='bilinear', align_corners=True)
        pred_2 = nn.functional.interpolate(pred, scale_factor=2.0, mode='bilinear', align_corners=True)
        
        # 特征融合
        pred_1 = nn.functional.interpolate(pred_1, size=pred.size()[2:], mode='bilinear', align_corners=True)
        pred_2 = nn.functional.interpolate(pred_2, size=pred.size()[2:], mode='bilinear', align_corners=True)
        
        fused = torch.cat([pred, pred_1, pred_2], dim=1)
        output = self.fusion(fused)
        
        return output 