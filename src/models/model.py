import torch
import torch.nn as nn
import timm
from typing import List, Tuple, Optional
import math

class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        use_bn: bool = True,
        use_act: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        use_bn: bool = True
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
        self.conv1 = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            use_bn=use_bn
        )
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            use_bn=use_bn
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class LandslideRiskNet(nn.Module):
    """滑坡风险评估模型"""
    def __init__(
        self,
        in_channels: int = 15,  # 输入通道数
        out_channels: int = 1,  # 输出通道数（风险图）
        encoder_name: str = 'resnet34',  # 编码器名称
        pretrained: bool = True,  # 是否使用预训练权重
        use_bn: bool = True,  # 是否使用批归一化
        decoder_channels: List[int] = [256, 128, 64, 32, 16]  # 解码器通道数
    ):
        super().__init__()
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True
        )
        
        # 获取编码器特征通道数
        encoder_channels = self.encoder.feature_info.channels()
        
        # 解码器
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[-(i+2)] if i < len(encoder_channels)-1 else encoder_channels[-1]
            skip_ch = encoder_channels[-(i+3)] if i < len(encoder_channels)-2 else 0
            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=decoder_channels[i],
                    skip_channels=skip_ch,
                    use_bn=use_bn
                )
            )
        
        # 最终输出层
        self.final_conv = nn.Conv2d(
            decoder_channels[-1],
            out_channels,
            kernel_size=1
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器前向传播
        features = self.encoder(x)
        
        # 解码器前向传播
        x = features[-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[-(i+3)] if i < len(features)-2 else None
            x = decoder_block(x, skip)
        
        # 最终输出
        x = self.final_conv(x)
        return torch.sigmoid(x)  # 使用sigmoid激活函数输出风险概率

class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv1(x)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class LandslideRiskNetWithAttention(LandslideRiskNet):
    """带注意力机制的滑坡风险评估模型"""
    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 1,
        encoder_name: str = 'resnet34',
        pretrained: bool = True,
        use_bn: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32, 16]
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_name=encoder_name,
            pretrained=pretrained,
            use_bn=use_bn,
            decoder_channels=decoder_channels
        )
        
        # 添加注意力层
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(channels)
            for channels in decoder_channels
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器前向传播
        features = self.encoder(x)
        
        # 解码器前向传播（带注意力）
        x = features[-1]
        for i, (decoder_block, attention_block) in enumerate(zip(self.decoder_blocks, self.attention_blocks)):
            skip = features[-(i+3)] if i < len(features)-2 else None
            x = decoder_block(x, skip)
            x = attention_block(x)
        
        # 最终输出
        x = self.final_conv(x)
        return torch.sigmoid(x)

def create_model(
    model_name: str = 'unet',
    in_channels: int = 15,
    out_channels: int = 1,
    pretrained: bool = True,
    use_attention: bool = False
) -> nn.Module:
    """
    创建模型实例
    
    Args:
        model_name (str): 模型名称
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        pretrained (bool): 是否使用预训练权重
        use_attention (bool): 是否使用注意力机制
    
    Returns:
        nn.Module: 模型实例
    """
    if model_name == 'unet':
        if use_attention:
            return LandslideRiskNetWithAttention(
                in_channels=in_channels,
                out_channels=out_channels,
                pretrained=pretrained
            )
        else:
            return LandslideRiskNet(
                in_channels=in_channels,
                out_channels=out_channels,
                pretrained=pretrained
            )
    else:
        raise ValueError(f"Unsupported model name: {model_name}") 