import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, smooth=1.0):
        """
        初始化Dice损失
        
        Args:
            smooth (float): 平滑因子
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        计算Dice损失
        
        Args:
            pred (torch.Tensor): 预测结果
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: Dice损失值
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    """Focal损失函数"""
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        初始化Focal损失
        
        Args:
            alpha (float): 类别权重因子
            gamma (float): 聚焦参数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """
        计算Focal损失
        
        Args:
            pred (torch.Tensor): 预测结果
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: Focal损失值
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, weights={'bce': 0.5, 'dice': 0.5}):
        """
        初始化组合损失
        
        Args:
            weights (dict): 各损失函数的权重
        """
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        """
        计算组合损失
        
        Args:
            pred (torch.Tensor): 预测结果
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 组合损失值
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        total_loss = self.weights['bce'] * bce_loss + self.weights['dice'] * dice_loss
        return total_loss

class SpatialContinuityLoss(nn.Module):
    """空间连续性损失函数"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred):
        """
        计算空间连续性损失
        
        Args:
            pred (torch.Tensor): 预测结果
            
        Returns:
            torch.Tensor: 空间连续性损失值
        """
        # 计算水平和垂直方向的梯度
        grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        # 计算梯度损失
        loss = torch.mean(grad_x) + torch.mean(grad_y)
        return loss 