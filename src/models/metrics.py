import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化评估指标计算器
        
        Args:
            threshold (float): 二值化阈值
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.tp = 0  # 真正例
        self.fp = 0  # 假正例
        self.fn = 0  # 假负例
        self.tn = 0  # 真负例
        self.total_pixels = 0
        self.predictions = []
        self.targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        更新指标
        
        Args:
            pred (torch.Tensor): 预测结果
            target (torch.Tensor): 真实标签
        """
        # 确保输入是二值化的
        pred = (pred > self.threshold).float()
        target = (target > self.threshold).float()
        
        # 计算混淆矩阵元素
        self.tp += torch.sum((pred == 1) & (target == 1)).item()
        self.fp += torch.sum((pred == 1) & (target == 0)).item()
        self.fn += torch.sum((pred == 0) & (target == 1)).item()
        self.tn += torch.sum((pred == 0) & (target == 0)).item()
        
        # 更新总像素数
        self.total_pixels += pred.numel()
        
        # 保存预测值和真实值用于计算PR曲线
        self.predictions.extend(pred.cpu().numpy().flatten())
        self.targets.extend(target.cpu().numpy().flatten())
    
    def compute(self) -> Dict[str, float]:
        """
        计算所有指标
        
        Returns:
            Dict[str, float]: 指标字典
        """
        metrics = {}
        
        # 计算基本指标
        metrics['accuracy'] = (self.tp + self.tn) / self.total_pixels
        
        # 计算精确率和召回率
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics['f1'] = f1
        
        # 计算IoU
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        iou = intersection / union if union > 0 else 0
        metrics['iou'] = iou
        
        # 计算Kappa系数
        expected_accuracy = ((self.tp + self.fp) * (self.tp + self.fn) + 
                           (self.fn + self.tn) * (self.fp + self.tn)) / (self.total_pixels ** 2)
        kappa = (metrics['accuracy'] - expected_accuracy) / (1 - expected_accuracy)
        metrics['kappa'] = kappa
        
        # 计算PR曲线下面积
        try:
            metrics['auc_pr'] = average_precision_score(self.targets, self.predictions)
        except:
            metrics['auc_pr'] = 0
        
        return metrics

class LandslideSpecificMetrics:
    """滑坡特定评估指标"""
    
    @staticmethod
    def calculate_risk_levels(pred: torch.Tensor) -> torch.Tensor:
        """
        计算风险等级
        
        Args:
            pred (torch.Tensor): 预测结果
            
        Returns:
            torch.Tensor: 风险等级
        """
        # 定义风险等级阈值
        thresholds = torch.tensor([0.2, 0.4, 0.6, 0.8], device=pred.device)
        risk_levels = torch.zeros_like(pred, dtype=torch.long)
        
        for i, threshold in enumerate(thresholds):
            risk_levels[pred > threshold] = i + 1
        
        return risk_levels
    
    @staticmethod
    def calculate_risk_area_ratio(pred: torch.Tensor, threshold: float = 0.5) -> float:
        """
        计算风险区域比例
        
        Args:
            pred (torch.Tensor): 预测结果
            threshold (float): 风险阈值
            
        Returns:
            float: 风险区域比例
        """
        risk_area = torch.sum(pred > threshold).item()
        total_area = pred.numel()
        return risk_area / total_area
    
    @staticmethod
    def calculate_risk_contiguity(pred: torch.Tensor, threshold: float = 0.5) -> float:
        """
        计算风险区域连续性
        
        Args:
            pred (torch.Tensor): 预测结果
            threshold (float): 风险阈值
            
        Returns:
            float: 风险区域连续性得分
        """
        # 二值化预测结果
        binary_pred = (pred > threshold).float()
        
        # 计算水平和垂直方向的梯度
        grad_x = torch.abs(binary_pred[:, :, :, 1:] - binary_pred[:, :, :, :-1])
        grad_y = torch.abs(binary_pred[:, :, 1:, :] - binary_pred[:, :, :-1, :])
        
        # 计算边界像素比例
        boundary_pixels = torch.sum(grad_x) + torch.sum(grad_y)
        total_pixels = binary_pred.numel()
        
        return 1 - (boundary_pixels / total_pixels)
    
    @staticmethod
    def calculate_risk_uncertainty(pred: torch.Tensor) -> float:
        """
        计算预测的不确定性
        
        Args:
            pred (torch.Tensor): 预测结果
            
        Returns:
            float: 不确定性得分
        """
        # 计算预测值到0.5的距离
        uncertainty = 1 - 2 * torch.abs(pred - 0.5)
        return torch.mean(uncertainty).item()

def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        pred (torch.Tensor): 预测结果
        target (torch.Tensor): 真实标签
        threshold (float): 二值化阈值
        
    Returns:
        Dict[str, float]: 指标字典
    """
    # 创建指标计算器
    calculator = MetricsCalculator(threshold=threshold)
    
    # 更新指标
    calculator.update(pred, target)
    
    # 计算基本指标
    metrics = calculator.compute()
    
    # 计算滑坡特定指标
    metrics['risk_area_ratio'] = LandslideSpecificMetrics.calculate_risk_area_ratio(pred, threshold)
    metrics['risk_contiguity'] = LandslideSpecificMetrics.calculate_risk_contiguity(pred, threshold)
    metrics['risk_uncertainty'] = LandslideSpecificMetrics.calculate_risk_uncertainty(pred)
    
    return metrics 