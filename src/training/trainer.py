import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import json

class LandslideTrainer:
    """滑坡风险预测模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            criterion (nn.Module): 损失函数
            optimizer (optim.Optimizer): 优化器
            scheduler (optim.lr_scheduler): 学习率调度器
            device (torch.device): 计算设备
            config (dict): 配置字典
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # 初始化混合精度训练
        self.scaler = GradScaler()
        self.use_amp = config['training'].get('use_amp', True)
        
        # 初始化梯度裁剪
        self.grad_clip = config['training'].get('grad_clip', 1.0)
        
        # 初始化日志
        self.setup_logging()
        
        # 初始化训练状态
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config['training']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self):
        """
        训练一个epoch
        
        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 使用混合精度训练
                if self.use_amp:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    
                    # 梯度裁剪
                    if self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # 记录学习率
                if batch_idx == 0:
                    self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """
        验证模型
        
        Returns:
            float: 平均验证损失
            dict: 评估指标
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算评估指标
        metrics = self.calculate_metrics(np.array(all_preds), np.array(all_targets))
        
        return total_loss / len(self.val_loader), metrics
    
    def calculate_metrics(self, preds, targets):
        """
        计算评估指标
        
        Args:
            preds (np.ndarray): 预测结果
            targets (np.ndarray): 真实标签
            
        Returns:
            dict: 评估指标
        """
        # 将预测结果展平
        preds = preds.ravel()
        targets = targets.ravel()
        
        # 计算二值化预测
        threshold = self.config['metrics']['threshold']
        binary_preds = (preds > threshold).astype(int)
        
        # 计算AUC-ROC
        auc_roc = roc_auc_score(targets, preds)
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(targets, preds)
        avg_precision = average_precision_score(targets, preds)
        
        # 计算IoU
        intersection = np.sum((binary_preds == 1) & (targets == 1))
        union = np.sum((binary_preds == 1) | (targets == 1))
        iou = intersection / (union + 1e-6)
        
        # 计算F1分数
        true_positives = np.sum((binary_preds == 1) & (targets == 1))
        false_positives = np.sum((binary_preds == 1) & (targets == 0))
        false_negatives = np.sum((binary_preds == 0) & (targets == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return {
            'auc_roc': auc_roc,
            'avg_precision': avg_precision,
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1
        }
    
    def plot_metrics(self, save_path=None):
        """
        绘制评估指标
        
        Args:
            save_path (str, optional): 保存路径
        """
        plt.figure(figsize=(15, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制学习率曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        
        # 绘制PR曲线
        plt.subplot(1, 3, 3)
        metrics = self.calculate_metrics(
            np.array([pred for batch in self.val_loader for pred in self.model(batch[0].to(self.device)).cpu().numpy()]),
            np.array([target for _, target in self.val_loader])
        )
        plt.plot(metrics['recall'], metrics['precision'])
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def train(self, num_epochs, early_stopping_patience=10, resume_from=None):
        """
        训练模型
        
        Args:
            num_epochs (int): 训练轮数
            early_stopping_patience (int): 早停耐心值
            resume_from (str, optional): 恢复训练的检查点路径
        """
        # 如果提供了检查点，加载训练状态
        if resume_from:
            self.load_model(resume_from)
            self.logger.info(f"Resumed training from {resume_from}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, metrics = self.validate()
            
            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 记录指标
            self.logger.info(f'Train Loss: {train_loss:.4f}')
            self.logger.info(f'Val Loss: {val_loss:.4f}')
            self.logger.info(f'AUC-ROC: {metrics["auc_roc"]:.4f}')
            self.logger.info(f'Average Precision: {metrics["avg_precision"]:.4f}')
            self.logger.info(f'IoU: {metrics["iou"]:.4f}')
            self.logger.info(f'F1 Score: {metrics["f1"]:.4f}')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
                self.save_model(
                    Path(self.config['training']['checkpoint_dir']) / 'best_model.pth'
                )
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % self.config['training']['save']['save_frequency'] == 0:
                self.save_model(
                    Path(self.config['training']['checkpoint_dir']) / f'checkpoint_epoch_{epoch+1}.pth'
                )
            
            # 早停
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path (str): 保存路径
        """
        save_dict = {
            'model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(save_dict, path)
        self.logger.info(f"Saved model to {path}")
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path (str): 模型路径
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.best_val_loss = checkpoint['best_val_loss']
        self.config = checkpoint['config']
        self.logger.info(f"Loaded model from {path}") 