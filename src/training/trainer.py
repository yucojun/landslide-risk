import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

class LandslideTrainer:
    """滑坡风险预测模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        """
        初始化训练器
        
        Args:
            model (nn.Module): 模型
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader): 验证数据加载器
            criterion (nn.Module): 损失函数
            optimizer (optim.Optimizer): 优化器
            device (torch.device): 计算设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.train_losses = []
        self.val_losses = []
        
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
                output = self.model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
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
        
        # 计算AUC-ROC
        auc_roc = roc_auc_score(targets, preds)
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(targets, preds)
        avg_precision = average_precision_score(targets, preds)
        
        return {
            'auc_roc': auc_roc,
            'avg_precision': avg_precision,
            'precision': precision,
            'recall': recall
        }
    
    def plot_metrics(self, save_path=None):
        """
        绘制评估指标
        
        Args:
            save_path (str, optional): 保存路径
        """
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制PR曲线
        plt.subplot(1, 2, 2)
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
    
    def train(self, num_epochs, early_stopping_patience=10):
        """
        训练模型
        
        Args:
            num_epochs (int): 训练轮数
            early_stopping_patience (int): 早停耐心值
        """
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, metrics = self.validate()
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 打印指标
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'AUC-ROC: {metrics["auc_roc"]:.4f}')
            print(f'Average Precision: {metrics["avg_precision"]:.4f}')
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path (str): 保存路径
        """
        torch.save({
            'model_state_dict': self.best_model_state,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path (str): 模型路径
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss'] 