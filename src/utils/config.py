import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.model import create_model
from src.models.loss import CombinedLoss, DiceLoss, FocalLoss
from src.data.dataset import LandslideDataset
from src.data.augmentation import GeospatialAugmentation

class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str):
        """
        初始化配置管理器
        
        Args:
            config_path (str): 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _validate_config(self):
        """验证配置有效性"""
        required_sections = [
            'data', 'model', 'training', 'optimizer',
            'scheduler', 'loss', 'metrics', 'augmentation'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
    
    def get_device(self) -> torch.device:
        """
        获取计算设备
        
        Returns:
            torch.device: 计算设备
        """
        device_name = self.config['training']['device']
        if device_name == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available, falling back to CPU")
            device_name = 'cpu'
        return torch.device(device_name)
    
    def create_model(self) -> torch.nn.Module:
        """
        创建模型
        
        Returns:
            torch.nn.Module: 模型实例
        """
        model_config = self.config['model']
        return create_model(
            model_name=model_config['name'],
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            pretrained=model_config['pretrained'],
            use_attention=model_config['use_attention']
        )
    
    def create_criterion(self) -> torch.nn.Module:
        """
        创建损失函数
        
        Returns:
            torch.nn.Module: 损失函数实例
        """
        loss_config = self.config['loss']
        if loss_config['name'] == 'combined':
            return CombinedLoss(weights=loss_config['weights'])
        elif loss_config['name'] == 'dice':
            return DiceLoss()
        elif loss_config['name'] == 'focal':
            return FocalLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_config['name']}")
    
    def create_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """
        创建优化器
        
        Args:
            model (torch.nn.Module): 模型实例
            
        Returns:
            optim.Optimizer: 优化器实例
        """
        optimizer_config = self.config['optimizer']
        if optimizer_config['name'] == 'adamw':
            return optim.AdamW(
                model.parameters(),
                **optimizer_config['params']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    def create_scheduler(
        self,
        optimizer: optim.Optimizer
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        创建学习率调度器
        
        Args:
            optimizer (optim.Optimizer): 优化器实例
            
        Returns:
            Optional[optim.lr_scheduler._LRScheduler]: 学习率调度器实例
        """
        scheduler_config = self.config['scheduler']
        if scheduler_config['name'] == 'reduce_lr_on_plateau':
            return ReduceLROnPlateau(
                optimizer,
                **scheduler_config['params']
            )
        return None
    
    def create_datasets(self) -> tuple:
        """
        创建数据集
        
        Returns:
            tuple: (训练集, 验证集, 测试集)
        """
        data_config = self.config['data']
        aug_config = self.config['augmentation']
        
        # 创建数据增强器
        train_aug = GeospatialAugmentation(**aug_config['train'])
        val_aug = GeospatialAugmentation(**aug_config['val'])
        
        # 创建数据集
        train_dataset = LandslideDataset(
            data_dir=data_config['train_data_dir'],
            landslide_points=data_config['landslide_points'],
            patch_size=data_config['patch_size'],
            transform=train_aug
        )
        
        val_dataset = LandslideDataset(
            data_dir=data_config['val_data_dir'],
            landslide_points=data_config['landslide_points'],
            patch_size=data_config['patch_size'],
            transform=val_aug
        )
        
        test_dataset = LandslideDataset(
            data_dir=data_config['test_data_dir'],
            landslide_points=data_config['landslide_points'],
            patch_size=data_config['patch_size'],
            transform=val_aug
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: LandslideDataset,
        val_dataset: LandslideDataset,
        test_dataset: LandslideDataset
    ) -> tuple:
        """
        创建数据加载器
        
        Args:
            train_dataset (LandslideDataset): 训练集
            val_dataset (LandslideDataset): 验证集
            test_dataset (LandslideDataset): 测试集
            
        Returns:
            tuple: (训练加载器, 验证加载器, 测试加载器)
        """
        data_config = self.config['data']
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        return train_loader, val_loader, test_loader
    
    def create_directories(self):
        """创建必要的目录"""
        dirs = [
            self.config['training']['checkpoint_dir'],
            self.config['training']['log_dir'],
            self.config['metrics']['plot_dir']
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def save_config(self, save_dir: str):
        """
        保存配置到文件
        
        Args:
            save_dir (str): 保存目录
        """
        save_path = os.path.join(save_dir, 'config.yaml')
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False) 