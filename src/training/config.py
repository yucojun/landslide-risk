"""训练配置文件"""

# 数据配置
DATA_CONFIG = {
    'data_dir': 'data',
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'batch_size': 8,
    'num_workers': 4,
    'input_size': (256, 256),
    'buffer_size': 100,  # 滑坡点缓冲区大小（米）
}

# 模型配置
MODEL_CONFIG = {
    'input_channels': 15,  # 输入特征通道数
    'num_classes': 1,      # 输出类别数
    'backbone': 'efficientnet_b0',
    'pretrained': True,
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'scheduler': {
        'type': 'ReduceLROnPlateau',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-6
    }
}

# 损失函数配置
LOSS_CONFIG = {
    'type': 'CombinedLoss',
    'weights': {
        'bce': 0.5,
        'dice': 0.5
    }
}

# 评估指标配置
METRICS_CONFIG = {
    'threshold': 0.5,  # 二值化阈值
    'iou_threshold': 0.5,  # IoU阈值
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'save_dir': 'results/visualizations',
    'plot_frequency': 5,  # 每隔多少个epoch绘制一次
}

# 模型保存配置
SAVE_CONFIG = {
    'save_dir': 'results/models',
    'save_frequency': 5,  # 每隔多少个epoch保存一次
    'save_best_only': True,  # 是否只保存最佳模型
}

# 日志配置
LOGGING_CONFIG = {
    'log_dir': 'results/logs',
    'log_frequency': 1,  # 每隔多少个batch记录一次
}

# 设备配置
DEVICE_CONFIG = {
    'use_gpu': True,  # 是否使用GPU
    'gpu_id': 0,      # GPU ID
} 