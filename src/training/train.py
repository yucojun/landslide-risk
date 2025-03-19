import argparse
import os
from pathlib import Path

from src.utils.config import Config
from src.training.trainer import LandslideTrainer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练滑坡风险评估模型')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复训练的检查点路径'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='训练设备 (cuda/cpu)'
    )
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 如果指定了设备，更新配置
    if args.device:
        config.config['training']['device'] = args.device
    
    # 创建必要的目录
    config.create_directories()
    
    # 获取设备
    device = config.get_device()
    
    # 创建模型
    model = config.create_model()
    model = model.to(device)
    
    # 创建损失函数
    criterion = config.create_criterion()
    
    # 创建优化器
    optimizer = config.create_optimizer(model)
    
    # 创建学习率调度器
    scheduler = config.create_scheduler(optimizer)
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset, test_dataset = config.create_datasets()
    train_loader, val_loader, test_loader = config.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # 创建训练器
    trainer = LandslideTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config.config
    )
    
    # 训练模型
    trainer.train(
        num_epochs=config.config['training']['num_epochs'],
        early_stopping_patience=config.config['training']['early_stopping_patience'],
        resume_from=args.resume
    )
    
    # 保存配置
    config.save_config(config.config['training']['checkpoint_dir'])
    
    # 绘制指标
    trainer.plot_metrics(
        os.path.join(
            config.config['training']['plot_dir'],
            'training_metrics.png'
        )
    )

if __name__ == '__main__':
    main() 