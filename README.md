# 滑坡危险性及风险性动态评价模型

## 项目简介
本项目基于PyTorch、timm、GDAL和terratorch等框架，开发了一个用于滑坡危险性及风险性动态评价的地理空间深度学习模型。该模型综合考虑了地形、地质、水文气候、地表植被、人类活动等多个维度的特征，结合历史滑坡灾害数据，实现对滑坡风险的动态评估。

## 技术栈
- PyTorch: 深度学习框架
- timm: 预训练模型库
- GDAL: 地理空间数据处理
- terratorch: 地理空间深度学习工具
- NumPy: 数值计算
- Pandas: 数据处理
- Matplotlib/Seaborn: 数据可视化
- scikit-learn: 机器学习工具
- albumentations: 数据增强
- PyYAML: 配置管理

## 项目结构
```
landslide_risk/
├── configs/                    # 配置文件目录
│   └── default.yaml           # 默认配置文件
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 预处理后的数据
│   └── vector/                # 矢量数据
├── src/                       # 源代码
│   ├── data/                  # 数据处理模块
│   │   ├── dataset.py        # 数据集定义
│   │   ├── augmentation.py   # 数据增强
│   │   └── validation.py     # 数据验证
│   ├── models/                # 模型定义
│   │   ├── model.py          # 模型架构
│   │   ├── loss.py           # 损失函数
│   │   └── metrics.py        # 评估指标
│   ├── training/              # 训练相关
│   │   ├── trainer.py        # 训练器
│   │   └── train.py          # 训练脚本
│   └── utils/                 # 工具函数
│       ├── config.py         # 配置管理
│       ├── visualization.py   # 可视化工具
│       └── metrics.py        # 评估指标
├── tests/                     # 单元测试
├── requirements.txt           # 项目依赖
└── README.md                  # 项目文档
```

## 功能特性

### 1. 数据处理
- 多源数据支持
  - 栅格数据（DEM、坡向、坡度等）
  - 矢量数据（历史滑坡点）
  - 遥感数据
- 数据预处理
  - 空间分辨率统一
  - 数据标准化/归一化
  - 特征工程
- 数据增强
  - 几何变换
  - 噪声添加
  - 亮度对比度调整
  - 空间扭曲
- 数据验证
  - 数据完整性检查
  - 空间一致性验证
  - 数据质量评估

### 2. 模型架构
- 基于U-Net的改进架构
  - ResNet34编码器
  - 可配置的解码器通道
  - 注意力机制（可选）
- 损失函数
  - 组合损失（BCE + Dice）
  - Focal Loss
  - 空间连续性损失
- 评估指标
  - IoU
  - F1 Score
  - AUC-ROC
  - 精确率-召回率曲线

### 3. 训练系统
- 训练特性
  - 混合精度训练
  - 梯度裁剪
  - 学习率调度
  - 早停策略
- 监控与可视化
  - 训练损失曲线
  - 验证指标曲线
  - 学习率变化曲线
- 模型管理
  - 检查点保存
  - 最佳模型保存
  - 训练状态恢复

### 4. 配置系统
- YAML配置文件
- 灵活的配置管理
- 运行时参数覆盖
- 配置验证

## 使用说明

### 1. 环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
1. 将原始数据放入 `data/raw/` 目录
2. 运行数据预处理脚本
3. 运行数据验证脚本

### 3. 模型训练
```bash
# 使用默认配置训练
python src/training/train.py

# 使用自定义配置训练
python src/training/train.py --config configs/custom.yaml

# 从检查点恢复训练
python src/training/train.py --resume checkpoints/best_model.pth

# 指定训练设备
python src/training/train.py --device cuda
```

### 4. 模型评估
```bash
# 评估模型性能
python src/evaluation/evaluate.py --model checkpoints/best_model.pth

# 生成预测结果
python src/inference/predict.py --model checkpoints/best_model.pth
```

## 注意事项
1. 数据预处理需要确保所有栅格数据的空间分辨率一致
2. 模型训练时需要注意类别不平衡问题
3. 评估指标需要同时考虑像素级和区域级的准确性
4. 使用GPU训练时注意显存管理
5. 定期备份训练检查点

## 后续优化方向
1. 引入时序特征
2. 考虑多尺度预测
3. 集成多个模型结果
4. 添加不确定性评估
5. 支持分布式训练
6. 添加模型量化功能
7. 优化推理性能
8. 添加Web界面

## 贡献指南
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证
MIT License 