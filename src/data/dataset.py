import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import albumentations as A
from typing import Dict, List, Tuple, Optional

class LandslideDataset(Dataset):
    """滑坡数据集类"""
    
    def __init__(
        self,
        raster_data: Dict[str, str],
        landslide_points: str,
        input_size: Tuple[int, int] = (256, 256),
        buffer_size: int = 100,
        transform: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        """
        初始化数据集
        
        Args:
            raster_data (Dict[str, str]): 栅格数据路径字典
            landslide_points (str): 滑坡点矢量文件路径
            input_size (Tuple[int, int]): 输入图像大小
            buffer_size (int): 滑坡点缓冲区大小（米）
            transform (Optional[A.Compose]): 数据增强转换
            is_training (bool): 是否为训练模式
        """
        self.raster_data = raster_data
        self.landslide_points = landslide_points
        self.input_size = input_size
        self.buffer_size = buffer_size
        self.transform = transform
        self.is_training = is_training
        
        # 加载滑坡点数据
        self.gdf = gpd.read_file(landslide_points)
        
        # 创建正样本缓冲区
        self.positive_samples = self.gdf.buffer(buffer_size)
        
        # 获取参考栅格（使用DEM作为参考）
        self.reference_raster = raster_data['dem']
        with rasterio.open(self.reference_raster) as src:
            self.reference_meta = src.meta
            self.reference_transform = src.transform
            self.reference_crs = src.crs
        
        # 生成采样点
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Tuple[float, float]]:
        """
        生成采样点
        
        Returns:
            List[Tuple[float, float]]: 采样点坐标列表
        """
        samples = []
        
        # 获取研究区域边界
        bounds = self.gdf.total_bounds
        
        # 根据输入大小计算采样步长
        pixel_size = self.reference_meta['transform'][0]  # 像素大小（米）
        stride_x = self.input_size[0] * pixel_size
        stride_y = self.input_size[1] * pixel_size
        
        # 生成网格点
        x_coords = np.arange(bounds[0], bounds[2], stride_x)
        y_coords = np.arange(bounds[1], bounds[3], stride_y)
        
        for x in x_coords:
            for y in y_coords:
                samples.append((x, y))
        
        return samples
    
    def _extract_patch(self, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取图像块
        
        Args:
            x (float): x坐标
            y (float): y坐标
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特征和标签
        """
        # 计算图像块边界
        half_width = self.input_size[0] * self.reference_meta['transform'][0] / 2
        half_height = self.input_size[1] * self.reference_meta['transform'][1] / 2
        
        # 创建图像块边界框
        from shapely.geometry import box
        patch_box = box(x - half_width, y - half_height, x + half_width, y + half_height)
        
        # 提取特征
        features = []
        for raster_path in self.raster_data.values():
            with rasterio.open(raster_path) as src:
                # 将图像块投影到栅格数据的坐标系
                patch_box_proj = gpd.GeoDataFrame({'geometry': [patch_box]}, crs=self.reference_crs)
                patch_box_proj = patch_box_proj.to_crs(src.crs)
                
                # 提取数据
                out_image, out_transform = mask(src, patch_box_proj.geometry, crop=True)
                features.append(out_image[0])
        
        # 堆叠特征
        features = np.stack(features, axis=0)
        
        # 生成标签
        label = np.zeros(self.input_size, dtype=np.float32)
        if self.is_training:
            # 检查图像块是否包含滑坡点
            if patch_box.intersects(self.positive_samples.unary_union):
                # 将滑坡点投影到图像块坐标系
                landslide_points_proj = self.gdf.to_crs(self.reference_crs)
                landslide_points_proj = landslide_points_proj[landslide_points_proj.intersects(patch_box)]
                
                if not landslide_points_proj.empty:
                    # 创建标签
                    label = np.ones(self.input_size, dtype=np.float32)
        
        return features, label
    
    def __len__(self) -> int:
        """
        获取数据集大小
        
        Returns:
            int: 数据集大小
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 特征和标签
        """
        x, y = self.samples[idx]
        features, label = self._extract_patch(x, y)
        
        # 转换为PyTorch张量
        features = torch.from_numpy(features).float()
        label = torch.from_numpy(label).float()
        
        # 应用数据增强
        if self.transform and self.is_training:
            transformed = self.transform(image=features.transpose(1, 2, 0), mask=label)
            features = torch.from_numpy(transformed['image'].transpose(2, 0, 1)).float()
            label = torch.from_numpy(transformed['mask']).float()
        
        return features, label

def create_data_loaders(
    data_dir: str,
    landslide_points: str,
    batch_size: int = 8,
    num_workers: int = 4,
    input_size: Tuple[int, int] = (256, 256),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    buffer_size: int = 100
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建数据加载器
    
    Args:
        data_dir (str): 数据目录路径
        landslide_points (str): 滑坡点矢量文件路径
        batch_size (int): 批次大小
        num_workers (int): 数据加载线程数
        input_size (Tuple[int, int]): 输入图像大小
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        buffer_size (int): 滑坡点缓冲区大小
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: 训练、验证和测试数据加载器
    """
    # 获取栅格数据路径
    raster_data = {
        'dem': os.path.join(data_dir, 'dem.tif'),
        'slope': os.path.join(data_dir, 'slope.tif'),
        'aspect': os.path.join(data_dir, 'aspect.tif'),
        'lithology': os.path.join(data_dir, 'lithology.tif'),
        'fault_density': os.path.join(data_dir, 'fault_density.tif'),
        'structure': os.path.join(data_dir, 'structure.tif'),
        'quaternary': os.path.join(data_dir, 'quaternary.tif'),
        'mndwi': os.path.join(data_dir, 'mndwi.tif'),
        'twi': os.path.join(data_dir, 'twi.tif'),
        'drainage_density': os.path.join(data_dir, 'drainage_density.tif'),
        'ndvi': os.path.join(data_dir, 'ndvi.tif'),
        'ndbi': os.path.join(data_dir, 'ndbi.tif'),
        'road_density': os.path.join(data_dir, 'road_density.tif'),
        'population_density': os.path.join(data_dir, 'population_density.tif'),
        'remote_sensing': os.path.join(data_dir, 'remote_sensing.tif')
    }
    
    # 创建数据增强pipeline
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.OneOf([
            A.GaussNoise(p=1),
            A.GaussianBlur(p=1),
            A.MotionBlur(p=1)
        ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.3),
    ])
    
    # 创建数据集
    train_dataset = LandslideDataset(
        raster_data=raster_data,
        landslide_points=landslide_points,
        input_size=input_size,
        buffer_size=buffer_size,
        transform=transform,
        is_training=True
    )
    
    val_dataset = LandslideDataset(
        raster_data=raster_data,
        landslide_points=landslide_points,
        input_size=input_size,
        buffer_size=buffer_size,
        transform=None,
        is_training=False
    )
    
    test_dataset = LandslideDataset(
        raster_data=raster_data,
        landslide_points=landslide_points,
        input_size=input_size,
        buffer_size=buffer_size,
        transform=None,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 