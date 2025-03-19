import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, List, Tuple, Optional

class GeospatialAugmentation:
    """地理空间数据增强类"""
    
    @staticmethod
    def get_training_augmentation(
        input_size: Tuple[int, int] = (256, 256),
        p: float = 0.5
    ) -> A.Compose:
        """
        获取训练数据增强pipeline
        
        Args:
            input_size (Tuple[int, int]): 输入图像大小
            p (float): 应用概率
            
        Returns:
            A.Compose: 数据增强pipeline
        """
        return A.Compose([
            # 几何变换
            A.OneOf([
                A.RandomRotate90(p=1),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            ], p=p),
            
            # 翻转
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Transpose(p=1),
            ], p=p),
            
            # 噪声和模糊
            A.OneOf([
                A.GaussNoise(p=1),
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1),
                A.MedianBlur(blur_limit=3, p=1),
            ], p=p * 0.5),
            
            # 亮度对比度
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.HueSaturationValue(p=1),
            ], p=p * 0.5),
            
            # 空间扭曲
            A.OneOf([
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                A.PiecewiseAffine(scale=(0.01, 0.05), p=1),
            ], p=p * 0.5),
            
            # 裁剪和填充
            A.OneOf([
                A.RandomCrop(height=input_size[0], width=input_size[1], p=1),
                A.Resize(height=input_size[0], width=input_size[1], p=1),
            ], p=p),
            
            # 标准化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            
            # 转换为PyTorch张量
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_validation_augmentation(
        input_size: Tuple[int, int] = (256, 256)
    ) -> A.Compose:
        """
        获取验证数据增强pipeline
        
        Args:
            input_size (Tuple[int, int]): 输入图像大小
            
        Returns:
            A.Compose: 数据增强pipeline
        """
        return A.Compose([
            A.Resize(height=input_size[0], width=input_size[1], p=1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_test_augmentation(
        input_size: Tuple[int, int] = (256, 256)
    ) -> A.Compose:
        """
        获取测试数据增强pipeline
        
        Args:
            input_size (Tuple[int, int]): 输入图像大小
            
        Returns:
            A.Compose: 数据增强pipeline
        """
        return A.Compose([
            A.Resize(height=input_size[0], width=input_size[1], p=1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_temporal_augmentation(
        input_size: Tuple[int, int] = (256, 256),
        p: float = 0.5
    ) -> A.Compose:
        """
        获取时序数据增强pipeline
        
        Args:
            input_size (Tuple[int, int]): 输入图像大小
            p (float): 应用概率
            
        Returns:
            A.Compose: 数据增强pipeline
        """
        return A.Compose([
            # 时序变换
            A.OneOf([
                A.TimeSeriesSmooth(p=1),
                A.TimeSeriesNoise(p=1),
                A.TimeSeriesShift(p=1),
            ], p=p),
            
            # 几何变换
            A.OneOf([
                A.RandomRotate90(p=1),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=1),
            ], p=p),
            
            # 空间扭曲
            A.OneOf([
                A.GridDistortion(p=1),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
            ], p=p * 0.5),
            
            # 裁剪和填充
            A.OneOf([
                A.RandomCrop(height=input_size[0], width=input_size[1], p=1),
                A.Resize(height=input_size[0], width=input_size[1], p=1),
            ], p=p),
            
            # 标准化
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            
            # 转换为PyTorch张量
            ToTensorV2(),
        ])

class TimeSeriesSmooth(A.ImageOnlyTransform):
    """时序平滑变换"""
    
    def __init__(self, window_size: int = 3, p: float = 0.5):
        """
        初始化时序平滑变换
        
        Args:
            window_size (int): 平滑窗口大小
            p (float): 应用概率
        """
        super().__init__(p=p)
        self.window_size = window_size
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        应用平滑变换
        
        Args:
            img (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 变换后的图像
        """
        if len(img.shape) == 3:
            # 对每个通道分别进行平滑
            smoothed = np.zeros_like(img)
            for i in range(img.shape[0]):
                smoothed[i] = np.convolve(img[i].flatten(), 
                                        np.ones(self.window_size)/self.window_size, 
                                        mode='same').reshape(img[i].shape)
            return smoothed
        else:
            return np.convolve(img.flatten(), 
                             np.ones(self.window_size)/self.window_size, 
                             mode='same').reshape(img.shape)

class TimeSeriesNoise(A.ImageOnlyTransform):
    """时序噪声变换"""
    
    def __init__(self, noise_factor: float = 0.1, p: float = 0.5):
        """
        初始化时序噪声变换
        
        Args:
            noise_factor (float): 噪声因子
            p (float): 应用概率
        """
        super().__init__(p=p)
        self.noise_factor = noise_factor
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        应用噪声变换
        
        Args:
            img (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 变换后的图像
        """
        noise = np.random.normal(0, self.noise_factor, img.shape)
        return img + noise

class TimeSeriesShift(A.ImageOnlyTransform):
    """时序位移变换"""
    
    def __init__(self, shift_limit: int = 5, p: float = 0.5):
        """
        初始化时序位移变换
        
        Args:
            shift_limit (int): 位移限制
            p (float): 应用概率
        """
        super().__init__(p=p)
        self.shift_limit = shift_limit
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """
        应用位移变换
        
        Args:
            img (np.ndarray): 输入图像
            
        Returns:
            np.ndarray: 变换后的图像
        """
        shift = np.random.randint(-self.shift_limit, self.shift_limit + 1)
        return np.roll(img, shift, axis=0) 