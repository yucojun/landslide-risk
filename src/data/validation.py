import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from typing import Dict, List, Tuple, Optional
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataValidator:
    """数据验证器类"""
    
    def __init__(self, data_dir: str, landslide_points: str):
        """
        初始化数据验证器
        
        Args:
            data_dir (str): 数据目录路径
            landslide_points (str): 滑坡点矢量文件路径
        """
        self.data_dir = data_dir
        self.landslide_points = landslide_points
        self.raster_data = {}
        self.validation_results = {}
    
    def validate_data_integrity(self) -> Dict[str, bool]:
        """
        验证数据完整性
        
        Returns:
            Dict[str, bool]: 验证结果
        """
        results = {
            'raster_files': True,
            'vector_files': True,
            'file_permissions': True,
            'file_sizes': True
        }
        
        # 检查栅格文件
        required_rasters = [
            'dem.tif', 'slope.tif', 'aspect.tif', 'lithology.tif',
            'fault_density.tif', 'structure.tif', 'quaternary.tif',
            'mndwi.tif', 'twi.tif', 'drainage_density.tif',
            'ndvi.tif', 'ndbi.tif', 'road_density.tif',
            'population_density.tif', 'remote_sensing.tif'
        ]
        
        for raster in required_rasters:
            file_path = os.path.join(self.data_dir, raster)
            if not os.path.exists(file_path):
                results['raster_files'] = False
                print(f"Missing raster file: {raster}")
            else:
                self.raster_data[raster] = file_path
        
        # 检查矢量文件
        if not os.path.exists(self.landslide_points):
            results['vector_files'] = False
            print(f"Missing vector file: {self.landslide_points}")
        
        # 检查文件权限
        for file_path in self.raster_data.values():
            if not os.access(file_path, os.R_OK):
                results['file_permissions'] = False
                print(f"Permission denied: {file_path}")
        
        # 检查文件大小
        for file_path in self.raster_data.values():
            if os.path.getsize(file_path) == 0:
                results['file_sizes'] = False
                print(f"Empty file: {file_path}")
        
        self.validation_results['integrity'] = results
        return results
    
    def validate_spatial_consistency(self) -> Dict[str, bool]:
        """
        验证空间一致性
        
        Returns:
            Dict[str, bool]: 验证结果
        """
        results = {
            'coordinate_system': True,
            'spatial_extent': True,
            'resolution': True,
            'alignment': True
        }
        
        # 获取参考栅格（DEM）的元数据
        with rasterio.open(self.raster_data['dem.tif']) as dem:
            reference_meta = dem.meta
            reference_transform = dem.transform
            reference_crs = dem.crs
        
        # 检查坐标系统一致性
        for raster_name, raster_path in self.raster_data.items():
            with rasterio.open(raster_path) as src:
                if src.crs != reference_crs:
                    results['coordinate_system'] = False
                    print(f"Coordinate system mismatch in {raster_name}")
        
        # 检查空间范围一致性
        dem_bounds = rasterio.open(self.raster_data['dem.tif']).bounds
        for raster_name, raster_path in self.raster_data.items():
            with rasterio.open(raster_path) as src:
                if src.bounds != dem_bounds:
                    results['spatial_extent'] = False
                    print(f"Spatial extent mismatch in {raster_name}")
        
        # 检查分辨率一致性
        dem_resolution = (reference_meta['transform'][0], reference_meta['transform'][4])
        for raster_name, raster_path in self.raster_data.items():
            with rasterio.open(raster_path) as src:
                if (src.res[0] != dem_resolution[0] or src.res[1] != dem_resolution[1]):
                    results['resolution'] = False
                    print(f"Resolution mismatch in {raster_name}")
        
        # 检查栅格对齐
        for raster_name, raster_path in self.raster_data.items():
            with rasterio.open(raster_path) as src:
                if src.transform != reference_transform:
                    results['alignment'] = False
                    print(f"Alignment mismatch in {raster_name}")
        
        self.validation_results['spatial'] = results
        return results
    
    def validate_data_quality(self) -> Dict[str, Dict[str, float]]:
        """
        验证数据质量
        
        Returns:
            Dict[str, Dict[str, float]]: 验证结果
        """
        results = {}
        
        for raster_name, raster_path in self.raster_data.items():
            with rasterio.open(raster_path) as src:
                data = src.read(1)
                
                # 计算基本统计量
                stats_dict = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'median': float(np.median(data)),
                    'missing_values': float(np.sum(np.isnan(data)) / data.size),
                    'unique_values': int(len(np.unique(data)))
                }
                
                # 检查异常值
                z_scores = np.abs(stats.zscore(data[~np.isnan(data)]))
                stats_dict['outliers'] = float(np.sum(z_scores > 3) / len(z_scores))
                
                results[raster_name] = stats_dict
        
        self.validation_results['quality'] = results
        return results
    
    def validate_landslide_points(self) -> Dict[str, bool]:
        """
        验证滑坡点数据
        
        Returns:
            Dict[str, bool]: 验证结果
        """
        results = {
            'geometry_type': True,
            'attribute_fields': True,
            'spatial_distribution': True,
            'coordinate_system': True
        }
        
        try:
            gdf = gpd.read_file(self.landslide_points)
            
            # 检查几何类型
            if not all(gdf.geometry.type == 'Point'):
                results['geometry_type'] = False
                print("Invalid geometry type in landslide points")
            
            # 检查属性字段
            required_fields = ['id', 'date', 'type', 'size']
            missing_fields = [field for field in required_fields if field not in gdf.columns]
            if missing_fields:
                results['attribute_fields'] = False
                print(f"Missing required fields: {missing_fields}")
            
            # 检查空间分布
            if len(gdf) < 10:
                results['spatial_distribution'] = False
                print("Insufficient number of landslide points")
            
            # 检查坐标系统
            with rasterio.open(self.raster_data['dem.tif']) as dem:
                if gdf.crs != dem.crs:
                    results['coordinate_system'] = False
                    print("Coordinate system mismatch between landslide points and raster data")
        
        except Exception as e:
            print(f"Error validating landslide points: {str(e)}")
            results = {k: False for k in results}
        
        self.validation_results['landslides'] = results
        return results
    
    def generate_validation_report(self, output_dir: str):
        """
        生成验证报告
        
        Args:
            output_dir (str): 输出目录路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成HTML报告
        html_content = """
        <html>
        <head>
            <title>数据验证报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .pass { color: green; }
                .fail { color: red; }
            </style>
        </head>
        <body>
            <h1>数据验证报告</h1>
        """
        
        # 添加完整性检查结果
        html_content += "<h2>数据完整性检查</h2>"
        html_content += "<table>"
        for key, value in self.validation_results['integrity'].items():
            status = "pass" if value else "fail"
            html_content += f"<tr><td>{key}</td><td class='{status}'>{value}</td></tr>"
        html_content += "</table>"
        
        # 添加空间一致性检查结果
        html_content += "<h2>空间一致性检查</h2>"
        html_content += "<table>"
        for key, value in self.validation_results['spatial'].items():
            status = "pass" if value else "fail"
            html_content += f"<tr><td>{key}</td><td class='{status}'>{value}</td></tr>"
        html_content += "</table>"
        
        # 添加数据质量检查结果
        html_content += "<h2>数据质量检查</h2>"
        html_content += "<table>"
        html_content += "<tr><th>Raster</th><th>Mean</th><th>Std</th><th>Missing Values</th><th>Outliers</th></tr>"
        for raster_name, stats in self.validation_results['quality'].items():
            html_content += f"""
            <tr>
                <td>{raster_name}</td>
                <td>{stats['mean']:.2f}</td>
                <td>{stats['std']:.2f}</td>
                <td>{stats['missing_values']:.2%}</td>
                <td>{stats['outliers']:.2%}</td>
            </tr>
            """
        html_content += "</table>"
        
        # 添加滑坡点检查结果
        html_content += "<h2>滑坡点数据检查</h2>"
        html_content += "<table>"
        for key, value in self.validation_results['landslides'].items():
            status = "pass" if value else "fail"
            html_content += f"<tr><td>{key}</td><td class='{status}'>{value}</td></tr>"
        html_content += "</table>"
        
        html_content += "</body></html>"
        
        # 保存报告
        with open(os.path.join(output_dir, 'validation_report.html'), 'w') as f:
            f.write(html_content)
        
        # 生成数据质量可视化
        self._generate_quality_plots(output_dir)
    
    def _generate_quality_plots(self, output_dir: str):
        """
        生成数据质量可视化图表
        
        Args:
            output_dir (str): 输出目录路径
        """
        # 创建图表目录
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 生成箱线图
        plt.figure(figsize=(15, 8))
        data = []
        labels = []
        for raster_name, stats in self.validation_results['quality'].items():
            with rasterio.open(self.raster_data[raster_name]) as src:
                data.append(src.read(1).flatten())
                labels.append(raster_name)
        
        plt.boxplot(data, labels=labels, vert=False)
        plt.title('Raster Data Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'boxplot.png'))
        plt.close()
        
        # 生成相关性热图
        plt.figure(figsize=(12, 10))
        correlation_matrix = np.corrcoef([d for d in data])
        sns.heatmap(correlation_matrix, 
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap='coolwarm',
                   center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation.png'))
        plt.close()
    
    def run_validation(self) -> bool:
        """
        运行所有验证
        
        Returns:
            bool: 验证是否通过
        """
        # 运行所有验证
        self.validate_data_integrity()
        self.validate_spatial_consistency()
        self.validate_data_quality()
        self.validate_landslide_points()
        
        # 检查是否有任何验证失败
        all_passed = True
        
        # 检查完整性
        if not all(self.validation_results['integrity'].values()):
            all_passed = False
        
        # 检查空间一致性
        if not all(self.validation_results['spatial'].values()):
            all_passed = False
        
        # 检查滑坡点数据
        if not all(self.validation_results['landslides'].values()):
            all_passed = False
        
        # 检查数据质量
        for stats in self.validation_results['quality'].values():
            if stats['missing_values'] > 0.1 or stats['outliers'] > 0.1:
                all_passed = False
                break
        
        return all_passed 