from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robust_buoy_mapping'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='highlevel',
    maintainer_email='highlevel@example.com',
    description='Dynamic KF semantic buoy mapping with LiDAR-camera buoy-detection fusion for USV navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dynamic_kf_ai_assisted_mapper_node = robust_buoy_mapping.dynamic_kf_ai_assisted_mapper_node:main',
            'train_self_supervised_ai_models = robust_buoy_mapping.train_self_supervised_ai_models:main',
            'build_self_supervised_ai_dataset = robust_buoy_mapping.build_self_supervised_ai_dataset:main',
            'dynamic_kf_ai_quality_logger_node = robust_buoy_mapping.dynamic_kf_ai_quality_logger_node:main',
            'train_zed_kf_teacher_model = robust_buoy_mapping.train_zed_kf_teacher_model:main',
            'dynamic_kf_ai_logger_node = robust_buoy_mapping.dynamic_kf_ai_logger_node:main',
            'dynamic_kf_buoy_mapper_node = robust_buoy_mapping.dynamic_kf_buoy_mapper_node:main',
            'gt_eval_node = robust_buoy_mapping.gt_eval_node:main',
            'lidar_camera_fusion_logger_node = robust_buoy_mapping.lidar_camera_fusion_logger_node:main',
            'train_lidar_camera_fusion_xgboost = robust_buoy_mapping.train_lidar_camera_fusion_xgboost:main',
            'lidar_camera_fusion_inference_node = robust_buoy_mapping.lidar_camera_fusion_inference_node:main',
            'fused_dynamic_kf_ai_mapper_node = robust_buoy_mapping.fused_dynamic_kf_ai_mapper_node:main',
            'fused_dynamic_kf_ai_quality_logger_node = robust_buoy_mapping.fused_dynamic_kf_ai_quality_logger_node:main',
            'lidar_camera_fusion_results_logger_node = robust_buoy_mapping.lidar_camera_fusion_results_logger_node:main',
            
        ],
    },
)
