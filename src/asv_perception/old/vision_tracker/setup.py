from setuptools import find_packages, setup

package_name = 'vision_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xavi',
    maintainer_email='xvicentnavarro@lssu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_tracker = vision_tracker.vision_tracking:main',
            'yolo_tracker = vision_tracker.yolo_tracker:main',
            'compressed_image = vision_tracker.image_transport:main',
            'tracking_pole_buoys = vision_tracker.tracking_pole_buoys:main',
            'tracking_pole_buoys_2 = vision_tracker.tracking_pole_buoys_2:main',
        ],
    },
)
