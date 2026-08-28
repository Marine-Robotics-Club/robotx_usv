from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'asv_launch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xavi',
    maintainer_email='xvicentnavarro@lssu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [''],
    },
)

