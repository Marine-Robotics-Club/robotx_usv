from setuptools import setup

package_name = 'ros2serial_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xavi',
    maintainer_email='xvicentnavarro@lssu.edu',
    description='ROS2 <-> Teensy serial bridge',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2serial_py = ros2serial_py.ros2serial:main',
            'usv_status_bridge = ros2serial_py.usv_status_bridge:main',
        ],
    },
)
