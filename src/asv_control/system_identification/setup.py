from setuptools import setup
from glob import glob

package_name = 'system_identification'

setup(
    name=package_name,
    version='0.0.1',
    description='WAM-V system identification node (tests: bollard pull, acceleration, etc.)',
    author='Your Name',
    author_email='you@example.com',
    license='Apache-2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Use glob to include all launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'system_identification = system_identification.system_identification_DD:main',
        ],
    },
)
