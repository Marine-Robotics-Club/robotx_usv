from setuptools import find_packages, setup

package_name = 'apf_grid_controller'

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
            'apf_controller_diff = apf_grid_controller.apf_controller_diff:main',
            'allocation_diff = apf_grid_controller.allocation_diff:main',
            'apf_planner = apf_grid_controller.apf_planner:main',
            'fixed_points = apf_grid_controller.fixed_points:main',
            "apf_controller_diff_vortex_D = apf_grid_controller.apf_controller_diff_vortex_D:main",
        ],
    },
)
