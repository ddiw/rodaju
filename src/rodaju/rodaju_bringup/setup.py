from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rodaju_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juno',
    maintainer_email='dlwnsh925@gmail.com',
    description='Rodaju bringup launch package',
    license='TODO: License declaration',
    entry_points={},
)
