from setuptools import find_packages, setup

package_name = 'manager_node'

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
    maintainer='juno',
    maintainer_email='dlwnsh925@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "mt = manager_node.main_node_test:main",
            "pt = manager_node.main_node_BT_B:main",
            "main_node = manager_node.main_node:main",
            "main_test = manager_node.main_test:main",
        ],
    },
)
