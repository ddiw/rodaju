from setuptools import find_packages, setup

package_name = 'vision_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/best_v1.pt']),
        ('share/' + package_name + '/resource', ['resource/best_v2.pt']),
        ('share/' + package_name + '/resource', ['resource/best_v3.pt']),
        ('share/' + package_name + '/resource', ['resource/best_v4.pt']),
        ('share/' + package_name + '/resource', ['resource/best_v5.pt']),
        ('share/' + package_name + '/resource', ['resource/best_v6.pt']),
        ('share/' + package_name + '/resource', ['resource/best_v7.pt']),
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
        'vision = vision_node.vision_node:main',
        "vt = vision_node.vision_test_pub:main"
        ],
    },
)
