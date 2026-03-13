from setuptools import find_packages, setup

package_name = 'voice_command_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', [
            'resource/.env',
            'resource/hello_rokey_8332_32.tflite',
        ]),
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
            'vn = voice_command_node.voice_command_node:main',
        ],
    },
)
