from setuptools import find_packages, setup

package_name = 'execute_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/T_gripper2camera.npy']),
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
        'execute = execute_node.m0609_exec_node:main',
        "en = execute_node.execute_node:main",
        "test = execute_node.stop_test_2:main",
        "at = execute_node.action_test:main",
        "test1 = execute_node.stop_test_1:main",
        ],
    },
)
