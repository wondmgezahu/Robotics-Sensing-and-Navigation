from setuptools import find_packages, setup
import os

package_name = 'gps_driver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Explicitly specify the launch file path
        ('share/' + package_name + '/launch', ['launch/gps_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wondm',
    maintainer_email='teshome.w@northeastern.edu',
    description='GPS driver package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'driver = gps_driver.driver:main',
        ],
    },
)
