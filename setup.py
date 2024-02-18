from setuptools import setup, find_packages

setup(
    name='gesture_detector',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-build',
        'opencv-python',
        'mediapipe',
        'tensorflow',     
    ],
)

# To install package: pip3 install -e .