from setuptools import setup, find_packages

setup(
    name="w2l",
    version=0.1,
    packages=find_packages(),
    install_requires=[
        "librosa==0.10.1",
        "numpy==1.24.4",
        "opencv-contrib-python==4.8.0.76",
        "opencv-python==4.8.0.76",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "tqdm==4.66.1",
        "numba==0.57.1"
    ]
)