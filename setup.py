import os
import sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_namespace_packages

requirements = [
    "imgaug>=0.4.0",
    "matplotlib>=3.7.1",
    "scikit-learn>=1.4.0",
    "pydot>=1.4.2",
    "yacs>=0.1.8",
    "tqdm>=4.66.1",
    "scikit-image>=0.22.0",
    "edt>=2.3.2",
    "fill-voids>=2.0.6",
    "opencv-python>=4.8.0.76",
    "pandas>=1.5.3",
    "torchinfo>=1.8.0",
    "tensorboardX>=2.6.2.2",
    "h5py>=3.9.0",
    "zarr>=2.16.1",
    "bioimageio.core==0.5.9",
    "imagecodecs>=2024.1.1",
    "pytorch-msssim>=1.0.0",
]


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]


def setup_package():
    __version__ = '3.3.8'
    url = 'https://github.com/BiaPyX/BiaPy'

    setup(name='biapy',
          description='Bioimage analysis pipelines in Python',
          version=__version__,
          url=url,
          license='MIT',
          author='BiaPy Contributors',
          install_requires=requirements,
          include_dirs=getInclude(),
          packages=find_namespace_packages(),
          include_package_data=True,
          )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
