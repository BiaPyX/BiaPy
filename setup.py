import os
import sys
import numpy as np
from distutils.sysconfig import get_python_inc
from setuptools import setup, Extension, find_packages

requirements = [
     'imgaug>=0.4.0',
     'matplotlib>=3.7.1',
     'scikit-learn>=1.2.2',
     'pydot>=1.4.2',
     'yacs>=0.1.8',
     'tqdm>=4.65.0',
     'scikit-image>=0.21.0',
     'edt>=2.3.1',
     'fill-voids>=2.0.5',
     'opencv-python>=4.7.0.72',
     'pandas>=2.0.2',
     'nvidia-cudnn-cu11==8.6.0.163',
     'tensorflow==2.12.0'
]


def getInclude():
    dirName = get_python_inc()
    return [dirName, os.path.dirname(dirName), np.get_include()]


def setup_package():
    __version__ = '3.0'
    url = 'https://github.com/danifranco/BiaPy'

    setup(name='biapy',
          description='Bioimage analysis pipelines in Python',
          version=__version__,
          url=url,
          license='MIT',
          author='BiaPy Contributors',
          install_requires=requirements,
          include_dirs=getInclude(),
          packages=find_packages(),
          )


if __name__ == '__main__':
    # pip install --editable .
    setup_package()
