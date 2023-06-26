FROM nvidia/cuda:11.4.0-base-ubuntu20.04

MAINTAINER Daniel Franco-Barranco "daniel.franco@dipc.org"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    graphviz \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    git 

# Download and prepare conda
RUN wget -q -P /tmp \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /installations && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /installations/miniconda3 \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && echo "export PATH="/installations/miniconda3/bin:$PATH"" >> ~/.bashrc \
    && /bin/bash -c "source ~/.bashrc"
ENV PATH /installations/miniconda3/bin:$PATH
RUN conda update --all

# Copy BiaPy
RUN git clone --depth 1 https://github.com/danifranco/BiaPy.git /installations/BiaPy

# Create the environment
ENV PATH="/opt/conda/bin:$PATH"
RUN . /installations/miniconda3/etc/profile.d/conda.sh
RUN conda update -qy conda \
    && conda create -n BiaPy_env python=3.10.10 \
    && conda init bash

# Set up shell to use the created env
SHELL ["conda", "run", "-n", "BiaPy_env", "/bin/bash", "-c"]

# Install dependencies 
RUN conda install -c conda-forge cudatoolkit=11.8.0
RUN python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.0
RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
RUN echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
RUN conda install -c nvidia cuda-nvcc=11.3.58
RUN printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
RUN source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

RUN mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
RUN cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/

RUN cd /installations/BiaPy/ && pip install --editable .

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BiaPy_env", "python3", "-u", "/installations/BiaPy/main.py"]
