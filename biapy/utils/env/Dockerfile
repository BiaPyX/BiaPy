FROM nvidia/cuda:11.8.0-base-ubuntu22.04

MAINTAINER Daniel Franco-Barranco "daniel.franco@dipc.org"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
	git

# Download and prepare conda
RUN wget -q -P /tmp \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /installations && \
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /installations/miniconda3 && \
    rm /tmp/Miniconda3-latest-Linux-x86_64.sh && \
    echo "export PATH=/installations/miniconda3/bin:$PATH" >> ~/.bashrc

ENV PATH /installations/miniconda3/bin:$PATH

RUN conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -y -n base -c defaults conda && \
    conda create -n BiaPy_env python=3.10 && \
    conda init bash

# Set up shell to use the created env
SHELL ["conda", "run", "-n", "BiaPy_env", "/bin/bash", "-c"]

# Install Pytorch 2.2.0 + CUDA 11.8
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118 

# Install BiaPy
RUN pip install biapy

# Clone BiaPy, as running it with python -c "" stops the container with no error after a few seconds running 
ARG REPO=https://github.com/BiaPyX/BiaPy.git
RUN git clone --depth 1 $REPO --branch $(git ls-remote --tags $REPO | cut -d/ -f3 | sort -V | tail -n1) /installations/BiaPy

# This time clone the last commit (25729c ; v3.5.2) as some minor bugs were solved but not created a new release yet
#RUN git clone --depth 1 https://github.com/BiaPyX/BiaPy.git /installations/BiaPy

# User home directory
RUN mkdir -p /home/user
ENV HOME /home/user
RUN echo 'export HOME=/home/user' >> ~/.bashrc

# To remove CuDNN issue 
WORKDIR /installations/miniconda3/envs/BiaPy_env/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/
RUN ln -s libnvrtc.so.11.2 libnvrtc.so

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BiaPy_env", "python3", "-u", "/installations/BiaPy/main.py"]
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BiaPy_env", "python3", "-u", "-c", "from biapy import BiaPy; BiaPy(..)"]
