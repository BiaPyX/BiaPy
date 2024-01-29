FROM nvidia/cuda:11.8.0-base-ubuntu22.04

MAINTAINER Daniel Franco-Barranco "daniel.franco@dipc.org"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 

# Download and prepare conda
RUN wget -q -P /tmp \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /installations && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /installations/miniconda3 \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh \
    && echo "export PATH="/installations/miniconda3/bin:$PATH"" >> ~/.bashrc \
    && /bin/bash -c "source ~/.bashrc"
ENV PATH /installations/miniconda3/bin:$PATH
RUN conda update --all

# Create the environment
ENV PATH="/opt/conda/bin:$PATH"
RUN . /installations/miniconda3/etc/profile.d/conda.sh
RUN conda update -qy conda \
    && conda create -n BiaPy_env python=3.8 \
    && conda init bash

# Set up shell to use the created env
SHELL ["conda", "run", "-n", "BiaPy_env", "/bin/bash", "-c"]

# Install BiaPy
RUN pip install biapy

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "BiaPy_env", "python3", "-u", "/installations/BiaPy/main.py"]
