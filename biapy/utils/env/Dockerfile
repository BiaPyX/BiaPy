FROM nvidia/cuda:11.8.0-base-ubuntu22.04

LABEL maintainer="Daniel Franco-Barranco <daniel.franco@dipc.org>"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    wget \
    libsm6 \
    libxext6 \
    git \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.11 on Ubuntu 22.04 (deadsnakes)
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Make python3/pip point to Python 3.11
RUN wget -q https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && rm -f get-pip.py && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/pip /usr/local/bin/pip3

RUN pip3 install --no-cache-dir -U pip

# Important: pin numpy<2 to avoid binary-extension breakage in many stacks
RUN pip3 install --no-cache-dir "numpy<2"

# Install PyTorch CUDA 11.8 wheels explicitly from cu118 index
RUN pip3 install --no-cache-dir \
      torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu118

# Install BiaPy (master requires Python >=3.11)
RUN pip3 install --no-cache-dir git+https://github.com/BiaPyX/BiaPy.git && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

# Don't set ENTRYPOINT to empty; keep default shell/CMD behavior
CMD ["python3"]
