FROM  nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

MAINTAINER Daniel Franco-Barranco "daniel.franco@dipc.org"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        python3-pip \
        python3 \
    	wget \
	    libsm6 \
    	libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
    torch \
    --extra-index-url https://download.pytorch.org/whl/cu121 && \
	pip install git+https://github.com/BiaPyX/BiaPy.git && \
    rm -rf /root/.cache/ && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/*

ENTRYPOINT [""]
