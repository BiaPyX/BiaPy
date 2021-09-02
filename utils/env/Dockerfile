ARG CUDA=11.0
FROM nvidia/cuda:${CUDA}-base
ARG CUDA

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    graphviz \
    wget \
    ffmpeg \
    libsm6 \
    libxext6

# Download and prepare conda
RUN wget -q -P /tmp \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /installations && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /installations/miniconda3 \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh \
    &&  echo "export PATH="/installations/miniconda3/bin:$PATH"" >> ~/.bashrc \
    && /bin/bash -c "source ~/.bashrc"
ENV PATH /installations/miniconda3/bin:$PATH
RUN conda update --all

# Create the environment
ENV PATH="/opt/conda/bin:$PATH"
RUN ls -al /installations/miniconda3/etc/profile.d/conda.sh
RUN . /installations/miniconda3/etc/profile.d/conda.sh
COPY utils/env/environment.yml /tmp/environment.yml
RUN conda update -qy conda \
    && conda env create -f /tmp/environment.yml \
    && conda init bash \
    && rm /tmp/environment.yml
COPY . /installations/EM_Image_Segmentation

ENTRYPOINT ["conda", "run", "-n", "EM_tools", "python3", "-u", "/installations/EM_Image_Segmentation/main.py"]
