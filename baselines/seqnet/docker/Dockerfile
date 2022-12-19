FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN apt-get update
RUN apt-get install -y apt-file
RUN apt-get update
RUN apt-get install -y build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    gfortran \
    libjpeg8-dev libpng-dev \
    libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev \
    libxine2-dev libv4l-dev \
    liblmdb-dev libleveldb-dev libsnappy-dev \
    mesa-utils and libgl1-mesa-glx x11-apps eog \
    vim tmux curl wget

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
RUN conda update -y -n base -c defaults conda

RUN conda install python=3.8.0
RUN conda install ninja==1.8.2 &&\
    conda install -c anaconda pillow=8.2.0 &&\
    conda install -c conda-forge ffmpeg==4.3.1
#RUN conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN conda install tqdm
RUN pip install numpy==1.19.0
RUN pip install pysimplegui einops munkres matplotlib numba shapely==1.7.1 jupyter black flake8 isort Pillow scikit-learn scipy tabulate tqdm yacs future tensorboard
RUN pip install pycocotools
RUN pip install opencv-python
