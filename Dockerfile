# Use an NVIDIA CUDA image as the base
FROM nvidia/cuda:12.6.0-devel-ubuntu20.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="${PATH}:/home/user/.local/bin"

# We love UTF!
ENV LANG C.UTF-8

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Set the nvidia container runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX 8.9"

# Install some handy tools. Even Guvcview for webcam support!
RUN set -x \
	&& apt-get update \
	&& apt-get install -y apt-transport-https ca-certificates \
	&& apt-get install -y git vim tmux nano htop sudo curl wget gnupg2 \
	&& apt-get install -y bash-completion \
	&& apt-get install -y guvcview \
	&& rm -rf /var/lib/apt/lists/* \
	&& useradd -ms /bin/bash user \
	&& echo "user:user" | chpasswd && adduser user sudo \
	&& echo "user ALL=(ALL) NOPASSWD: ALL " >> /etc/sudoers

RUN set -x \
    && apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN set -x \
    && apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && apt-get install -y python3.11-tk

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /home/user


RUN git clone https://github.com/facebookresearch/segment-anything-2 && \
    cd segment-anything-2 && \
    python3.11 -m pip install -e . -v && \
    python3.11   -m pip install -e ".[demo]" && \
    cd checkpoints && ./download_ckpts.sh && cd ..

RUN apt-get update \
 && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales
 
RUN set -x \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && apt install -y curl \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - \
    && apt-get update \
    && apt install -y ros-noetic-desktop-full
RUN set -x \
    && apt install -y python3-rosdep \
    && apt-get install -y python3-pyqt5 \
    && pip3 install shiboken2
RUN set -x \
    && rosdep init \
    && rosdep fix-permissions \
    && apt install -y gnome-terminal \
    && apt install -y libgirepository1.0-dev \
    && apt remove -y python3-pycryptodome

RUN usermod -aG dialout user
USER user

RUN rosdep update
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN sudo apt-get install -y python3-pip
RUN python3 -m pip install PyGObject --force-reinstall
RUN pip3 install pycryptodome
RUN sudo apt install -y ros-noetic-desktop-full

STOPSIGNAL SIGTERM


CMD sudo service ssh start && /bin/bash
