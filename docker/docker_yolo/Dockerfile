FROM nvcr.io/nvidia/pytorch:20.12-py3

MAINTAINER Soh Yoshida <sohy@kansai-u.ac.jp>

ENV https_proxy http://proxy.itc.kansai-u.ac.jp:8080
ENV http_proxy http://proxy.itc.kansai-u.ac.jp:8080

# Specify number of CPUs can be used while building Tensorflow and OpenCV.
ARG NUM_CPUS_FOR_BUILD=4

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-dev

RUN apt-get -y clean all
RUN rm -rf /var/lib/apt/lists/*

# Install some useful and machine/deep-learning-related packages for Python3.
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install \
    opencv-python==4.6.0.66 \
	opencv-python-headless==4.6.0.66 \
	albumentations==1.0.3 \
	scipy==1.6.2 \
	thop \
	pytorch_ranger \
    adabelief-pytorch \
	einops \
	ml_collections \
	tensorboardX \
	ptflops \
	mmcv==1.3.0 \
	cvxpy \
    torchvision==0.10.0 \
	timm==0.4.9
#
ENV USER student 
ENV HOME /home/${USER}
ENV SHELL /bin/bash
ENV DISPLAY :10
#
RUN groupadd -g 1002 student
RUN useradd -g 1002 -u 1002 -m -s /bin/bash ${USER}
#
RUN gpasswd -a ${USER} sudo
#
RUN echo "${USER}:student" | chpasswd
#
RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo '${USER} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#
USER student

