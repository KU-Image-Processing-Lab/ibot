#FROM nvcr.io/nvidia/pytorch:20.12-py3
#FROM nvcr.io/nvidia/pytorch:19.09-py3
FROM nvcr.io/nvidia/pytorch:22.08-py3

MAINTAINER Soh Yoshida <sohy@kansai-u.ac.jp>

ENV https_proxy http://proxy.itc.kansai-u.ac.jp:8080
ENV http_proxy http://proxy.itc.kansai-u.ac.jp:8080

# Specify number of CPUs can be used while building Tensorflow and OpenCV.
ARG NUM_CPUS_FOR_BUILD=4

RUN echo 'Acquire::http::proxy "http://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::https::proxy "https://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::ftp::proxy "ftp://proxy.itc.kansai-u.ac.jp:8080/";\nAcquire::socks::proxy "socks://proxy.itc.kansai-u.ac.jp:8080/";' >> /etc/apt/apt.conf

RUN echo 'http_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nhttps_proxy="https://proxy.itc.kansai-u.ac.jp:8080/"\nftp_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"\nsocks_proxy="http://proxy.itc.kansai-u.ac.jp:8080/"' >> /etc/environment

RUN echo 'https_proxy = http://proxy.itc.kansai-u.ac.jp:8080/\nhttp_proxy = http://proxy.itc.kansai-u.ac.jp:8080/\nftp_proxy = http://proxy.itc.kansai-u.ac.jp:8080/' >>/etc/wgetrc

RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.jaist.ac.jp/pub/Linux/ubuntu/%g" /etc/apt/sources.list

# Install some useful and machine/deep-learning-related packages for Python3.
RUN pip install \
        opencv-python==4.6.0.66 \
        opencv-python-headless==4.6.0.66 \
	albumentations==1.0.3 \
	scipy \
	thop \
	pytorch_ranger \
        adabelief-pytorch \
	einops \
	timm==0.3.2 \
	ml_collections \
	tensorboardX \
	ptflops \
	mmcv==1.3.0 \
	cvxpy \
	PySnooper==1.1.1 \
	TorchSnooper \
        
#
ENV USER student 
ENV HOME /home/${USER}
ENV SHELL /bin/bash

#
RUN groupadd -g 1002 student
RUN useradd -g 1002 -u 1002 -m -s /bin/bash ${USER}
#
RUN gpasswd -a ${USER} sudo
#
RUN echo "${USER}:student" | chpasswd

RUN echo 'Defaults visiblepw'             >> /etc/sudoers
RUN echo '${USER} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER student

