# Written by Seongmoon Jeong - 2022.08.22

FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Asia/Seoul

# Temporarily added for issue.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Use kakao mirror
RUN sed -i 's@archive.ubuntu.com@mirror.kakao.com@g' /etc/apt/sources.list

# Install linux programs.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    ca-certificates lsb-release git \
    protobuf-compiler ninja-build \
    tree curl wget nano vim htop screen rsync default-jdk 

# Install latest cmake.
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt-get install -y kitware-archive-keyring
RUN rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-get install -y cmake

# Install python3 & deps.
ENV PYTHON3_VERSION="3.8"
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y \
    python${PYTHON3_VERSION}-dev python3-pip python3-opencv python3-lxml \
    python-is-python3

# Install pytorch & detectron2.
ENV DETECTRON2_VERSION="0.6"
COPY install_detectron2.sh /install_detectron2.sh
RUN ./install_detectron2.sh ${DETECTRON2_VERSION}
RUN rm /install_detectron2.sh

# Install object_dection API.
RUN pip install tensorflow-cpu==2.8.1
WORKDIR /root
RUN git clone https://github.com/tensorflow/models.git
RUN cd /root/models/research/ && protoc object_detection/protos/*.proto --python_out=.
WORKDIR /root/models/research
RUN cp object_detection/packages/tf2/setup.py .
RUN pip install --no-deps .

# Install others.
RUN python -m pip install \
    seaborn ray[default]==2.0.0 compressai fiftyone hydra-core==1.2.0

# Install FFmpeg
RUN add-apt-repository ppa:savoury1/ffmpeg4
RUN apt-get install -y ffmpeg

# Set config file directory.
ENV CONFIG_PATH="/usr/local/etc"

# Install VTM
ENV VTM_VERSION="12.0"
ENV VTM_CFG_NAME="encoder_intra_vtm.cfg"
ENV VTM_CFG_PATH=${CONFIG_PATH}/${VTM_CFG_NAME}
WORKDIR /root
RUN git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
RUN cd VVCSoftware_VTM && git checkout tags/VTM-${VTM_VERSION} && mkdir build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
RUN mv /root/VVCSoftware_VTM/bin/EncoderAppStatic /usr/bin/vtm
RUN mv /root/VVCSoftware_VTM/cfg/${VTM_CFG_NAME} ${VTM_CFG_PATH}

# Install VVenC.
ENV VVENC_VERSION="1.5.0"
ENV VVENC_ORG_CFG_NAME="randomaccess_medium.cfg"
ENV VVENC_CFG_NAME="encoder_intra_vvenc.cfg"
ENV VVENC_ORG_CFG_PATH=${CONFIG_PATH}/${VVENC_ORG_CFG_NAME}
ENV VVENC_CFG_PATH=${CONFIG_PATH}/${VVENC_CFG_NAME}
WORKDIR /root
RUN git clone https://github.com/fraunhoferhhi/vvenc.git
RUN cd vvenc && git checkout tags/v${VVENC_VERSION} && make install-release
RUN mv /root/vvenc/bin/release-static/vvencFFapp /usr/bin/vvencFFapp
RUN mv /root/vvenc/cfg/${VVENC_ORG_CFG_NAME} ${VVENC_ORG_CFG_PATH}

# Create VVenC config for intra mode.
COPY create_vvenc_cfg.py /create_vvenc_cfg.py
RUN python /create_vvenc_cfg.py ${VVENC_ORG_CFG_PATH} ${VTM_CFG_PATH} ${VVENC_CFG_PATH}
RUN rm /create_vvenc_cfg.py

# Due to torch tensorboard issue.
RUN pip install setuptools==59.5.0

# Set working directory.
WORKDIR /filtering

# Copy python files.
COPY *.py .
