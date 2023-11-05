ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

MAINTAINER xujun@pjlab.org.cn

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="(dirname(which conda))/../"
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone
RUN pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html && pip install gradio
RUN apt-get update && apt-get install -y git && apt-get install -y libgl1-mesa-glx && apt-get -y install libglib2.0-dev
RUN pip install cmake && pip install dlib && pip install wget
RUN git clone https://github.com/open-mmlab/mmgeneration.git && pip install -r mmgeneration/requirements.txt && pip install -v -e mmgeneration/.
