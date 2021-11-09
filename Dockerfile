FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV all_proxy=${HTTP_PROXY}

RUN apt-get update \
    && apt-get install -y wget curl git build-essential

COPY . /workspace
WORKDIR /workspace
RUN pip install -r requirements.txt
