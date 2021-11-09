FROM ubuntu:18.04

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV all_proxy=${HTTP_PROXY}

RUN apt-get update \
    && apt-get install -y wget curl git build-essential

# Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda create -n xp
SHELL ["conda", "run", "--no-capture-output", "-n", "xp", "/bin/bash", "-c"]
RUN conda install python=3.7 \
    && conda init bash \
    && echo "conda activate xp" >> ~/.bashrc

COPY . /root/JOS/
WORKDIR /root/JOS/
RUN pip install -r requirements.txt
