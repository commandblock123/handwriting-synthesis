#FROM tensorflow/tensorflow:1.6.0-gpu-py3
FROM python:3.9-slim
WORKDIR /app
COPY . .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip


RUN apt-get update && apt-get install -y \
    wget \
    && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && \
    bash ~/miniconda.sh -b -p /usr/local \
    && \
    conda init bash \
    && \
    conda config --set always_yes yes --set changeps1 no \
    && \
    conda update -q conda \
    && \
    conda create -n myenv python=3.5 -r requirements3.5.txt \
    && \
    rm -f ~/miniconda.sh


RUN conda create -n myenv python=3.5 -r requirements3.5.txt

RUN python setup2.py
EXPOSE 8080