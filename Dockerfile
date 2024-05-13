#FROM tensorflow/tensorflow:1.6.0-gpu-py3
FROM python:3.9-slim
WORKDIR /app
COPY . .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip

RUN apt-get install -y wget
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm -rf ~/miniconda3/miniconda.sh


RUN conda create -n myenv python=3.5 -r requirements3.5.txt

RUN python setup2.py
EXPOSE 8080