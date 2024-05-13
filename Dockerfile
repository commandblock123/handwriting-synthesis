#FROM tensorflow/tensorflow:1.6.0-gpu-py3
FROM ubuntu:20.04
WORKDIR /app
COPY . .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip
RUN apt-get install -y python3.5
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.9
RUN apt-get install -y python3.9-pip

ENV PATH=$PATH:/usr/lib/python3.5/bin
RUN python3.5 -m ensurepip

RUN python3.5 setup2.py
EXPOSE 8080