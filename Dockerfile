#FROM tensorflow/tensorflow:1.6.0-gpu-py3
FROM ubuntu:20.04
WORKDIR /app
COPY . .
RUN apt-get update
# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip
RUN apt-get install -y python3.5
RUN apt-get install -y python3.9
RUN ls
RUN python setup2.py
EXPOSE 8080