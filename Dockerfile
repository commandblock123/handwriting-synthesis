#FROM tensorflow/tensorflow:1.6.0-gpu-py3
FROM python:3.7-slim-buster
WORKDIR /app
COPY . .
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN ls
RUN python setup2.py
EXPOSE 8080