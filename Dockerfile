FROM tensorflow/tensorflow:1.6.0-gpu-py3
WORKDIR /app
COPY . .
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN ls
RUN python setup2.py