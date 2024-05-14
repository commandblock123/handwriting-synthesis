#FROM tensorflow/tensorflow:1.6.0-gpu-py3
FROM python:3.9-slim-buster
WORKDIR /app
COPY . .

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# RUN apt-get install -y python3
# RUN apt-get install -y python3-pip

RUN apt-get install -y wget
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init bash
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda create -n python3.5 python=3.5
RUN /bin/bash -l -c "conda init bash && conda activate python3.5"
#RUN conda run -n python3.5 pip install -r requirements3.5.txt

RUN conda run -n python3.5 apt-get install -y build-essential
RUN conda run -n python3.5 pip install protobuf==3.4.0
RUN conda run -n python3.5 pip install tensorflow==1.6.0

RUN conda deactivate
RUN python3.9 -m pip install -r requirements3.9.txt 


#RUN python setup2.py
EXPOSE 8080