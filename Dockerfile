FROM python:3.9-slim-buster
WORKDIR /app
COPY demo.py .
COPY generate_handwriting.py .
COPY api.py .
COPY drawing.py .
COPY lyrics.py .
COPY prepare_data.py .
COPY data_frame.py .
COPY rnn_cell.py .
COPY rnn_ops.py .
COPY rnn.py .
COPY setup1.py .
COPY setup2.py .
COPY tf_base_model.py .
COPY tf_utils.py .
COPY requirements3.5.txt .
COPY requirements3.9.txt .
COPY checkpoints ./checkpoints
COPY data ./data
COPY styles ./styles


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y wget
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda init bash
RUN conda config --set always_yes yes --set changeps1 no
RUN conda update -q conda
RUN conda create -n python3.5 python=3.5
RUN /bin/bash -l -c "conda init bash && conda activate python3.5"

RUN conda run -n python3.5 apt-get install -y build-essential
RUN conda run -v -n python3.5 pip install -r requirements3.5.txt
RUN python3.9 -m pip install -r requirements3.9.txt 

EXPOSE 8080

CMD [ "python3.9", "api.py" ]