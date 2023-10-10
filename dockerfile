FROM python:3.11-slim-buster

ENC HOST=0.0.0.0
ENV LISTEN_HOST 8080
EXPOSE 8080

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y vim

RUN mkdir /root/.jupyter
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

WORKDIR /app/

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
# RUN jupyter notebook --generate-config -y