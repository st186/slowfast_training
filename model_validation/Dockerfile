FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

#set up environment
RUN apt-get update
RUN apt-get install unzip
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip

ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
RUN apt-get update
RUN pip3 install --upgrade pip
COPY install.sh /app/
RUN chmod +x /app/install.sh
RUN sh install.sh

COPY requirements.txt /app/
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv

COPY . /app/

# RUN chmod +x /app/model_testing_process.sh
CMD python3 main.py