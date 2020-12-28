FROM tensorflow/tensorflow:2.4.0-gpu

RUN apt-get update && apt-get install -y

RUN apt install -y libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip

WORKDIR /tf

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir

COPY . /tf/
