FROM ubuntu:18.04
RUN apt-get update \
    && apt-get install tesseract-ocr poppler-utils vim -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && apt-get autoremove

ADD . /app
WORKDIR /app

RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_lg