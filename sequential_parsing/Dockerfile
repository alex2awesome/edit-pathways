#FROM ubuntu:18.04
FROM gcr.io/google.com/cloudsdktool/cloud-sdk:latest
RUN apt-get clean
RUN apt-get update \
    && apt-get install vim -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && apt-get autoremove

#
ADD ../util /app/util
ADD requirements.txt /app/
ADD parsing_script.py /app/
WORKDIR /app

## authenticate
ADD usc-research-data-access.json /app/creds.json
RUN gcloud auth activate-service-account --key-file=/app/creds.json
RUN gcloud config set project usc-research
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/creds.json

RUN pip3 install -r /app/requirements.txt
RUN python3 -m spacy download en_core_web_lg