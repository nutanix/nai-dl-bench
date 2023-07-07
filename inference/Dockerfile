FROM pytorch/torchserve:0.8.1-gpu

USER root

RUN apt-get update \
    && apt-get install -y curl

COPY . .