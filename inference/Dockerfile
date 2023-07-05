FROM pytorch/torchserve:latest-gpu

USER root

RUN apt-get update \
    && apt-get install -y curl

COPY . .

RUN pip install --no-cache-dir -r ./code/torchserve/requirements.txt