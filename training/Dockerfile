FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN pip install tensorboardX==1.6.0
RUN mkdir data
RUN mkdir output
WORKDIR training
COPY training.py .
CMD []