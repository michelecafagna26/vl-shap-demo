# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

#CPU-RUNTIME
#FROM python:3.10
#RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

#GPU-RUNTIME
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get -y update; apt-get -y install curl

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt install -y git-lfs \
    && git-lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --no-cache git+https://github.com/michelecafagna26/vl-shap.git@adding_clipseg#egg=semshap

COPY . /app

RUN cd /app \
    && pip install --no-cache -r requirements.txt \
    && git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git \
    && pip install OFA/transformers/ \
    && git clone https://huggingface.co/OFA-Sys/OFA-tiny \
    && git clone https://huggingface.co/OFA-Sys/OFA-base \
    && git clone https://huggingface.co/OFA-Sys/OFA-large

EXPOSE 7860

CMD ["gradio", "app.py"]