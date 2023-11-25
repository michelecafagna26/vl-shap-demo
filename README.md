# VL-SHAP Demo

Gradio Demo for the paper: *["Interpreting Vision and Language Generative Models with Semantic Visual Priors"](https://arxiv.org/abs/2304.14986).*

---

## Run it with Docker (recommended)

Make sure to have **nvidia-docker runtime** installed.

Clone this repo:
```bash
git clone https://github.com/michelecafagna26/vl-shap-demo.git
```

Build the docker image:

```bash
docker build . -t vl-shap-demo
```

Run:
```bash
sudo docker run --runtime=nvidia --gpus all -p 7860:7860 vl-shap-demo
```

## Run it locally

### Installation

Make sure to have ```git lfs``` installed.

Clone this repo:
```bash
git clone https://github.com/michelecafagna26/vl-shap-demo.git
```
Install VL-SHAP (Clipseg branch):
```bash
pip install git+https://github.com/michelecafagna26/vl-shap.git@adding_clipseg#egg=semshap
```
Install requirements:
```bash
pip install -r requirements
```
Install the VL model [OFA](https://github.com/OFA-Sys/OFA/blob/feature/add_transformers/transformers.md):

### Download the models

OFA models
```bash
git clone https://huggingface.co/OFA-Sys/OFA-tiny 
git clone https://huggingface.co/OFA-Sys/OFA-base
```

### Run the gradio server

```bash
gradio app.py
```

## Hardware Requirements
**The use of a GPU is strongly recommended.**
Depending on VL-SHAP specific method parameters, it can requires from a few seconds to a few minutes to generate and explanation on a GPU.
This obviously depends also on the model you want to explain. For this reason the demo is currently limited to OFA-base and OFA-tiny.



