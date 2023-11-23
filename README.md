# VL-SHAP Demo

Demo for the paper: *["Interpreting Vision and Language Generative Models with Semantic Visual Priors"](https://arxiv.org/abs/2304.14986).*

---

## Installation

Clone this repo:
```bash
git clone ...
```
Install VL-SHAP
```bash
pip install git+https://github.com/michelecafagna26/vl-shap.git@adding_clipseg#egg=semshap
```
Install requirements
```bash
pip install -r requirements
```
Install [OFA](hhttps://github.com/OFA-Sys/OFA/blob/feature/add_transformers/transformers.md)

### Download the models

OFA models
```bash
git clone https://huggingface.co/OFA-Sys/OFA-tiny 
git clone https://huggingface.co/OFA-Sys/OFA-base
```

Clipseg segmentation model
```bash
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d . -j weights.zip
```

## Run it in local

```bash
gradio app.py
```
