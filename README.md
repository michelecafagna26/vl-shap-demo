# VL-SHAP Demo

🚀 Gradio Demo for the paper: *["Interpreting Vision and Language Generative Models with Semantic Visual Priors"](https://arxiv.org/abs/2304.14986).*

- **🗃️ Repository:** [github.com/michelecafagna26/vl-shap](https://github.com/michelecafagna26/vl-shap)
- **📜 Paper:** [Interpreting Vision and Language Generative Models with Semantic Visual Priors](https://arxiv.org/abs/2304.14986)
- **🚀 Gradio Demo:** [michelecafagna26/vl-shap-demo](https://github.com/michelecafagna26/vl-shap-demo)
- **🖊️ Contact:** michele.cafagna@um.edu.mt

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
Go to ```http://0.0.0.0:7860``` from your browser to play 🎮 with the demo.

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
pip install -r requirements.txt
```
Install the VL model [OFA](https://github.com/OFA-Sys/OFA/blob/feature/add_transformers/transformers.md):

### Download the models to explain

OFA models
```bash
git clone https://huggingface.co/OFA-Sys/OFA-tiny 
git clone https://huggingface.co/OFA-Sys/OFA-base
git clone https://huggingface.co/OFA-Sys/OFA-large
```

### Run the gradio server

```bash
gradio app.py
```
## Note ⚠️
The visual explanation generation may take a while. Check out the logs check the progress of the algorithm.

## Hardware Requirements
**The use of a GPU is strongly recommended.**
Depending on VL-SHAP parameters and visual feature extraction method, it can require from a few seconds to a few minutes to generate and explanation on a GPU.
This obviously depends also on the model you want to explain. For this reason the demo is currently limited to OFA-base and OFA-tiny.



