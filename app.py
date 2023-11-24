import gradio as gr
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import OFATokenizer, OFAModel

from semshap.masking import generate_dff_masks, generate_superpixel_masks, generate_segmentation_masks
from semshap.plot import heatmap
from semshap.explainers import BaseExplainer

device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (512, 512)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LonelyModel(metaclass=Singleton):
    def __init__(self, query, ckpt_dir, device="cpu"):
        model = OFAModel.from_pretrained(ckpt_dir, use_cache=False).to(device)
        tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
        self.model_wrapper = ModelWrapper(model, tokenizer, query, resolution=IMG_SIZE, device=device)


class ModelWrapper(nn.Module):
    def __init__(self, model, tokenizer, query, resolution, device="cpu"):
        super().__init__()

        self.resolution = resolution
        self.num_beams = 5
        self.no_repeat_ngram_size = 3
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize(self.resolution, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.inputs = tokenizer([query], return_tensors="pt").input_ids.to(self.device)

    def forward(self, img):
        # put here all to code to generate a caption from an image

        patch_img = self.patch_resize_transform(img).unsqueeze(0).to(self.device)
        out_ids = self.model.generate(self.inputs, patch_images=patch_img, num_beams=self.num_beams,
                                      no_repeat_ngram_size=self.no_repeat_ngram_size)

        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

def get_num_samples(num_features, sampling_space):

    num_samples = int(2 ** (num_features-1) * sampling_space / 100)
    if num_samples < num_features + 2:
        num_samples = num_features + 2

    return num_samples

def explain_sp(img, query, ckpt_dir, sampling_space, rows, cols):

    lonely_model = LonelyModel(query, ckpt_dir, device=device)
    model_wrapper = lonely_model.model_wrapper

    patch_resize = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize(IMG_SIZE, interpolation=Image.BICUBIC),
    ])

    # generate Superpixel masks
    out = generate_superpixel_masks(patch_resize(img).size, grid_shape=(rows, cols))

    # compute the number of samples
    num_features = rows*cols
    num_samples = get_num_samples(num_features, sampling_space)

    explainer = BaseExplainer(model_wrapper, device=device)
    shap, base = explainer.explain(patch_resize(img), out['masks'], k=num_samples)

    # labels = [f"f_{i}" for i in range(shap.shape[0])]

    fig, ax, im, bar = heatmap(img, out['masks'], shap, alpha=0.75, vmin=-max(abs(shap)), vmax=max(abs(shap)))
    fig.canvas.draw()
    visual_explanation = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    visual_explanation = visual_explanation.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return visual_explanation

def explain_dff(img, query, ckpt_dir, sampling_space, num_features=10):

    lonely_model = LonelyModel(query, ckpt_dir, device=device)
    model_wrapper = lonely_model.model_wrapper

    # preprocess the image
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize(IMG_SIZE, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    patch_resize = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize(IMG_SIZE, interpolation=Image.BICUBIC),
    ])

    # extract CNN features from the model
    with torch.no_grad():
        visual_embeds = model_wrapper.model.encoder.embed_images(patch_resize_transform(img).unsqueeze(0).to(device))

    visual_embeds = visual_embeds.detach().cpu().squeeze(0).permute(1, 2, 0)

    # generate DFF semantic masks
    out = generate_dff_masks(visual_embeds, k=num_features, img_size=IMG_SIZE, mask_th=25, return_heatmaps=True)

    num_samples = get_num_samples(num_features, sampling_space)

    explainer = BaseExplainer(model_wrapper, device=device)
    shap, base = explainer.explain(patch_resize(img), out['masks'], k=num_samples)

    # labels = [f"f_{i}" for i in range(shap.shape[0])]

    fig, ax, im, bar = heatmap(img, out['heatmaps'], shap, alpha=0.65)
    fig.canvas.draw()
    visual_explanation = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    visual_explanation = visual_explanation.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return visual_explanation


def explain_semantic_seg(img, query, ckpt_dir, sampling_space, prompts):

    # convert a string into a list of objects/entities
    prompts = prompts.split(",")

    lonely_model = LonelyModel(query, ckpt_dir, device=device)
    model_wrapper = lonely_model.model_wrapper

    # preprocess the image
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize(IMG_SIZE, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    patch_resize = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize(IMG_SIZE, interpolation=Image.BICUBIC),
    ])

    # generate semantic masks
    out = generate_segmentation_masks(img, prompts, img_size=IMG_SIZE)

    # compute the number of samples
    num_features = len(out['masks'])
    num_samples = get_num_samples(num_features, sampling_space)

    explainer = BaseExplainer(model_wrapper, device=device)
    shap, base = explainer.explain(patch_resize(img), out['masks'], k=num_samples)

    # labels = [f"f_{i}" for i in range(shap.shape[0])]

    fig, ax, im, bar = heatmap(img, out['masks'], shap, alpha=0.75, vmin=-max(abs(shap)), vmax=max(abs(shap)))
    fig.canvas.draw()
    visual_explanation = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    visual_explanation = visual_explanation.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return visual_explanation


def generate(img, query, ckpt_dir):
    lonely_model = LonelyModel(query, ckpt_dir, device=device)

    return lonely_model.model_wrapper(img)


with gr.Blocks() as demo:
    with gr.Row():
        # Generation
        with gr.Column():
            with gr.Group("Inputs"):
                ckpt_dir = gr.Dropdown(
                    ["OFA-tiny", "OFA-base"], label="Choose the Model", info="Select the model to use", value="OFA-tiny")
                input_image = gr.Image(label="Upload an image", type='pil')
                input_text = gr.Textbox(label="Ask a question", placeholder="Ask a question...",
                                        value="What is the subject doing?")

            model_output = gr.Textbox(label="Model Output", interactive=False)
            generate_btn = gr.Button("Generate")
            generate_btn.click(fn=generate, inputs=[input_image, input_text, ckpt_dir], outputs=[model_output],
                               api_name="explain")

        # Explanation 5
        MIN_SPACE = 5
        MAX_SPACE = 100
        STEP = 1
        with gr.Tab("Deep Feature Factorization Features"):
            with gr.Column():
                sampling_space = gr.Slider(MIN_SPACE, MAX_SPACE, interactive=True, value=50, step=STEP,
                                        label="Sampling Size",
                                        info=f"Choose between {MIN_SPACE}% and {MAX_SPACE}%")
                explanation = gr.Image(label="SHAP Explanation", type='pil')

                explain_btn = gr.Button("Explain")
                explain_btn.click(fn=explain_dff, inputs=[input_image, model_output, ckpt_dir, sampling_space], outputs=explanation,
                                  api_name="explain")

        with gr.Tab("Superpixel Features"):
            with gr.Column():
                with gr.Group("Grid shape"):
                    MIN_ROWS = MIN_COLS = 2
                    MAX_ROWS = MAX_COLS = 4

                    rows = gr.Slider(MIN_ROWS, MAX_ROWS, interactive=True, value=3, step=1,
                                            label="Grid number of Rows",
                                            info=f"Choose between {MIN_ROWS} and {MAX_ROWS}")

                    cols = gr.Slider(MIN_COLS, MAX_COLS, interactive=True, value=3, step=1,
                                     label="Grid number of Columns",
                                     info=f"Choose between {MIN_COLS} and {MAX_COLS}")



                sample_space = gr.Slider(MIN_SPACE, MAX_SPACE, interactive=True, value=50, step=STEP,
                                        label="% of sampling size",
                                        info=f"Choose between {MIN_SPACE}% and {MAX_SPACE}%")

                explanation = gr.Image(label="SHAP Explanation", type='pil')

                explain_btn = gr.Button("Explain")
                explain_btn.click(fn=explain_sp, inputs=[input_image, model_output, ckpt_dir, sample_space, rows, cols], outputs=explanation,
                                  api_name="explain")


        with gr.Tab(" Semantic Segmentation Features"):
            with gr.Column():
                with gr.Group("Grid shape"):
                    sampling_space = gr.Slider(MIN_SPACE, MAX_SPACE, interactive=True, value=50, step=STEP,
                                               label="Sampling Size",
                                               info=f"Choose between {MIN_SPACE}% and {MAX_SPACE}%")
                    prompts = gr.Textbox(label="Prompt", placeholder="person,bike,sky", info="Specify the name of 3 "
                                                                                             "to 7 objects or "
                                                                                             "entities visible in the "
                                                                                             "image divided by a comma. These will be "
                                                                                             "use to extract "
                                                                                             "semantically relevant "
                                                                                             "regions")

                explanation = gr.Image(label="SHAP Explanation", type='pil')

                explain_btn = gr.Button("Explain")
                explain_btn.click(fn=explain_semantic_seg, inputs=[input_image, model_output, ckpt_dir, sampling_space, prompts],
                                  outputs=explanation,
                                  api_name="explain")

if __name__ == "__main__":

    demo.launch(show_api=False, server_name="0.0.0.0")
