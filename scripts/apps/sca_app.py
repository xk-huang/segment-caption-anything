import sys

sys.path.append(".")
sys.path.append("./scripts/notebooks")

import gradio as gr
import torch
from PIL import Image
import requests
import numpy as np
import time
import matplotlib.pyplot as plt
import io


import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from src.arguments import (
    global_setup,
)

from transformers import set_seed
from src.train import prepare_model, prepare_processor
from amcg import ScaAutomaticMaskCaptionGenerator
import dotenv

logger = logging.getLogger(__name__)

model = None
processor = None


@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(args: DictConfig) -> None:
    global model, processor
    # NOTE(xiaoke): follow https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

    logger.info(OmegaConf.to_yaml(args))
    args, training_args, model_args = global_setup(args)

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # NOTE(xiaoke): load sas_key from .env for huggingface model downloading.
    logger.info(f"Try to load sas_key from .env file: {dotenv.load_dotenv('.env')}.")
    use_auth_token = os.getenv("USE_AUTH_TOKEN", False)

    processor = prepare_processor(model_args, use_auth_token)

    model = prepare_model(model_args, use_auth_token)
    return model, processor


if __name__ == "__main__":
    main()

cache_dir = ".model.cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
# sam_model = "facebook/sam-vit-huge"
# captioner_model = "Salesforce/blip-image-captioning-base"
# captioner_model = "microsoft/git-large"
# captioner_model = "Salesforce/blip2-opt-2.7b"
clip_model = "openai/clip-vit-base-patch32"

img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw)

model = model.to(device)

# sam_processor = processor.sam_processor
# captioner_processor = processor.captioner_processor

# clip = CLIPModel.from_pretrained(clip_model, cache_dir=cache_dir).to(device)
# clip_processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=cache_dir)
# NOTE(xiaoke): in original clip, dtype is float16, here we use float32 as hf default
dtype = model.dtype


NUM_OUTPUT_HEADS = 3
LIBRARIES = ["multimask_output"]
DEFAULT_LIBRARIES = ["multimask_output"]

auto_mask_caption_generator = ScaAutomaticMaskCaptionGenerator(model, processor)


def click_and_assign(args, visual_prompt_mode, input_point_text, input_boxes_text, evt: gr.SelectData):
    x, y = evt.index
    if visual_prompt_mode == "point":
        input_point_text = f"{x},{y}"
    elif visual_prompt_mode == "box":
        if len(input_boxes_text.split(",")) == 2:
            input_boxes_text = f"{input_boxes_text},{x},{y}"
        else:
            input_boxes_text = f"{x},{y}"
    return input_point_text, input_boxes_text


def box_and_run(input_image, args, input_boxes_text):
    x, y, x2, y2 = list(map(int, input_boxes_text.split(",")))
    input_boxes = [[[x, y, x2, y2]]]
    return run(args, input_image, input_boxes=input_boxes)


def point_and_run(input_image, args, input_point_text):
    x, y = list(map(int, input_point_text.split(",")))
    input_points = [[[[x, y]]]]
    return run(args, input_image, input_points=input_points)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        if "caption" in ann:
            captions: str = ann["caption"]
            # calculate the centroid of the mask
            y, x = np.where(m)
            random_index = np.random.choice(range(len(x)))
            random_position = (x[random_index], y[random_index])
            # display the caption at the centroid of the mask
            ax.text(*random_position, captions, color="white", fontsize=12, ha="center", va="center")
    ax.imshow(img)


def auto_mode(input_image):
    np_input_image = np.array(input_image)
    outputs = auto_mask_caption_generator.generate(np_input_image)

    dpi = 80
    height, width, _ = np_input_image.shape
    figsize = width / float(dpi), height / float(dpi)

    plt.figure(figsize=figsize)
    plt.imshow(input_image)
    show_anns(outputs)
    plt.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    return img


def run(args, input_image, input_points=None, input_boxes=None):
    if input_points is None and input_boxes is None:
        raise ValueError("input_points and input_boxes cannot be both None")

    multimask_output = "multimask_output" in args
    return_patches = "return_patches" in args

    inputs = processor(input_image, input_points=input_points, input_boxes=input_boxes, return_tensors="pt")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            # NOTE(xiaoke): in original clip, dtype is float16
            inputs[k] = v.to(device, dtype if v.dtype == torch.float32 else v.dtype)
    tic = time.perf_counter()
    with torch.inference_mode():
        model_outputs = model.generate(**inputs, multimask_output=multimask_output, num_beams=3)
    toc = time.perf_counter()
    print(f"Time taken: {(toc - tic)*1000:0.4f} ms")

    batch_size, num_masks, num_text_heads, num_tokens = model_outputs.sequences.shape
    batch_size, num_masks, num_mask_heads, *_ = model_outputs.pred_masks.shape
    if batch_size != 1 or num_masks != 1:
        raise ValueError("batch_size and num_masks must be 1")

    captions = processor.tokenizer.batch_decode(
        model_outputs.sequences.reshape(-1, num_tokens), skip_special_tokens=True
    )
    # NOTE: sometimes, num_text_heads < num_mask_heads, as we have split the text head with the mask head in SCA.
    if num_text_heads < num_mask_heads:
        captions += [captions[-1]] * (num_mask_heads - num_text_heads)

    masks = processor.post_process_masks(
        model_outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )  # List[(num_masks, num_heads, H, W)]
    iou_scores = model_outputs.iou_scores  # (batch_size, num_masks, num_heads)
    # patches = model_outputs.patches  # List[List[Image.Image]]
    patches = None

    outputs = []
    # Tuple[numpy.ndarray | PIL.Image | str, List[Tuple[numpy.ndarray | Tuple[int, int, int, int], str]]]
    # (batch_size(1), region_size(1), num_heads)
    iou_scores = iou_scores[0][0]
    for i in range(num_mask_heads):
        output = [input_image, [[masks[0][:, i].cpu().numpy(), f"{captions[i]}|iou:{iou_scores[i]:.4f}"]]]
        outputs.append(output)
    for i in range(num_mask_heads, NUM_OUTPUT_HEADS):
        output = [np.ones((1, 1)), []]
        outputs.append(output)

        # if return_patches:
        #     # (batch_size(1), region_size(1), num_heads)
        #     patches = patches[0][0]
        #     num_patches = len(patches)
        #     for i in range(num_patches):
        #         patch = patches[i]
        #         caption = captions[i]
        #         # https://huggingface.co/openai/clip-vit-base-patch32
        #         clip_inputs = clip_processor(text=[caption], images=[patch], return_tensors="pt", padding=True).to(device)
        #         clip_outputs = clip(**clip_inputs)
        #         logits_per_image = clip_outputs.logits_per_image

        #         output = [patches[i], [[[0, 0, 0, 0], f"{caption}|clip{logits_per_image.item():.4f}"]]]
        #         outputs.append(output)
        #     for i in range(num_patches, NUM_OUTPUT_HEADS):
        #         output = [np.ones((1, 1)), []]
        #         outputs.append(output)
        # else:
    for i in range(NUM_OUTPUT_HEADS):
        output = [np.ones((1, 1)), []]
        outputs.append(output)
    return outputs


def fake_click_and_run(input_image, args, evt: gr.SelectData):
    outputs = []
    # Tuple[numpy.ndarray | PIL.Image | str, List[Tuple[numpy.ndarray | Tuple[int, int, int, int], str]]]
    num_heads = 1
    for i in range(num_heads):
        output = [input_image, []]
        outputs.append(output)
    for i in range(num_heads, NUM_OUTPUT_HEADS):
        output = [input_image, []]
        outputs.append(output)
    return outputs


with gr.Blocks() as demo:
    gr.Markdown("Welcome to the SCA Demo! We have two modes: **Prompt Mode** and **Anything Mode**.")

    input_image = gr.Image(value=raw_image, label="Input Image", interactive=True, type="pil", height=500)

    with gr.Tab("Prompt Mode"):
        visual_prompt_mode = gr.Radio(choices=["point", "box"], value="point", label="Visual Prompt Mode")
        args = gr.CheckboxGroup(choices=LIBRARIES, value=DEFAULT_LIBRARIES, label="SCA Arguments")
        input_point_text = gr.Textbox(lines=1, label="Input Points (x,y)", value="0,0")
        input_point_button = gr.Button(value="Run with Input Points")
        input_boxes_text = gr.Textbox(lines=1, label="Input Boxes (x,y,x2,y2)", value="0,0,100,100")
        input_boxes_button = gr.Button(value="Run with Input Boxes")

        output_images = []
        with gr.Row():
            for i in range(NUM_OUTPUT_HEADS):
                output_images.append(gr.AnnotatedImage(label=f"Output Image {i}", height=500))
        with gr.Row():
            for i in range(NUM_OUTPUT_HEADS):
                output_images.append(gr.AnnotatedImage(label=f"Output Image {i}", height=500))
    with gr.Tab("Anything Mode"):
        run_anything_mode_button = gr.Button(value="Run Anything Mode")
        output_image_for_anything_mode = gr.Image(
            value=raw_image, label="Output Image", interactive=False, type="pil", height=500
        )

    input_image.select(
        click_and_assign,
        [args, visual_prompt_mode, input_point_text, input_boxes_text],
        [input_point_text, input_boxes_text],
    )
    input_point_button.click(point_and_run, [input_image, args, input_point_text], [*output_images])
    input_boxes_button.click(box_and_run, [input_image, args, input_boxes_text], [*output_images])

    run_anything_mode_button.click(auto_mode, [input_image], [output_image_for_anything_mode])

demo.launch()
