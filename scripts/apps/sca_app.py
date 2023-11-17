import sys

sys.path.append(".")

import gradio as gr
from src.models.sam_captioner import SAMCaptionerProcessor
import torch
from PIL import Image
import requests
import numpy as np
import time


import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from src.arguments import (
    global_setup,
    SAMCaptionerModelArguments,
    SCAModelBaseArguments,
)
from src.models.sam_captioner import SAMCaptionerProcessor
from src.models.sca import ScaProcessor

from transformers import set_seed
from src.train import prepare_model

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

    if isinstance(model_args, SAMCaptionerModelArguments):
        processor = SAMCaptionerProcessor.from_sam_captioner_pretrained(
            model_args.sam_model_name_or_path,
            model_args.captioner_model_name_or_path,
            cache_dir=model_args.cache_dir,
            model_max_length=model_args.model_max_length,
        )
    elif isinstance(model_args, SCAModelBaseArguments):
        processor = ScaProcessor.from_sam_text_pretrained(
            model_args.sam_model_name_or_path,
            model_args.lm_head_model_name_or_path,
            cache_dir=model_args.cache_dir,
            model_max_length=model_args.model_max_length,
        )
    else:
        raise ValueError(
            f"model_args must be one of [SAMCaptionerModelArguments, SCAModelBaseArguments], got {type(model_args)}"
        )
    # NOTE(xiaoke): add pad_token if not exists
    if processor.tokenizer.pad_token is None:
        if processor.tokenizer.eos_token is None:
            raise ValueError("tokenizer must have either eos_token")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = prepare_model(model_args)
    return model, processor


if __name__ == "__main__":
    main()

cache_dir = ".model.cache"
device = "cuda" if torch.cuda.is_available() else "cpu"

img_url = "https://segment-anything.com/assets/gallery/AdobeStock_94274587_welsh_corgi_pembroke_CD.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw)

model = model.to(device)
dtype = model.dtype


NUM_OUTPUT_HEADS = 3
LIBRARIES = ["multimask_output"]
DEFAULT_LIBRARIES = ["multimask_output"]


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
    input_image = gr.Image(value=raw_image, label="Input Image", interactive=True, type="pil", height=500)
    visual_prompt_mode = gr.Radio(choices=["point", "box"], value="point", label="Visual Prompt Mode")
    args = gr.CheckboxGroup(choices=LIBRARIES, value=DEFAULT_LIBRARIES, label="SCA Arguments")
    input_point_text = gr.Textbox(lines=1, label="Input Points (x,y)", value="0,0")
    input_point_button = gr.Button(value="Run with Input Points")
    input_boxes_text = gr.Textbox(lines=1, label="Input Boxes (x,y,x2,y2)", value="0,0,100,100")
    input_boxes_button = gr.Button(value="Run with Input Boxes")

    output_images = []
    for i in range(NUM_OUTPUT_HEADS):
        output_images.append(gr.AnnotatedImage(label=f"Output Image {i}", height=500))
    for i in range(NUM_OUTPUT_HEADS):
        output_images.append(gr.AnnotatedImage(label=f"Output Image {i}", height=500))

    input_image.select(
        click_and_assign,
        [args, visual_prompt_mode, input_point_text, input_boxes_text],
        [input_point_text, input_boxes_text],
    )
    input_point_button.click(point_and_run, [input_image, args, input_point_text], [*output_images])
    input_boxes_button.click(box_and_run, [input_image, args, input_boxes_text], [*output_images])

demo.launch()
