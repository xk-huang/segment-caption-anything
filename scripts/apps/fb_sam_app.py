import sys

sys.path.append(".")

import gradio as gr
from src.models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor
import torch
from PIL import Image
import requests
import numpy as np
import time
from transformers import CLIPProcessor, CLIPModel
from segment_anything import SamPredictor, sam_model_registry


cache_dir = ".cache"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam_model = "facebook/sam-vit-huge"
#  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  -O tmp/data/sam_vit_h_4b8939.pth
sam_ckpt = "tmp/data/sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](sam_ckpt)
sam = sam.to(device)
sam = SamPredictor(sam)

captioner_model = "Salesforce/blip-image-captioning-base"
clip_model = "openai/clip-vit-base-patch32"
clip = CLIPModel.from_pretrained(clip_model, cache_dir=cache_dir).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=cache_dir)
# NOTE(xiaoke): in original clip, dtype is float16, here we use float32 as hf default

dtype = clip.dtype

img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw)

NUM_OUTPUT_HEADS = 3
LIBRARIES = ["caption_mask_with_highest_iou", "multimask_output", "return_patches"]
DEFAULT_LIBRARIES = ["multimask_output", "return_patches"]


def click_and_run(input_image, args, evt: gr.SelectData):
    x, y = evt.index
    input_points = [[x, y]]
    return run(args, input_image, input_points=input_points, input_labels=[1])


def box_and_run(input_image, args, input_boxes_text):
    x, y, x2, y2 = list(map(int, input_boxes_text.split(",")))
    input_boxes = [[x, y, x2, y2]]
    return run(args, input_image, input_boxes=input_boxes)


def run(args, input_image, input_points=None, input_boxes=None, input_labels=None):
    if input_points is None and input_boxes is None:
        raise ValueError("input_points and input_boxes cannot be both None")
    if input_points is not None:
        input_points = np.array(input_points)
    if input_boxes is not None:
        input_boxes = np.array(input_boxes)

    caption_mask_with_highest_iou = "caption_mask_with_highest_iou" in args
    multimask_output = "multimask_output" in args
    return_patches = "return_patches" in args

    input_image = np.array(input_image)
    sam.set_image(input_image)
    masks, iou_predictions, low_res_masks = sam.predict(
        point_coords=input_points, box=input_boxes, point_labels=input_labels, multimask_output=multimask_output
    )

    outputs = []
    num_heads = len(masks)
    # Tuple[numpy.ndarray | PIL.Image | str, List[Tuple[numpy.ndarray | Tuple[int, int, int, int], str]]]
    # (batch_size(1), region_size(1), num_heads)
    iou_scores = iou_predictions
    for i in range(num_heads):
        output = [input_image, [[masks[i], f"iou:{iou_scores[i]:.4f}"]]]
        outputs.append(output)
    for i in range(num_heads, NUM_OUTPUT_HEADS):
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
    args = gr.CheckboxGroup(choices=LIBRARIES, value=DEFAULT_LIBRARIES, label="SAM Captioner Arguments")
    input_boxes_text = gr.Textbox(lines=1, label="Input Boxes (x,y,x2,y2)", value="0,0,100,100")
    input_boxes_button = gr.Button(value="Run with Input Boxes")

    output_images = []
    with gr.Row():
        for i in range(NUM_OUTPUT_HEADS):
            output_images.append(gr.AnnotatedImage(label=f"Output Image {i}", height=500))
    with gr.Row():
        for i in range(NUM_OUTPUT_HEADS):
            output_images.append(gr.AnnotatedImage(label=f"Output Image {i}", height=500))

    input_image.select(click_and_run, [input_image, args], [*output_images])
    input_boxes_button.click(box_and_run, [input_image, args, input_boxes_text], [*output_images])

demo.launch()
