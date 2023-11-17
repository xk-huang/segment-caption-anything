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

cache_dir = ".model.cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = "facebook/sam-vit-huge"
# captioner_model = "Salesforce/blip-image-captioning-base"
# captioner_model = "microsoft/git-large"
captioner_model = "Salesforce/blip2-opt-2.7b"
clip_model = "openai/clip-vit-base-patch32"

img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw)

model = SAMCaptionerModel.from_sam_captioner_pretrained(sam_model, captioner_model, cache_dir=cache_dir).to(device)
processor = SAMCaptionerProcessor.from_sam_captioner_pretrained(sam_model, captioner_model, cache_dir=cache_dir)

sam_processor = processor.sam_processor
captioner_processor = processor.captioner_processor

clip = CLIPModel.from_pretrained(clip_model, cache_dir=cache_dir).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model, cache_dir=cache_dir)
# NOTE(xiaoke): in original clip, dtype is float16, here we use float32 as hf default
dtype = clip.dtype


NUM_OUTPUT_HEADS = 3
LIBRARIES = ["multimask_output", "return_patches"]
DEFAULT_LIBRARIES = ["multimask_output", "return_patches"]


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
        model_outputs = model.generate(
            **inputs,
            multimask_output=multimask_output,
            return_patches=return_patches,
            return_dict_in_generate=True,
        )
    toc = time.perf_counter()
    print(f"Time taken: {(toc - tic)*1000:0.4f} ms")

    batch_size, num_masks, num_heads, num_tokens = model_outputs.sequences.shape
    if batch_size != 1 or num_masks != 1:
        raise ValueError("batch_size and num_masks must be 1")

    captions = captioner_processor.batch_decode(
        model_outputs.sequences.reshape(-1, num_tokens), skip_special_tokens=True
    )
    masks = sam_processor.post_process_masks(
        model_outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )  # List[(num_masks, num_heads, H, W)]
    iou_scores = model_outputs.iou_scores  # (batch_size, num_masks, num_heads)
    patches = model_outputs.patches  # List[List[Image.Image]]

    outputs = []
    # Tuple[numpy.ndarray | PIL.Image | str, List[Tuple[numpy.ndarray | Tuple[int, int, int, int], str]]]
    # (batch_size(1), region_size(1), num_heads)
    iou_scores = iou_scores[0][0]
    for i in range(num_heads):
        output = [input_image, [[masks[0][:, i].cpu().numpy(), f"{captions[i]}|iou:{iou_scores[i]:.4f}"]]]
        outputs.append(output)
    for i in range(num_heads, NUM_OUTPUT_HEADS):
        output = [np.ones((1, 1)), []]
        outputs.append(output)

    if return_patches:
        # (batch_size(1), region_size(1), num_heads)
        patches = patches[0][0]
        num_patches = len(patches)
        for i in range(num_patches):
            patch = patches[i]
            caption = captions[i]
            # https://huggingface.co/openai/clip-vit-base-patch32
            clip_inputs = clip_processor(text=[caption], images=[patch], return_tensors="pt", padding=True).to(device)
            clip_outputs = clip(**clip_inputs)
            logits_per_image = clip_outputs.logits_per_image

            output = [patches[i], [[[0, 0, 0, 0], f"{caption}|clip{logits_per_image.item():.4f}"]]]
            outputs.append(output)
        for i in range(num_patches, NUM_OUTPUT_HEADS):
            output = [np.ones((1, 1)), []]
            outputs.append(output)
    else:
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
    args = gr.CheckboxGroup(choices=LIBRARIES, value=DEFAULT_LIBRARIES, label="SAM Captioner Arguments")
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

    input_image.select(
        click_and_assign,
        [args, visual_prompt_mode, input_point_text, input_boxes_text],
        [input_point_text, input_boxes_text],
    )
    input_point_button.click(point_and_run, [input_image, args, input_point_text], [*output_images])
    input_boxes_button.click(box_and_run, [input_image, args, input_boxes_text], [*output_images])

demo.launch()
