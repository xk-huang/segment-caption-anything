import sys

sys.path.append(".")

import pytest
from PIL import Image
import requests
import torch
import time

from src.models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor

cache_dir = ".model.cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = "facebook/sam-vit-base"
captioner_model = "Salesforce/blip-image-captioning-base"


@pytest.fixture
def model():
    model = SAMCaptionerModel.from_sam_captioner_pretrained(sam_model, captioner_model, cache_dir=cache_dir).to(device)
    return model


@pytest.fixture
def processor():
    # FIXME(xiaoke): use `from_sam_captioner_pretrained`
    processor = SAMCaptionerProcessor.from_pretrained(sam_model, cache_dir=cache_dir)
    return processor


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_masks", [1, 2, 8])
# FIXME(xiaoke): no more `caption_mask_with_highest_iou`. Remove it.
@pytest.mark.parametrize("caption_mask_with_highest_iou", [False])
def test_modeling(
    batch_size,
    num_masks,
    caption_mask_with_highest_iou,
    processor: SAMCaptionerProcessor,
    model: SAMCaptionerModel,
):
    img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
    raw_image = [Image.open(requests.get(img_url, stream=True).raw).convert("RGB")]
    input_points = [[[[500, 375]], [[500, 375]]]]  # 2D location of a window in the image

    raw_image = raw_image * batch_size
    input_points[0] *= num_masks
    input_points *= batch_size

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    # warmup GPUs
    with torch.inference_mode():
        outputs = model.generate(**inputs, caption_mask_with_highest_iou=caption_mask_with_highest_iou)

    tic = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, caption_mask_with_highest_iou=caption_mask_with_highest_iou)
    toc = time.perf_counter()
    print(f"Time taken: {(toc - tic)*1000:0.4f} ms")

    print("tensor shapes")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape} {v.stride()}")

    batch_size, num_masks, num_heads, num_tokens = outputs.generate_ids.shape
    print(
        model.captioner_processor.batch_decode(outputs.generate_ids.reshape(-1, num_tokens), skip_special_tokens=True)
    )
