from transformers import CLIPProcessor, CLIPModel
import clip
from PIL import Image
import requests
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
openai_model, openai_preprocess = clip.load("ViT-B/32", device=device)

dtype = torch.float16
cache_dir = ".cache"
hf_model_name = "openai/clip-vit-base-patch32"
hf_model = CLIPModel.from_pretrained(hf_model_name, cache_dir=cache_dir, torch_dtype=dtype).to(device)
hf_processor = CLIPProcessor.from_pretrained(hf_model_name, cache_dir=cache_dir)


img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw)
text = "a photo of a car parking on the side of the road in front of a building"


def hf_clip_inference(raw_image, text):
    inputs = hf_processor(text=text, images=raw_image, return_tensors="pt")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, dtype if v.dtype == torch.float32 else v.dtype)
    breakpoint()
    with torch.inference_mode():
        outputs = hf_model(**inputs)
    return outputs.logits_per_image, inputs["pixel_values"], inputs["input_ids"]


def openai_clip_inference(raw_image, text):
    image = openai_preprocess(raw_image).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device)
    breakpoint()

    with torch.inference_mode():
        logits_per_image, _ = openai_model(image, text)
    return logits_per_image, image, text


hf_logits, hf_image, hf_text = hf_clip_inference(raw_image, text)
openai_logits, openai_image, openai_text = openai_clip_inference(raw_image, text)
assert torch.allclose(hf_logits, openai_logits)
breakpoint()
