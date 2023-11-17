import sys

sys.path.append(".")

import pytest
from PIL import Image
import requests
import torch
import time

from src.models.sca import ScaConfig, ScaModel, ScaProcessor
from typing import Sequence
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import transformers

cache_dir = ".model.cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model_name = "facebook/sam-vit-base"
text_model_name = "gpt2"
additional_num_hidden_layers = 2


@pytest.fixture
def model():
    model = ScaModel.from_sam_text_pretrained(
        sam_model_name, text_model_name, additional_num_hidden_layers, cache_dir=cache_dir
    ).to(device)
    return model


@pytest.fixture
def processor():
    processor = ScaProcessor.from_sam_text_pretrained(sam_model_name, text_model_name, cache_dir=cache_dir)
    return processor


@pytest.fixture
def sam_model():
    model = transformers.AutoModel.from_pretrained(sam_model_name, cache_dir=cache_dir).to(device)
    return model


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_masks", [4, 7])
def test_modeling(batch_size, num_masks, model, processor):
    img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
    raw_image = [Image.open(requests.get(img_url, stream=True).raw).convert("RGB")]
    input_points = [[[[500, 375]]]]  # 2D location of a window in the image
    raw_text = [["This is a test sentence."]]

    raw_image = raw_image * batch_size

    input_points = np.array(input_points)
    raw_text = np.array(raw_text, dtype=object)
    input_points = input_points.repeat(batch_size, axis=0).repeat(num_masks, axis=1).tolist()
    raw_text = raw_text.repeat(batch_size, axis=0).repeat(num_masks, axis=1).reshape(-1).tolist()

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt")

    # prepare tokenizer
    tokenizer = processor.tokenizer
    raw_text_inputs = tokenizer(raw_text)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer does not have an eos token id")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    label_pad_token_id = -100
    # get tokenized inputs
    tokenized_inputs = tokenizer(raw_text)
    raw_input_ids = tokenized_inputs["input_ids"]
    raw_attention_mask = tokenized_inputs["attention_mask"]
    # add eos token
    for i in range(len(raw_input_ids)):
        raw_input_ids[i] += [eos_token_id]
        raw_attention_mask[i] += [1]
    # trim to max length
    max_length = tokenizer.model_max_length
    for i in range(len(raw_input_ids)):
        raw_input_ids[i] = raw_input_ids[i][:max_length]
        raw_attention_mask[i] = raw_attention_mask[i][:max_length]
    # right pad and get batch of data
    input_ids = pad_sequence([torch.tensor(x) for x in raw_input_ids], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([torch.tensor(x) for x in raw_attention_mask], batch_first=True, padding_value=0)
    # get label and left pad the label by 1, to avoid insert BOS token
    labels = pad_sequence([torch.tensor(x) for x in raw_input_ids], batch_first=True, padding_value=label_pad_token_id)
    labels = torch.nn.functional.pad(labels, (1, 0), value=label_pad_token_id)
    # get text inputs
    text_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    for k in text_inputs:
        text_inputs[k] = text_inputs[k].view(batch_size, num_masks, -1)

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, Sequence):
            print(k, "sequence of ", type(v[0]), len(v))
        else:
            print(k, type(v), len(v))
    for k, v in text_inputs.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, Sequence):
            print(k, "sequence of ", type(v[0]), len(v))
        else:
            print(k, type(v), len(v))

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    for k, v in text_inputs.items():
        if isinstance(v, torch.Tensor):
            text_inputs[k] = v.to(device)

    # test training
    model.train()
    outputs = model(**inputs, **text_inputs)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
    sequence_texts = tokenizer.batch_decode(outputs["logits"].argmax(dim=-1))
    sequence_texts = sequence_texts[:1]
    print(sequence_texts)

    # test inference
    model.eval()
    with torch.no_grad():
        outputs = model.generate(**inputs, **text_inputs)
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
    outputs["sequences"] = outputs["sequences"].view(-1, outputs["sequences"].shape[-1])
    sequence_texts = tokenizer.batch_decode(outputs["sequences"])
    sequence_texts = sequence_texts[:1]
    print(sequence_texts)

    # batch_size, num_masks, num_output_heads, 1, hidden_size -> 1, 1, hidden_size
    inputs_embeds = outputs["projected_query_logits"][0, 0, 0:1]
    inputs_ids = torch.tensor([[tokenizer.eos_token_id]]).to(device)
    attention_masks = torch.tensor([[1]]).to(device)

    language_model = transformers.AutoModelForCausalLM.from_pretrained(
        text_model_name, config=model.config.text_config, cache_dir=cache_dir
    ).to(device)
    language_model.eval()
    with torch.no_grad():
        original_output = language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_masks)
        sca_text_output = model.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_masks)
    assert torch.allclose(original_output, sca_text_output)

    language_model = transformers.AutoModelForCausalLM.from_pretrained(text_model_name, cache_dir=cache_dir).to(device)
    language_model.eval()
    with torch.no_grad():
        original_output = language_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_masks)
    assert torch.allclose(original_output, sca_text_output)

    validate_texts = tokenizer.batch_decode(sca_text_output)
    validate_texts = validate_texts[:1]
    print(validate_texts)

    assert validate_texts[0] == sequence_texts[0]


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_masks", [4, 7])
def test_modeling_with_sam(batch_size, num_masks, model, sam_model, processor):
    img_url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"
    raw_image = [Image.open(requests.get(img_url, stream=True).raw).convert("RGB")]
    input_points = [[[[500, 375]]]]  # 2D location of a window in the image
    raw_text = [["This is a test sentence."]]

    raw_image = raw_image * batch_size

    input_points = np.array(input_points)
    raw_text = np.array(raw_text, dtype=object)
    input_points = input_points.repeat(batch_size, axis=0).repeat(num_masks, axis=1).tolist()
    raw_text = raw_text.repeat(batch_size, axis=0).repeat(num_masks, axis=1).reshape(-1).tolist()

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt")

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    sam_model.train()
    model.train()
    sam_output = sam_model(**inputs)
    sca_output = model(**inputs)
    sam_output_from_sca = sca_output.segmentation_outputs
    for k in sam_output:
        if isinstance(sam_output[k], torch.Tensor):
            assert torch.allclose(sam_output[k], sam_output_from_sca[k])

    sam_model.eval()
    model.eval()
    with torch.no_grad():
        sam_output = sam_model(**inputs)
        sca_output = model.generate(**inputs)
    sam_output_from_sca = sca_output.segmentation_outputs
    for k in sam_output:
        if isinstance(sam_output[k], torch.Tensor):
            assert torch.allclose(sam_output[k], sam_output_from_sca[k])
