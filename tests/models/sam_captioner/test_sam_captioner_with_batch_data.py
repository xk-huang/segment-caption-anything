import sys

sys.path.append(".")

import logging
import os
from typing import Optional, Dict

import hydra
import torch
from hydra.utils import instantiate
from datasets import DatasetDict, load_dataset, IterableDatasetDict
from omegaconf import DictConfig, OmegaConf
from src.data.transforms import SamCaptionerDataTransform
from src.data.collator import SamCaptionerDataCollator
from src.arguments import Arguments, global_setup, SAMCaptionerModelArguments, SCAModelArguments
from src.models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor

from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed, Trainer
import gradio as gr
from dataclasses import dataclass
from hydra import initialize, compose
from src.train import prepare_datasets, prepare_model, prepare_data_transform
from datasets import Dataset, IterableDataset


logger = logging.getLogger(__name__)


def test_with_initialize() -> None:
    with initialize(version_base="1.3", config_path="../../../src/conf"):
        args = compose(
            config_name="conf",
            overrides=[
                "train_data=[vg-densecap-region_descriptions]",
                "eval_data=[vg-densecap-region_descriptions]",
                "+model=base_sam_captioner",
                "training.do_train=True",
                "training.do_eval=True",
                "training.overwrite_output_dir=True",
                "training.num_masks_per_sample=6",
            ],
        )
        main(args)


def main(args: DictConfig) -> None:
    # NOTE(xiaoke): follow https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

    logger.info(OmegaConf.to_yaml(args))
    args, training_args, model_args = global_setup(args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Initialize our dataset and prepare it
    train_dataset, eval_dataset = prepare_datasets(args)

    processor = SAMCaptionerProcessor.from_sam_captioner_pretrained(
        model_args.sam_model_name_or_path,
        model_args.captioner_model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
    )
    # NOTE(xiaoke): add pad_token if not exists
    if processor.tokenizer.pad_token is None:
        if processor.tokenizer.eos_token is None:
            raise ValueError("tokenizer must have either eos_token")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset, eval_dataset = prepare_data_transform(
        training_args, model_args, train_dataset, eval_dataset, processor
    )
    if len(eval_dataset) > 1:
        raise ValueError(
            f"len(eval_dataset) should be 1, but got {len(eval_dataset)}. Check args.eval_data: {args.eval_data}"
        )
    eval_dataset = next(iter(eval_dataset.values()))

    collate_fn = SamCaptionerDataCollator(processor.tokenizer)

    model = prepare_model(model_args)

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    if training_args.do_train:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=collate_fn
        )
        train_data_loader = cycle(train_data_loader)
    else:
        train_data_loader = None

    if training_args.do_eval:
        eval_data_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=collate_fn
        )
        eval_data_loader = cycle(eval_data_loader)
    else:
        eval_data_loader = None

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    dtype = model.dtype

    @torch.no_grad()
    def run_one_batch(batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    batch[k] = v.to(device, dtype)
                else:
                    batch[k] = v.to(device)

        with torch.no_grad():
            model_outputs = model.sam(**batch)

        original_sizes = batch["original_sizes"]
        reshaped_input_sizes = batch["reshaped_input_sizes"]
        pred_masks = model_outputs.pred_masks
        masks = processor.post_process_masks(pred_masks, original_sizes, reshaped_input_sizes)

        model_outputs["masks"] = masks
        return model_outputs

    full_batch = next(train_data_loader)
    full_batch_model_outputs = run_one_batch(full_batch)

    def print_batch(batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)}")

    def make_sub_batch(batch, batch_dim_slice: slice, num_masks_dim_slice: slice):
        returned_batch = {}
        batch_num_masks_keys_to_be_edited = ["input_boxes"]
        batch_keys_to_be_edited = [
            "images",
            "pixel_values",
            "original_sizes",
            "reshaped_input_sizes",
        ]
        assert len(set(batch_num_masks_keys_to_be_edited).intersection(set(batch_keys_to_be_edited))) == 0
        for k, v in batch.items():
            if k in batch_num_masks_keys_to_be_edited:
                returned_batch[k] = v[batch_dim_slice, num_masks_dim_slice]
            elif k in batch_keys_to_be_edited:
                returned_batch[k] = v[batch_dim_slice]
            else:
                returned_batch[k] = v
        return returned_batch

    batch_dim_slice = slice(3, 5)
    num_masks_dim_slice = slice(0, 5)
    sub_batch = make_sub_batch(full_batch, batch_dim_slice, num_masks_dim_slice)
    print("sub_batch")
    print_batch(sub_batch)
    print("full_batch")
    print_batch(full_batch)
    sub_batch_model_outputs = run_one_batch(sub_batch)

    def make_sub_model_outputs(model_outputs, batch_dim_slice: slice, num_masks_dim_slice: slice):
        returned_model_outputs = {}

        batch_num_masks_model_outputs_keys_to_be_edited = ["iou_scores", "pred_masks"]
        batch_model_outputs_keys_to_be_edited = ["masks"]
        assert (
            len(
                set(batch_num_masks_model_outputs_keys_to_be_edited).intersection(
                    set(batch_model_outputs_keys_to_be_edited)
                )
            )
            == 0
        )

        for k, v in model_outputs.items():
            if k in batch_num_masks_model_outputs_keys_to_be_edited:
                returned_model_outputs[k] = v[batch_dim_slice, num_masks_dim_slice]
            elif k in batch_model_outputs_keys_to_be_edited:
                returned_model_outputs[k] = v[batch_dim_slice]
            else:
                returned_model_outputs[k] = v
        return returned_model_outputs

    sub_full_batch_model_outputs = make_sub_model_outputs(
        full_batch_model_outputs, batch_dim_slice, num_masks_dim_slice
    )
    print("sub_full_batch_model_outputs")
    print_batch(sub_full_batch_model_outputs)
    print("full_batch_model_outputs")
    print(sub_full_batch_model_outputs.get("iou_scores"))
    print(sub_batch_model_outputs.iou_scores)
    for k, v in sub_full_batch_model_outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"torch allclose {k}: {v.shape}")
            print(torch.allclose(v, getattr(sub_batch_model_outputs, k)))
    sub_full_batch_model_outputs.get("pred_masks")
    sub_batch_model_outputs.get("pred_masks")
    torch.max(torch.abs(sub_full_batch_model_outputs.get("pred_masks") - sub_batch_model_outputs.get("pred_masks")))

    full_batch_input_boxes = full_batch["input_boxes"]
    with torch.no_grad():
        full_batch_out_1, full_batch_out_2 = model.sam.prompt_encoder(
            input_boxes=full_batch_input_boxes, input_points=None, input_labels=None, input_masks=None
        )
    sub_batch_input_boxes = sub_batch["input_boxes"]
    with torch.no_grad():
        sub_batch_out_1, sub_batch_out_2 = model.sam.prompt_encoder(
            input_boxes=sub_batch_input_boxes, input_points=None, input_labels=None, input_masks=None
        )
    full_batch_out_1[batch_dim_slice, num_masks_dim_slice], sub_batch_out_1
    torch.allclose(full_batch_out_1[batch_dim_slice, num_masks_dim_slice], sub_batch_out_1, atol=1e-6)
    torch.allclose(full_batch_input_boxes[batch_dim_slice, num_masks_dim_slice], sub_batch_input_boxes)
