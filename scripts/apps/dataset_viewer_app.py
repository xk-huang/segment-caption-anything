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
from src.arguments import Arguments, global_setup, SAMCaptionerModelArguments, SCAModelArguments, SCAModelBaseArguments
from src.models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor
from src.models.sca import ScaProcessor

from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed, Trainer
import gradio as gr
from dataclasses import dataclass
import numpy as np
from functools import partial
import pandas as pd
from src.train import prepare_datasets, prepare_data_transform
import pycocotools.mask
from PIL import Image

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(args: DictConfig) -> None:
    # NOTE(xiaoke): follow https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

    logger.info(OmegaConf.to_yaml(args))
    args, training_args, model_args = global_setup(args)

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Initialize our dataset and prepare it
    train_dataset, eval_dataset = prepare_datasets(args)

    if isinstance(model_args, SAMCaptionerModelArguments):
        processor = SAMCaptionerProcessor.from_sam_captioner_pretrained(
            model_args.sam_model_name_or_path,
            model_args.captioner_model_name_or_path,
            cache_dir=model_args.cache_dir,
            model_max_length=model_args.model_max_length,
        )
    # FIXME: when load weights from existing sca model, we should use the same tokenizer as the existing sca model
    # model.lm_head_model_name_or_path=$(grep lm_head_model_name_or_path $AMLT_MAP_INPUT_DIR/.hydra/config.yaml | tail -n1 | sed 's/ *//g' | cut -d ':' -f2)
    # model.sam_model_name_or_path=$(grep sam_model_name_or_path $AMLT_MAP_INPUT_DIR/.hydra/config.yaml | tail -n1 | sed 's/ *//g' | cut -d ':' -f2)
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

    train_dataset, eval_dataset = prepare_data_transform(
        training_args, model_args, train_dataset, eval_dataset, processor
    )

    # [NOTE] Used to restore the image tensor after transformed
    # Use global to avoid passing too many arguments
    global image_mean, image_std
    image_mean, image_std = (
        processor.sam_processor.image_processor.image_mean,
        processor.sam_processor.image_processor.image_std,
    )

    def view_one_batch(dataset_split, batch_idx, dataset_type):
        if dataset_type == "before_transform":
            return _view_one_batch_before_transform(dataset_split, batch_idx, dataset_type)
        elif dataset_type == "after_transform":
            return _view_one_batch_after_transform(dataset_split, batch_idx, dataset_type)
        else:
            raise ValueError(f"Unknown type of sample: {dataset_type}")

    def _view_one_batch_before_transform(dataset_split, batch_idx, dataset_type):
        sample = dataset_split[batch_idx]
        image = sample["image"]

        text = f"dataset_type: {dataset_type}\nsample_id: {batch_idx}\n"
        for k, v in sample.items():
            if isinstance(v, (int, str)):
                text += f"{k}: {v}\n"

        regions = sample["regions"]
        regions = pd.DataFrame(regions)
        regions.sort_values(by=["region_id"], ascending=True, inplace=True)
        return image, text, regions

    def _view_one_batch_after_transform(dataset_split, batch_idx, dataset_type):
        sample = dataset_split[batch_idx]
        image = sample["images"]
        image = sample["pixel_values"]
        image_mean_tensor = torch.tensor(image_mean).view(3, 1, 1)
        image_std_tensor = torch.tensor(image_std).view(3, 1, 1)
        image = image * image_std_tensor + image_mean_tensor
        image = image.clamp(0, 1) * 255
        image = image.permute(1, 2, 0).numpy().astype(np.uint8)

        PRINT_VALUE_KEYS = ["original_sizes", "reshaped_input_sizes"]
        text = f"dataset_type: {dataset_type}\nsample_id: {batch_idx}\n"
        for k, v in sample.items():
            text += f"{k}:\t{type(v)}\t"
            if k in PRINT_VALUE_KEYS:
                text += f"{v}\n"
            elif isinstance(v, str):
                text += f"{v}\n"
            elif isinstance(v, torch.Tensor):
                text += f"{v.shape}\n"
            elif isinstance(v, list):
                text += f"{len(v)}\n"
            elif isinstance(v, np.ndarray):
                text += f"{v.shape}\n"
            else:
                try:
                    text += f"{v.size}\n"
                except AttributeError:
                    text += f"{v}\n"

        REGION_KEYS = [
            "input_boxes",
            "metadata_input_boxes",
            "metadata_image_id",
            "metadata_region_id",
            "metadata_captions",
        ]
        pd_series = []
        for region_tensor_key in REGION_KEYS:
            region_tensor = sample[region_tensor_key]
            # NOTE: cast the float to int in bbox.
            if region_tensor_key == "input_boxes":
                if isinstance(region_tensor, torch.Tensor):
                    region_tensor = region_tensor.long()
                elif isinstance(region_tensor, np.ndarray):
                    region_tensor = region_tensor.astype(np.int64)
            if isinstance(region_tensor, (torch.Tensor, np.ndarray)):
                region_list = region_tensor.tolist()
            elif isinstance(region_tensor, list):
                region_list = region_tensor
            else:
                raise ValueError(f"Unknown type of region_tensor: {type(region_tensor)}")
            pd_series.append(pd.Series(region_list, name=region_tensor_key))
        regions = pd.concat(pd_series, axis=1)
        regions.sort_values(by=["metadata_region_id"], ascending=True, inplace=True)

        return image, text, regions

    def view_one_region(image, data_frame, output_chioce_radio, dataset_type, evt: gr.SelectData):
        if dataset_type == "before_transform":
            return _view_one_region_before_transform(image, data_frame, output_chioce_radio, evt)
        elif dataset_type == "after_transform":
            return _view_one_region_after_transform(image, data_frame, output_chioce_radio, evt)
        else:
            raise ValueError(f"Unknown type of sample: {dataset_type}")

    def _view_one_region_before_transform(image, data_frame, output_chioce_radio, evt):
        row_id, _ = evt.index
        region = data_frame.iloc[row_id]
        if output_chioce_radio == "segmentation" and region.get("mask", None) is not None:
            annot = region["mask"]
            annot = pycocotools.mask.decode(annot)
        elif output_chioce_radio == "segmentation" and region.get("mask", None) is None:
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            x2, y2 = x + w, y + h
            annot = [x, y, x2, y2]
        elif output_chioce_radio == "bbox":
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            x2, y2 = x + w, y + h
            annot = [x, y, x2, y2]
        else:
            raise ValueError(f"Unknown output_chioce_radio: {output_chioce_radio}")

        phrases = [f"{idx}: {phrase}" for idx, phrase in enumerate(region["phrases"])]
        phrases = "; ".join(phrases)
        return image, [[annot, phrases]]

    def _view_one_region_after_transform(image, data_frame, output_chioce_radio, evt):
        row_id, _ = evt.index
        region = data_frame.iloc[row_id]
        if output_chioce_radio == "segmentation" and region.get("mask", None) is not None:
            raise NotImplementedError("TODO: implement segmentation for after_transform")
        elif output_chioce_radio == "segmentation" and region.get("mask", None) is None:
            annot = list(map(int, region["input_boxes"]))
        elif output_chioce_radio == "bbox":
            annot = list(map(int, region["input_boxes"]))
        else:
            raise ValueError(f"Unknown output_chioce_radio: {output_chioce_radio}")

        phrases = region["metadata_captions"]
        if not isinstance(phrases[0], list):
            phrases = [phrases]
        phrases = [f"{idx}: {phrase}" for idx, phrase in enumerate(phrases)]
        phrases = "; ".join(phrases)
        return image, [[annot, phrases]]

    def get_gr_frame(frame_name, dataset_split):
        dataset_type = "before_transform" if dataset_split[0].get("images", None) is None else "after_transform"
        dataset_type = gr.Variable(dataset_type)
        with gr.Accordion(label=frame_name) as frame:
            batch_idx = gr.Slider(minimum=0, maximum=len(dataset_split), step=1, default=0)
            button = gr.Button(text="View the batch")
            output_chioce_radio = gr.Radio(["bbox", "segmentation"], value="bbox")

            image = gr.Image(height=500)
            text = gr.Textbox(lines=1)
            data_frame = gr.DataFrame()
            annotated_image = gr.AnnotatedImage(height=500)

            dataset_split = gr.Variable(dataset_split)
            button.click(
                view_one_batch, inputs=[dataset_split, batch_idx, dataset_type], outputs=[image, text, data_frame]
            )
            data_frame.select(
                view_one_region,
                inputs=[image, data_frame, output_chioce_radio, dataset_type],
                outputs=[annotated_image],
            )

        return frame

    with gr.Blocks() as app:
        get_gr_frame("train", train_dataset)
        for eval_data_k, eval_data_v in eval_dataset.items():
            get_gr_frame(f"validate-{eval_data_k}", eval_data_v)
    app.launch()


if __name__ == "__main__":
    main()
