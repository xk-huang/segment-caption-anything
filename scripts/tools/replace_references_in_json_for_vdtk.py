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
from dataclasses import dataclass
import numpy as np
from functools import partial
import pandas as pd
import json
import tqdm
import yaml
from src.train import prepare_datasets

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(args: DictConfig) -> None:
    # NOTE(xiaoke): follow https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py
    logger.info(OmegaConf.to_yaml(args))

    logger.info(f"Add command args: +no_sanity_check=False")
    no_sanity_check = args.get("no_sanity_check", False)
    output_dir = args.training.output_dir
    if output_dir is None:
        raise ValueError("output_dir is None, which should not happen.")

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Initialize our dataset and prepare it
    # NOTE(xiaoke): We should only inference one eval dataset
    train_dataset, eval_dataset = prepare_datasets(args)

    for eval_dataset_k, eval_dataset_v in eval_dataset.items():
        replace_one_eval_dataset(no_sanity_check, output_dir, eval_dataset_k, eval_dataset_v)


def replace_one_eval_dataset(no_sanity_check, output_dir, eval_dataset_name, eval_dataset):
    infer_json_dir = os.path.join(output_dir, "infer")
    json_path = os.path.join(infer_json_dir, f"infer-{eval_dataset_name}.json")
    if not os.path.exists(json_path):
        raise ValueError(f"json_path={json_path} does not exist, which should not happen.")
    infer_replace_gt_json_dir = os.path.join(output_dir, "infer-post_processed")
    os.makedirs(infer_replace_gt_json_dir, exist_ok=True)
    output_json_path = os.path.join(infer_replace_gt_json_dir, f"infer-{eval_dataset_name}.json")

    with open(json_path, "r") as f:
        json_data = json.load(f)
    with open(output_json_path, "w") as f:
        json.dump({}, f, indent=4)

    if no_sanity_check is False:
        # NOTE: Check the sanity. We want the region orders in both eval_dataset and json_data are the same.
        logger.info(f"Check the sanity. We want the region orders in both eval_dataset and json_data are the same.")
        json_data_region_cnt = 0
        for sample in tqdm.tqdm(eval_dataset):
            try:
                for region in sample["regions"]:
                    eval_dataset_region_id = region["region_id"]
                    eval_dataset_image_id = region["image_id"]
                    json_data_region_id = json_data[json_data_region_cnt]["metadata"]["metadata_region_id"]
                    json_data_image_id = json_data[json_data_region_cnt]["metadata"]["metadata_image_id"]
                    assert eval_dataset_region_id == json_data_region_id
                    assert eval_dataset_image_id == json_data_image_id
                    json_data_region_cnt += 1
            except IndexError as e:
                logger.warning(f"Error: {e}. There are not enough samples in the predction, so we stop here.")
                break

    # NOTE: Now we start to replace the references in json_data with the references in eval_dataset
    json_data_region_cnt = 0
    pbar = tqdm.tqdm(eval_dataset)
    for sample in pbar:
        try:
            for region in sample["regions"]:
                gt_phrases = region["phrases"]
                old_phrases = json_data[json_data_region_cnt]["references"]
                json_data[json_data_region_cnt]["references"] = gt_phrases
                # pbar.set_description(f"old: {old_phrases}, new: {gt_phrases}")
                json_data_region_cnt += 1
        except IndexError as e:
            logger.warning(f"Error: {e}. There are not enough samples in the predction, so we stop here.")
            break

    logger.info(f"Save the new json_data to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    main()
