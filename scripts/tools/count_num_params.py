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
from src.train import prepare_model, prepare_model_trainable_parameters
import fvcore.nn

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(args: DictConfig) -> None:
    # NOTE(xiaoke): follow https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

    logger.info(OmegaConf.to_yaml(args))
    args, training_args, model_args = global_setup(args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            logger.warning(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "There is no checkpoint in the directory. Or we can resume from `resume_from_checkpoint`."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(args.training.seed)
    model = prepare_model(model_args)
    print(fvcore.nn.parameter_count_table(model))


if __name__ == "__main__":
    main()
