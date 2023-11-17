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
    args, training_args, model_args = global_setup(args)
    breakpoint()


if __name__ == "__main__":
    main()
