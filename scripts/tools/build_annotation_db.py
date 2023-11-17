# TODO: extract images from refcoco series
import sys

sys.path.append(".")
import base64
import io
import json
import logging
import os
import os.path as osp

import datasets
import hydra
import numpy as np
import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import pycocotools.mask
from utils.git_utils import TSVWriter
from src.arguments import Arguments, global_setup
import logging
from hydra.utils import instantiate
from transformers import set_seed, AutoTokenizer
from datasets import interleave_datasets, concatenate_datasets
import torch
from src.train import prepare_datasets
import sqlite3
import json
from torch.utils.data import IterableDataset, DataLoader

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(args: Arguments):
    logger.warning(f"Turn no_cuda = True.")
    args.training.no_cuda = True

    # NOTE: ddp is initialized in _setup_devices class in `transformers/training_args.py`
    args, training_args, _ = global_setup(args)

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Initialize our dataset and prepare it
    train_dataset, eval_dataset = prepare_datasets(args)
    if len(eval_dataset) > 1:
        raise ValueError(f"Only support one eval dataset, but got {len(eval_dataset)}. args: {args.eval_data}")

    # NOTE: According to https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb,
    # we use alternatively GPT2TokenizerFast.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Connect to the SQLite database (this creates a new file called "coco_annotations.db" if it doesn't exist)
    db_path = os.path.join(training_args.output_dir, "annotations.db")
    if os.path.exists(db_path):
        logger.info(f"Remove existing db file: {db_path}, and create a new one.")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    def _get_dataset_name_from_path(path):
        return osp.splitext(osp.basename(path))[0]

    process_dataset(
        train_dataset,
        training_args.max_train_samples,
        "_".join([_get_dataset_name_from_path(i["path"]) for i in args.train_data]),
        "train",
        training_args,
        tokenizer,
        args,
        conn,
    )
    for (eval_dataset_k, eval_dataset_v), eval_data_ in zip(eval_dataset.items(), args.eval_data):
        process_dataset(
            eval_dataset_v,
            training_args.max_eval_samples,
            _get_dataset_name_from_path(eval_data_["path"]),
            f"eval-{eval_dataset_k}",
            training_args,
            tokenizer,
            args,
            conn,
        )

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


SAMPLE_KEYS = ["image_id", "width", "height", "file_name", "coco_url", "task_type", "regions"]
REGION_KEYS = ["region_id", "image_id", "phrases", "x", "y", "width", "height"]


class PerRegionIterableDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, args, max_samples):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args
        self.max_samples = max_samples

    def get_len(self):
        return len(self.dataset)

    def generate(self):
        for sample_idx, sample in enumerate(self.dataset):
            if self.max_samples is not None and sample_idx >= self.max_samples:
                break
            for region in sample["regions"]:
                phrases = region["phrases"]
                tokenized_phrases = [self.tokenizer.tokenize(phrase) for phrase in phrases]
                region["tokenized_phrases"] = tokenized_phrases
                yield sample_idx, sample, region

    def __iter__(self):
        from itertools import islice
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        return_iter= islice(self.generate(), worker_id, None, num_workers)
        return return_iter



def process_dataset(dataset, max_samples, dataset_name, split_name, training_args, tokenizer, args, conn, batch_size=100_000):
    logger.info(f"Processing {dataset_name}.{split_name}: {dataset}...")
    if dataset is None:
        logger.warning(
            f"[{training_args.process_index}/{training_args.world_size}]: {split_name} is None, skip processing"
        )
        return

    per_region_dataset = PerRegionIterableDataset(dataset, tokenizer, args, max_samples)
    pre_region_dataloader = DataLoader(per_region_dataset, batch_size=1, num_workers=args.training.dataloader_num_workers, collate_fn=lambda x: x[0])

    cursor = conn.cursor()

    # Create a table for storing the annotations
    # cursor.execute(
    #     """
    # CREATE TABLE IF NOT EXISTS annotations (
    #     id INTEGER PRIMARY KEY,
    #     image_id INTEGER,
    #     category_id INTEGER,
    #     segmentation TEXT,
    #     area REAL,
    #     bbox TEXT,
    #     iscrowd INTEGER
    # )
    # """
    # )
    def _clean_table_name(name):
        return (
            name.replace("-", "_")
            .replace(".", "_")
            .replace(" ", "_")
            .replace(":", "_")
            .replace("/", "_")
            .lower()
            .strip()
            .replace("__", "_")
        )

    table_name = f"{dataset_name}_{split_name}"
    table_name = _clean_table_name(table_name)
    print(f"Creating {table_name} table...")
    cursor.execute(
        f"""  
    CREATE TABLE IF NOT EXISTS {table_name} (  
        region_id INTEGER PRIMARY KEY,  
        image_id INTEGER,  
        width INTEGER,  
        height INTEGER,  
        file_name TEXT,  
        coco_url TEXT,  
        task_type TEXT,  
        phrases TEXT,  
        tokenized_phrases TEXT,
        x REAL,  
        y REAL,  
        region_width REAL,  
        region_height REAL  
    )  
    """
    )

    region_cnt = 0
    sample_cnt = 0
    sent_cnt = 0
    token_cnt = 0
    word_cnt = 0
    pbar = tqdm.tqdm(total=per_region_dataset.get_len())
    prev_sample_idx = None
    for sample_idx, sample, region in pre_region_dataloader:
        phrases = region["phrases"]
        tokenized_phrases = region["tokenized_phrases"]

        dumped_phrases = json.dumps(phrases)
        dumped_tokenized_phrases = json.dumps(tokenized_phrases)

        sent_cnt += len(phrases)
        region_cnt += 1

        cursor.execute(
            f"""
        INSERT INTO {table_name} (region_id, image_id, width, height, file_name, coco_url, task_type, phrases, tokenized_phrases, x, y, region_width, region_height)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                region["region_id"],
                sample["image_id"],
                sample["width"],
                sample["height"],
                sample["file_name"],
                sample["coco_url"],
                sample["task_type"],
                dumped_phrases,
                dumped_tokenized_phrases,
                region["x"],
                region["y"],
                region["width"],
                region["height"],
            ),
        )

        sample_cnt += 1
        if sample_cnt % batch_size == 0:
            conn.commit()

        if prev_sample_idx != sample_idx:
            pbar.set_description(
                f"[{training_args.process_index}/{training_args.world_size}]: Already processing {sample_cnt} samples, {region_cnt} regions, {sent_cnt} sentences, and {token_cnt} tokens."
            )
            pbar.update(1)
            prev_sample_idx = sample_idx

    conn.commit()
    logger.info(
        f"[{training_args.process_index}/{training_args.world_size}]: Total samples: {sample_cnt}, total regions: {region_cnt}, total sents: {sent_cnt}, total tokens: {token_cnt}"
    )

    if training_args.process_index == 0:
        all_sample_cnt = sample_cnt
        all_region_cnt = region_cnt
        all_sent_cnt = sent_cnt
        all_token_cnt = token_cnt
        all_word_cnt = word_cnt
        logger.info(
            f"[FULL]: split name: {split_name}, total samples: {all_sample_cnt}, total regions: {all_region_cnt}, total sents: {all_sent_cnt}, total tokens: {all_token_cnt}, total words: {all_word_cnt}"
        )


if __name__ == "__main__":
    main()
