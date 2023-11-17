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

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(args: Arguments):
    logger.warning(f"Turn no_cuda = True.")
    args.training.no_cuda = True

    # NOTE: ddp is initialized in _setup_devices class in `transformers/training_args.py`
    args, training_args, _ = global_setup(args)

    dummy_data = torch.tensor(training_args.process_index)
    dummy_data_ls = [torch.tensor(1) for _ in range(training_args.world_size)]
    torch.distributed.all_gather(dummy_data_ls, dummy_data)
    logger.info(f"Try gloo ddp: rank {training_args.process_index} dummy_data_ls: {dummy_data_ls}")

    # Set seed before initializing model.
    set_seed(args.training.seed)

    # Initialize our dataset and prepare it
    train_dataset, eval_dataset = prepare_datasets(args)
    if len(eval_dataset) > 1:
        raise ValueError(f"Only support one eval dataset, but got {len(eval_dataset)}. args: {args.eval_data}")

    logger.info(f"world_size {training_args.world_size}")
    logger.info(f"process_index {training_args.process_index}")
    logger.info(f"local_process_index {training_args.local_process_index}")

    if training_args.world_size > 1:
        from datasets.distributed import split_dataset_by_node

        # NOTE: to debug ddp code
        # if training_args.should_log:
        #     breakpoint()
        # torch.distributed.barrier()

        logger.info(f"Splitting dataset by node {training_args.process_index}/{training_args.world_size}")
        if train_dataset is not None:
            train_dataset = split_dataset_by_node(train_dataset, training_args.process_index, training_args.world_size)
        if eval_dataset is not None:
            for eval_dataset_k, eval_dataset_v in eval_dataset.items():
                eval_dataset[eval_dataset_k] = split_dataset_by_node(
                    eval_dataset_v, training_args.process_index, training_args.world_size
                )

    logger.info(f"[{training_args.process_index}/{training_args.world_size}]: Train Dataset: {train_dataset}")
    logger.info(f"[{training_args.process_index}/{training_args.world_size}]: Eval Dataset: {eval_dataset}")

    # NOTE: According to https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb,
    # we use alternatively GPT2TokenizerFast.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    process_dataset(train_dataset, "train", training_args, tokenizer)
    for eval_dataset_k, eval_dataset_v in eval_dataset.items():
        process_dataset(eval_dataset_v, f"eval-{eval_dataset_k}", training_args, tokenizer)


def process_dataset(dataset, split_name, training_args, tokenizer):
    if dataset is None:
        logger.warning(
            f"[{training_args.process_index}/{training_args.world_size}]: {split_name} is None, skip processing"
        )
        return

    region_cnt = 0
    sample_cnt = 0
    sent_cnt = 0
    token_cnt = 0
    word_cnt = 0
    tbar = tqdm.trange(len(dataset), desc=f"Processing {split_name}")
    for sample_idx in tbar:
        try:
            sample = dataset[sample_idx]
        except Exception as e:
            logger.error(f"Error when processing {split_name} sample_idx {sample_idx}: {e}")
            continue

        for annot_idx, region in enumerate(sample["regions"]):
            cap = region["phrases"]
            for cap_ in cap:
                # NOTE: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.tokenize
                token_cnt += len(tokenizer.tokenize(cap_))
                # NOTE: https://huggingface.co/learn/nlp-course/chapter6/4#pre-tokenization
                word_cnt += len(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(cap_))
            sent_cnt += len(cap)
            region_cnt += 1

        sample_cnt += 1
        tbar.set_description(
            f"[{training_args.process_index}/{training_args.world_size}]: Already processing {sample_cnt} samples, {region_cnt} regions, {sent_cnt} sentences, and {token_cnt} tokens."
        )

    logger.info(
        f"[{training_args.process_index}/{training_args.world_size}]: Total samples: {sample_cnt}, total regions: {region_cnt}, total sents: {sent_cnt}, total tokens: {token_cnt}"
    )

    def _gather_cnts(cnt, training_args, name=""):
        if training_args.world_size > 1:
            cnt_ls = [torch.tensor(0) for _ in range(training_args.world_size)]
            torch.distributed.all_gather(cnt_ls, torch.tensor(cnt))
            cnt = sum(cnt_ls)
            if training_args.process_index == 0:
                logger.info(f"[{training_args.process_index}/{training_args.world_size}]: {name}_cnt_ls: {cnt}")
        return cnt

    all_sample_cnt = _gather_cnts(sample_cnt, training_args, "sample")
    all_region_cnt = _gather_cnts(region_cnt, training_args, "region")
    all_sent_cnt = _gather_cnts(sent_cnt, training_args, "sent")
    all_token_cnt = _gather_cnts(token_cnt, training_args, "token")
    all_word_cnt = _gather_cnts(word_cnt, training_args, "word")

    if training_args.process_index == 0:
        logger.info(
            f"[FULL]: split name: {split_name}, total samples: {all_sample_cnt}, total regions: {all_region_cnt}, total sents: {all_sent_cnt}, total tokens: {all_token_cnt}, total words: {all_word_cnt}"
        )


if __name__ == "__main__":
    main()
