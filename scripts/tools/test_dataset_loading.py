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

from transformers.trainer_utils import get_last_checkpoint, seed_worker
from transformers import set_seed, Trainer
from dataclasses import dataclass
import numpy as np
from functools import partial
import pandas as pd
import json
import tqdm
import yaml
from src.train import prepare_datasets, prepare_data_transform, SCASeq2SeqTrainer

from src.data.transforms import SamCaptionerDataTransform, SCADataTransform
from src.data.collator import SamCaptionerDataCollator, SCADataCollator
from src.arguments import (
    Arguments,
    global_setup,
    SAMCaptionerModelArguments,
    SCAModelBaseArguments,
    SCAModelArguments,
    SCADirectDecodingModelArguments,
    SCAMultitaskModelArguments,
    SCAMultitaskSplitMixerModelArguments,
    ScaMultitaskV2ModelArguments,
    VGDenseCapDataArgument,
    RefCOCODataArgument,
    SA1BCapDataArgument,
    COCOInstanceDataArgument,
)
from src.models.sca import (
    ScaModel,
    ScaConfig,
    ScaProcessor,
    ScaDirectDecodingModel,
    ScaMultitaskModel,
    ScaMultitaskSplitMixerModel,
    ScaMultitaskV2Model,
)
from torch.utils.data import DataLoader


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

    # def collate_fn(examples):
    #     pixel_values = torch.stack([example["pixel_values"] for example in examples])
    #     labels = torch.tensor([example["labels"] for example in examples])
    #     return {"pixel_values": pixel_values, "labels": labels}
    DataCollatorClass = None
    if isinstance(model_args, SAMCaptionerModelArguments):
        DataCollatorClass = SamCaptionerDataCollator
    elif isinstance(model_args, SCAModelBaseArguments):
        DataCollatorClass = SCADataCollator
    collate_fn = DataCollatorClass(processor.tokenizer)

    # Load the accuracy metric from the datasets package
    # metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    # def compute_metrics(p):
    # """Computes accuracy on a batch of predictions"""
    # return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    compute_metrics = training_args.compute_metrics
    if compute_metrics is not True:
        # NOTE: compute_metrics = None triggers the default `prediction_loss_only=True`
        # NOTE: compute_metrics should be a function, but we define the function in the trainer, so we use bool here to indicate the usage.
        compute_metrics = None

    # config = AutoConfig.from_pretrained(
    #     model_args.config_name or model_args.model_name_or_path,
    #     num_labels=len(labels),
    #     label2id=label2id,
    #     id2label=id2label,
    #     finetuning_task="image-classification",
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    # model = AutoModelForImageClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    #     ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    # )
    # image_processor = AutoImageProcessor.from_pretrained(
    #     model_args.image_processor_name or model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    # model = prepare_model(model_args)
    # prepare_model_trainable_parameters(model, args)

    # # Initalize our trainer
    # custom_callbacks = [LoggerCallback(), EvalLossCallback()]
    # if args.wandb.log is True:
    #     custom_callbacks.append(CustomWandbCallBack(args))
    # if training_args.evaluate_before_train:
    #     custom_callbacks.append(EvaluateFirstStepCallback())

    model = torch.nn.Linear(10, 2)
    trainer = SCASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval or training_args.do_train else None,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        callbacks=None,
    )

    # Training
    if training_args.do_train:
        train_dataloader = trainer.get_train_dataloader()
        run_and_print_dataloader(train_dataloader)

    # Evaluation or Inference
    if training_args.do_eval or training_args.do_inference:
        for eval_dataset_k, eval_dataset_v in eval_dataset.items():
            eval_dataloader = trainer.get_eval_dataloader(eval_dataset_v)
            run_and_print_dataloader(eval_dataloader)


def run_and_print_dataloader(dataloader):
    pbar = tqdm.tqdm(dataloader)
    for batch in pbar:
        batch_str = ""
        for k, v in batch.items():
            if v is None:
                batch_str += f"{k}: None\n"
            elif isinstance(v, torch.Tensor):
                batch_str += f"{k}: {v.shape}\n"
            elif isinstance(v, list):
                batch_str += f"{k}: {len(v)}\n"
            else:
                batch_str += f"{k}: {v}\n"
        pbar.write(batch_str)


if __name__ == "__main__":
    main()
