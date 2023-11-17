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
from src.data.transforms import SamCaptionerDataTransform, SCADataTransform
from src.data.collator import SamCaptionerDataCollator, SCADataCollator
from src.arguments import (
    Arguments,
    global_setup,
    SAMCaptionerModelArguments,
    SCAModelArguments,
    SCAModelBaseArguments,
    SCADirectDecodingModelArguments,
    SCAMultitaskModelArguments,
)
from src.models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor
from src.models.sca import ScaProcessor, ScaModel, ScaDirectDecodingModel, ScaMultitaskModel

from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed, Trainer
import gradio as gr
from dataclasses import dataclass
import numpy as np
import datasets
from datasets import Dataset, IterableDataset
from src.train import prepare_datasets, prepare_model, prepare_data_transform

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
    if len(eval_dataset) > 1:
        raise ValueError(f"Only support one eval dataset, but got {len(eval_dataset)}. args: {args.eval_data}")
    eval_dataset = next(iter(eval_dataset.values()))

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

    if training_args.do_eval or training_args.do_inference:
        eval_data_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=training_args.per_device_eval_batch_size, collate_fn=collate_fn
        )
        eval_data_loader = cycle(eval_data_loader)
    else:
        eval_data_loader = None

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    dtype = model.dtype

    @dataclass
    class BatchVariable:
        batch_input: Optional[dict] = None
        batch_output: Optional[dict] = None
        batch_id: int = 0
        region_id: int = 0

    @torch.no_grad()
    def run_one_batch(data_loader, batch_variable: BatchVariable):
        batch = next(data_loader)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    batch[k] = v.to(device, dtype)
                else:
                    batch[k] = v.to(device)

        with torch.inference_mode():
            if isinstance(model_args, SAMCaptionerModelArguments):
                model_outputs = model.generate(**batch, return_patches=True, return_dict_in_generate=True)
            else:
                model_outputs = model.generate(**batch)
        # add masks to model_outputs
        original_sizes = batch["original_sizes"]
        reshaped_input_sizes = batch["reshaped_input_sizes"]
        pred_masks = model_outputs.pred_masks
        masks = processor.post_process_masks(pred_masks, original_sizes, reshaped_input_sizes)
        model_outputs.masks = masks

        # add generated_captions to model_outputs
        batch_size, region_size, num_heads, num_tokens = model_outputs.sequences.shape
        generated_captions = processor.tokenizer.batch_decode(
            model_outputs.sequences.view(-1, num_tokens), skip_special_tokens=True
        )
        generated_captions = (
            np.array(generated_captions, dtype=object).reshape(batch_size, region_size, num_heads).tolist()
        )
        model_outputs.generated_captions = generated_captions

        batch_variable.batch_input = batch
        batch_variable.batch_output = model_outputs

        return f"finished running one batch, batch_size={len(batch['images'])}, region_size={len(masks[0])}"

    def run_one_batch_train(batch_variable: BatchVariable):
        if train_data_loader is None:
            raise ValueError("train_data_loader is None, use `training.do_train=True`.")
        return run_one_batch(train_data_loader, batch_variable)

    def run_one_batch_eval(batch_variable: BatchVariable):
        if eval_data_loader is None:
            raise ValueError("eval_data_loader is None, use `training.do_eval=True` or `training.do_inference=True`.")
        return run_one_batch(eval_data_loader, batch_variable)

    def display_one_batch(batch_variable):
        masks = batch_variable.batch_output.masks
        generated_captions = batch_variable.batch_output.generated_captions
        batch = batch_variable.batch_input
        batch_id = batch_variable.batch_id
        region_id = batch_variable.region_id

        batch_size = len(batch["images"])
        region_size = len(masks[0])
        num_mask_heads = len(masks[0][0])
        num_caption_heads = len(generated_captions[0][0])
        batch_variable.region_id = (region_id + 1) % region_size
        if batch_variable.region_id == 0:
            batch_variable.batch_id = (batch_id + 1) % batch_size
            if batch_variable.batch_id == 0:
                print("reached the end of the batch")

        if isinstance(model_args, SAMCaptionerModelArguments):
            patches = batch_variable.batch_output.patches[batch_id][region_id]
        else:
            # NOTE: This will lead to no images displayed.
            patches = [None] * 3

        # Tuple[numpy.ndarray | PIL.Image | str, List[Tuple[numpy.ndarray | Tuple[int, int, int, int], str]]]
        # NOTE: repeat the captions if there are less than 3 heads
        # NOTE: shape is list of list of obj, (batch, region, head)
        return (
            (
                batch["images"][batch_id],
                [
                    (
                        i.cpu().numpy(),
                        f"mask-{head_id}:{generated_captions[batch_id][region_id][min(head_id, num_caption_heads - 1)]}",
                    )
                    for head_id, i in enumerate(masks[batch_id][region_id])
                ]
                + [(batch["metadata_input_boxes"][batch_id][region_id].int().tolist(), "box")],
            ),
            f"batch_id={batch_id}({batch_size}), region_id={region_id}({region_size})",
            *patches,
        )

    with gr.Blocks() as app_main:
        train_annotated_image = gr.AnnotatedImage(height=500)
        with gr.Row():
            train_patch_images = [gr.Image(height=100) for _ in range(3)]
        train_batch_output = gr.Variable(BatchVariable())

        train_run_button = gr.Button(value="Run one batch")
        train_run_button_text = gr.Textbox(lines=1, label="train_run_button_text")
        train_display_button = gr.Button(value="Display one region")
        train_display_button_text = gr.Textbox(lines=1, label="train_display_button")

        train_run_button_handle = train_run_button.click(
            run_one_batch_train, inputs=[train_batch_output], outputs=[train_run_button_text]
        )
        train_run_button_handle.then(
            display_one_batch,
            inputs=[train_batch_output],
            outputs=[train_annotated_image, train_display_button_text, *train_patch_images],
        )
        train_display_button.click(
            display_one_batch,
            inputs=[train_batch_output],
            outputs=[train_annotated_image, train_display_button_text, *train_patch_images],
        )

        eval_annotated_image = gr.AnnotatedImage(height=500)
        with gr.Row():
            eval_patch_images = [gr.Image(height=100) for _ in range(3)]
        eval_batch_output = gr.Variable(BatchVariable())

        eval_run_button = gr.Button(value="Run one batch")
        eval_run_button_text = gr.Textbox(lines=1, label="eval_run_button")
        eval_display_button = gr.Button(value="Display one region")
        eval_display_button_text = gr.Textbox(lines=1, label="eval_display_button")

        eval_run_button_handle = eval_run_button.click(
            run_one_batch_eval, inputs=[eval_batch_output], outputs=[eval_run_button_text]
        )
        eval_run_button_handle.then(
            display_one_batch,
            inputs=[eval_batch_output],
            outputs=[eval_annotated_image, eval_display_button_text, *eval_patch_images],
        )
        eval_display_button.click(
            display_one_batch,
            inputs=[eval_batch_output],
            outputs=[eval_annotated_image, eval_display_button_text, *eval_patch_images],
        )

    app_main.launch()


if __name__ == "__main__":
    main()
