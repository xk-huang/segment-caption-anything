import sys

sys.path.append(".")

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from src.arguments import global_setup

from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
import tqdm
from src.train import (
    prepare_datasets,
    prepare_data_transform,
    SCASeq2SeqTrainer,
    prepare_processor,
    prepare_collate_fn,
)

from src.arguments import global_setup
import dotenv


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

    # NOTE(xiaoke): load sas_key from .env for huggingface model downloading.
    logger.info(f"Try to load sas_key from .env file: {dotenv.load_dotenv('.env')}.")
    use_auth_token = os.getenv("USE_AUTH_TOKEN", False)

    processor = prepare_processor(model_args, use_auth_token)

    train_dataset, eval_dataset = prepare_data_transform(
        training_args, model_args, train_dataset, eval_dataset, processor
    )

    collate_fn = prepare_collate_fn(training_args, model_args, processor)

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
    # model = prepare_model(model_args, use_auth_token)
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
