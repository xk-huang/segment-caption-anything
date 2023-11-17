import logging
import os

import hydra
from hydra.utils import instantiate
from datasets import Dataset, load_dataset, IterableDataset, concatenate_datasets, interleave_datasets
from omegaconf import DictConfig, OmegaConf
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
    SCADirectDecodingV2ModelArguments,
    SCAMultitaskROIPoolModelArguments,
    ScaTimmMultitaskV2ModelArguments,
)
from src.models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor
from src.sca_seq2seq_trainer import SCASeq2SeqTrainer, get_parameter_by_name, SAVING_FINISHED_FLAG
from src.models.sca import (
    ScaModel,
    ScaConfig,
    ScaProcessor,
    ScaDirectDecodingModel,
    ScaMultitaskModel,
    ScaMultitaskSplitMixerModel,
    ScaMultitaskV2Model,
    ScaDirectDecodingV2Model,
    ScaMultitaskROIPoolModel,
    ScaTimmMultitaskV2Model,
)
from src.integrations import CustomWandbCallBack, EvaluateFirstStepCallback, LoggerCallback, EvalLossCallback
import src.models.sca
import src.utils

from transformers.trainer_utils import _re_checkpoint
from transformers import set_seed
import json

logger = logging.getLogger(__name__)


# Copied from `transformers/trainer_utils.py`
def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        logger.warning(f"No checkpoint found in {folder}, but we got: {content}")
        return
    checkpoints = sorted(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0]), reverse=True)
    for ckeckpoint in checkpoints:
        # NOTE: it is possible for partial saving which cannot be read.
        if os.path.isfile(os.path.join(folder, ckeckpoint, SAVING_FINISHED_FLAG)):
            return os.path.join(folder, ckeckpoint)
        else:
            logger.warning(f"Checkpoint {os.path.join(folder, ckeckpoint)} does not have {SAVING_FINISHED_FLAG}, skip")
    return


@hydra.main(version_base="1.3", config_path="conf", config_name="conf")
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
    model = prepare_model(model_args)
    prepare_model_trainable_parameters(model, args)

    # Initalize our trainer
    custom_callbacks = [LoggerCallback(), EvalLossCallback()]
    if args.wandb.log is True:
        custom_callbacks.append(CustomWandbCallBack(args))
    if training_args.evaluate_before_train:
        custom_callbacks.append(EvaluateFirstStepCallback())

    trainer = SCASeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval or training_args.do_train else None,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        callbacks=custom_callbacks,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        for eval_dataset_k, eval_dataset_v in eval_dataset.items():
            metrics = trainer.evaluate(eval_dataset_v, metric_key_prefix=eval_dataset_k)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_inference:
        for eval_dataset_k, eval_dataset_v in eval_dataset.items():
            trainer.inference(eval_dataset_v, metric_key_prefix=eval_dataset_k)


def prepare_data_transform(training_args, model_args, train_dataset, eval_dataset, processor):
    DataTransformClass = None
    if isinstance(model_args, SAMCaptionerModelArguments):
        DataTransformClass = SamCaptionerDataTransform
    elif isinstance(model_args, SCAModelBaseArguments):
        DataTransformClass = SCADataTransform
    if training_args.do_train:
        if train_dataset is None:
            raise ValueError("train_dataset must be provided if do_train is True")

        num_masks_per_sample = training_args.num_masks_per_sample
        if num_masks_per_sample is None:
            num_masks_per_sample = 64
            logger.info(f"num_masks_per_sample not provided, defaulting to {num_masks_per_sample}")

        data_transforms = training_args.data_transforms

        train_transforms = DataTransformClass(
            processor.sam_processor, processor.tokenizer, "train", num_masks_per_sample, data_transforms
        )

        if isinstance(train_dataset, Dataset) and training_args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(
                range(training_args.max_train_samples)
            )
        # Set the training transforms
        if isinstance(train_dataset, Dataset):
            train_dataset = train_dataset.with_transform(train_transforms)
        elif isinstance(train_dataset, IterableDataset):
            train_dataset = train_dataset.map(
                train_transforms, batched=True, batch_size=training_args.per_device_train_batch_size
            )
        else:
            raise ValueError(f"dataset must be one of [Dataset, IterableDataset], got {type(train_dataset)}")
    else:
        logger.warning("do_train is False, so we do not apply data augmentation to train_dataset")

    if training_args.do_eval or training_args.do_inference or training_args.do_train:
        if eval_dataset is None:
            raise ValueError("eval_dataset must be provided if do_eval or do_inference is True")

        eval_transforms = DataTransformClass(processor.sam_processor, processor.tokenizer, "inference")
        for eval_dataset_k, eval_dataset_v in eval_dataset.items():
            if isinstance(eval_dataset_v, Dataset) and training_args.max_eval_samples is not None:
                eval_dataset_v = eval_dataset_v.select(range(training_args.max_eval_samples))
            # Set the validation transforms
            if isinstance(eval_dataset_v, Dataset):
                eval_dataset_v = eval_dataset_v.with_transform(eval_transforms)
            elif isinstance(eval_dataset_v, IterableDataset):
                eval_dataset_v = eval_dataset_v.map(
                    eval_transforms, batched=True, batch_size=training_args.per_device_eval_batch_size
                )
            else:
                raise ValueError(f"dataset must be one of [Dataset, IterableDataset], got {type(eval_dataset_v)}")
            eval_dataset[eval_dataset_k] = eval_dataset_v
    else:
        logger.warning(
            "do_eval and do_inference and do_train are False, so we do not apply data augmentation to eval_dataset"
        )
    return train_dataset, eval_dataset


def prepare_model_trainable_parameters(model, args):
    trainable_params = args.training.trainable_params
    if trainable_params is None:
        logger.info("trainable_params is not provided, defaulting to the `config_parameters` method of the model.")
        return

    for param in model.parameters():
        param.requires_grad = False

    logger.info(f"Config trainable_params: {trainable_params}")
    for param_name in trainable_params:
        param = get_parameter_by_name(model, param_name)
        for _param in param.parameters():
            _param.requires_grad = True
        logger.info(f"Set {param_name} to trainable")


def prepare_model(model_args):
    if isinstance(model_args, SAMCaptionerModelArguments):
        model_args: SAMCaptionerModelArguments
        model = SAMCaptionerModel.from_sam_captioner_pretrained(
            model_args.sam_model_name_or_path, model_args.captioner_model_name_or_path, cache_dir=model_args.cache_dir
        )
    elif isinstance(model_args, SCAModelBaseArguments):
        model_args: SCAModelBaseArguments
        if model_args.model_name_or_path is None:
            if model_args.sam_model_name_or_path is None:
                raise ValueError(
                    "model_args.sam_model_name_or_path must be specified in SCAModelBaseArguments if model_args.model_name_or_path is None. "
                    "Since we are not loading from a existing sca model."
                )
            if model_args.lm_head_model_name_or_path is None:
                raise ValueError(
                    "model_args.lm_head_model_name_or_path must be specified in SCAModelBaseArguments if model_args.model_name_or_path is None. "
                    "Since we are not loading from a existing sca model."
                )
            # NOTE(xiaoke): Initalize different kinds of sca models
            if isinstance(model_args, SCAModelArguments):
                model = ScaModel.from_sam_text_pretrained(
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    model_args.additional_num_hidden_layers,
                    model_args.num_caption_tokens,
                    cache_dir=model_args.cache_dir,
                )
            elif isinstance(model_args, SCADirectDecodingModelArguments):
                model = ScaDirectDecodingModel.from_sam_text_pretrained(
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    model_args.additional_num_hidden_layers,
                    cache_dir=model_args.cache_dir,
                )
            elif isinstance(model_args, SCAMultitaskModelArguments):
                model = ScaMultitaskModel.from_sam_text_pretrained(
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    model_args.additional_num_hidden_layers,
                    model_args.num_caption_tokens,
                    model_args.num_task_tokens,
                    cache_dir=model_args.cache_dir,
                )
            elif isinstance(model_args, ScaMultitaskV2ModelArguments):
                model = ScaMultitaskV2Model.from_sam_text_pretrained(
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    model_args.additional_num_hidden_layers,
                    model_args.num_caption_tokens,
                    model_args.num_task_tokens,
                    model_args.num_caption_heads,
                    cache_dir=model_args.cache_dir,
                )
            elif isinstance(model_args, SCAMultitaskSplitMixerModelArguments):
                model = ScaMultitaskSplitMixerModel.from_sam_text_pretrained(
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    model_args.additional_num_hidden_layers,
                    model_args.num_caption_tokens,
                    model_args.num_task_tokens,
                    model_args.num_caption_heads,
                    cache_dir=model_args.cache_dir,
                )
            elif isinstance(model_args, SCADirectDecodingV2ModelArguments):
                model = ScaDirectDecodingV2Model.from_sam_text_pretrained(
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    model_args.additional_num_hidden_layers,
                    model_args.num_task_tokens,
                    cache_dir=model_args.cache_dir,
                )
            elif isinstance(model_args, SCAMultitaskROIPoolModelArguments):
                model = ScaMultitaskROIPoolModel.from_sam_text_pretrained(
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    num_task_tokens=model_args.num_task_tokens,
                    vl_projector_type=model_args.vl_projector_type,
                    vl_projector_norm_type=model_args.vl_projector_norm_type,
                    cache_dir=model_args.cache_dir,
                )
            elif isinstance(model_args, ScaTimmMultitaskV2ModelArguments):
                timm_vision_name = getattr(model_args, "timm_vision_name", None)
                logger.info(f"timm_vision_name: {timm_vision_name}; sam path: {model_args.sam_model_name_or_path}")
                model = ScaTimmMultitaskV2Model.from_sam_timm_text_pretrained(
                    timm_vision_name,
                    model_args.sam_model_name_or_path,
                    model_args.lm_head_model_name_or_path,
                    model_args.additional_num_hidden_layers,
                    model_args.num_caption_tokens,
                    model_args.num_task_tokens,
                    model_args.num_caption_heads,
                    cache_dir=model_args.cache_dir,
                )
            else:
                raise ValueError(
                    f"model_args must be one of [SCAModelArguments, SCADirectDecodingModelArguments, SCAMultitaskModelArguments]"
                )
            logger.info(
                f"Initalized sca model from sam model {model_args.sam_model_name_or_path} and lm head model {model_args.lm_head_model_name_or_path}"
            )
        else:
            # NOTE(xiaoke): load from existing sca series model for inference
            model_config_json_path = os.path.join(model_args.model_name_or_path, "config.json")
            with open(model_config_json_path, "r") as f:
                model_config = json.load(f)
            if len(model_config["architectures"]) > 1:
                raise ValueError(f"Only support one architecture in model_config, got {model_config['architectures']}")
            architecture = model_config["architectures"][0]

            architecture_class = getattr(src.models.sca, architecture)
            model = architecture_class.from_pretrained(model_args.model_name_or_path)
            logger.info(f"Loaded sca model from {model_args.model_name_or_path}")
    else:
        raise ValueError(
            f"model_args must be one of [SAMCaptionerModelArguments, SCAModelBaseArguments], got {model_args}"
        )
    logger.info(f"Model: {model.config}")
    return model


def prepare_datasets(args):
    train_data = []
    for train_data_config_name in args.train_data:
        cfg = hydra.compose(config_name=f"data/{train_data_config_name}", overrides=args.train_data_overrides)
        train_data.append(cfg.data)
    args.train_data = train_data

    # NOTE(xiaoke): We should only inference one eval dataset
    if len(args.eval_data) > 1:
        logger.warning(f"We should only inference one dataset, got {args.eval_data}")
    eval_data = []
    for eval_data_config_name in args.eval_data:
        cfg = hydra.compose(config_name=f"data/{eval_data_config_name}", overrides=args.eval_data_overrides)
        eval_data.append(cfg.data)

    train_dataset = []
    for i, each_train_data in enumerate(train_data):
        # NOTE: add data `split` to each dataset
        each_train_data.split = "train"

        _train_dataset = instantiate(each_train_data)
        train_dataset.append(_train_dataset)
        logger.info(f"Train Dataset [{i}]: {each_train_data}\n{_train_dataset}")

    eval_dataset = {}
    for i, each_eval_data in enumerate(eval_data):
        # NOTE: add data `split` to each dataset
        # NOTE: visual genome has validation set, but we use test set for evaluation
        if "visual_genome.py" in each_eval_data.path and getattr(each_eval_data, "use_densecap_splits", None) is True:
            logger.info("Using densecap splits in Visual Genome, using test split to eval")
            each_eval_data.split = "test"

        # NOTE: refcoco has validation set, but we use test set for evaluation
        elif "refcoco.py" in each_eval_data.path:
            if each_eval_data.name.startswith("refcoco-") or each_eval_data.name.startswith("refcoco+-"):
                if each_eval_data.split is None or each_eval_data.split == "train":
                    raise ValueError(f"refcoco{{,+}} must have split for eval. got {each_eval_data.split}")
                logger.info(f"Using refcoco{{,+}}: {each_eval_data.split} split to eval")
            elif each_eval_data.name.startswith("refcocog"):
                logger.info("Using refcocog val split to eval")
                each_eval_data.split = "validation"
            elif each_eval_data.name.startswith("refclef"):
                logger.info("Using refclef val split to eval")
                each_eval_data.split = "validation"

        # NOTE: coco has validation set, but it does not have test set.
        elif "coco_instance.py" in each_eval_data.path or "coco_instance-local.py" in each_eval_data.path:
            logger.info("Using coco val split to eval")
            each_eval_data.split = "validation"

        elif "objects365-local.py" in each_eval_data.path:
            logger.info("Using objects365 (in fact, it is COCO) val split to eval")
            each_eval_data.split = "validation"

        elif "v3det-local.py" in each_eval_data.path:
            logger.info("Using v3det (in fact, it is COCO) val split to eval")
            each_eval_data.split = "validation"

        elif "sbu-pseudo_region-local.py" in each_eval_data.path or "sbu-pseudo_region.py" in each_eval_data.path:
            logger.info("Using sbu to eval, but it does not have test split, so we use train split")
            each_eval_data.split = "train"

        elif "coco_caption-pseudo_region.py" in each_eval_data.path:
            logger.info("Using coco_caption (in fact, it is COCO) val split to eval")
            each_eval_data.split = "validation"

        elif (
            "visual_genome-densecap-local.py" in each_eval_data.path
            or "visual_genome-grit-local.py" in each_eval_data.path
        ):
            logger.info(f"Using visual_genome (They are my custom splits for GRiT and Densecap) test split to eval")
            each_eval_data.split = "test"
        else:
            raise ValueError(
                f"Unknown dataset {each_eval_data.path}, we cannot determine the split for it. Please edit `src/train.py:prepare_datasets` to add the split for it."
            )

        _eval_dataset = instantiate(each_eval_data)
        eval_dataset_name = _get_data_name(each_eval_data)
        eval_dataset[eval_dataset_name] = _eval_dataset
        logger.info(f"Eval Dataset [{i}]: {each_eval_data}\n{_eval_dataset}")
    args.eval_data = eval_data  # NOTE: overwrite previous eval_data

    if args.train_data_interleave_probabilities is not None and len(train_dataset) != len(
        args.train_data_interleave_probabilities
    ):
        raise ValueError(
            f"train_data_interleave_probabilities must have the same length as train_data, got {len(train_dataset)} and {len(args.train_data_interleave_probabilities)}"
        )
    # NOTE(xiaoke): Expected a list of Dataset objects or a list of IterableDataset objects.
    if len(train_dataset) > 0:
        if args.train_data_interleave_probabilities is None:
            logger.warning(
                "train_data_interleave_probabilities is not provided, "
                "the resulting dataset will have max_length_datasets*nb_dataset samples. "
                "As we use `all_exhausted` stopping strategy which is a oversampling strategy."
            )
        else:
            if sum(args.train_data_interleave_probabilities) != 1.0:
                logger.info(f"Normalize train_data_interleave_probabilities to sum to 1.0")
                args.train_data_interleave_probabilities = [
                    each_prob / sum(args.train_data_interleave_probabilities)
                    for each_prob in args.train_data_interleave_probabilities
                ]
                logger.info(f"train_data_interleave_probabilities: {args.train_data_interleave_probabilities}")
        # NOTE(xiaoke): Accourding to `datasets/src/datasets/arrow_dataset.py:_interleave_map_style_datasets:6079` and
        # `Breadcrumbsdatasets/src/datasets/iterable_dataset.py:_interleave_iterable_datasets:2293`
        train_dataset = interleave_datasets(
            train_dataset,
            probabilities=args.train_data_interleave_probabilities,
            seed=args.training.seed,
            stopping_strategy="all_exhausted",
        )
    else:
        train_dataset = None

    logger.info(f"Train Dataset: {train_dataset}")
    logger.info(f"Eval Dataset: {eval_dataset}")
    return train_dataset, eval_dataset


def _get_data_name(dataset_config_dict):
    # NOTE: path is the path for data script
    path = dataset_config_dict.path
    path_name = os.path.splitext(os.path.basename(path))[0]
    name = dataset_config_dict.name
    split = dataset_config_dict.split
    return f"{path_name}-{name}-{split}"


if __name__ == "__main__":
    main()
