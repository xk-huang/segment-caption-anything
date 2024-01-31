import logging
import os
from typing import Optional, List, Dict, Union, Tuple, Any, NamedTuple, Mapping
import time
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import hydra
from hydra.utils import instantiate
from datasets import DatasetDict, load_dataset, IterableDatasetDict
from omegaconf import DictConfig, OmegaConf
from .data.transforms import SamCaptionerDataTransform
from .data.collator import SamCaptionerDataCollator
from .arguments import Arguments, global_setup, SAMCaptionerModelArguments, SCAModelArguments
from .models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor

from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed, Seq2SeqTrainer, GenerationConfig
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.trainer import (
    speed_metrics,
    deepspeed_init,
    is_torch_tpu_available,
    has_length,
    find_batch_size,
    nested_concat,
    nested_numpify,
    IterableDatasetShard,
    EvalLoopOutput,
    denumpify_detensorize,
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    Trainer,
    EvalPrediction,
    TrainerState,
    deepspeed_load_checkpoint,
    get_model_param_count,
    TRAINER_STATE_NAME,
    skip_first_batches,
    sys,
    HPSearchBackend,
    hp_params,
    RandomSampler,
    is_torch_less_than_1_11,
    ParallelMode,
    dist,
    shutil,
    TrainOutput,
    PREFIX_CHECKPOINT_DIR,
    SCHEDULER_NAME,
    SCALER_NAME,
    reissue_pt_warnings,
)
from functools import wraps
from collections import defaultdict

try:
    from transformers.trainer import xm, met, pl
except ImportError:
    pass
try:
    from transformers.trainer import amp
except ImportError:
    pass
try:
    from transformers.trainer import smp_forward_backward
except ImportError:
    pass
try:
    from transformers.trainer import smp
except ImportError:
    pass
try:
    from transformers.trainer import OSS
except ImportError:
    pass
# NOTE: bump transformers from 4.30.2 to 4.36.2
try:
    from transformers.trainer import (
        ShardedDDPOption,
        nested_truncate,
        tqdm,
        DistributedSampler,
    )
except ImportError:
    pass
try:
    from transformers.trainer_callback import TrainerCallback
except ImportError:
    pass
try:
    from transformers.trainer_seq2seq import is_deepspeed_zero3_enabled
except ImportError:
    pass


# NOTE: Fix the resume of DS optimizer + HF scheduler. https://github.com/huggingface/transformers/pull/25863/files
def is_deepspeed_available():
    return importlib.util.find_spec("deepspeed") is not None


import importlib.util
import warnings

if is_deepspeed_available():
    from accelerate.utils import DeepSpeedSchedulerWrapper

logger = logging.getLogger(__name__)

SAVING_FINISHED_FLAG = "saving_finished.flag"


class InferenceLoopOutput(NamedTuple):
    logits: Optional[Dict]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metadata: Optional[Dict]
    batch_num_regions_shape: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class FunctionTimers:
    def __init__(self):
        import time
        import numpy as np

        self.timers = defaultdict(list)

    def get_timer(self, f):
        @wraps(f)
        def _decorate(*args, **kwargs):
            start = time.perf_counter()
            ret = f(*args, **kwargs)
            end = time.perf_counter()
            if f.__name__ not in self.timers:
                self.timers[f.__name__] = []
            self.timers[f.__name__].append((end - start) * 1000)
            return ret

        return _decorate

    def clear(self):
        for k in self.timers:
            self.timers[k] = []

    def report(self):
        return {f"{k}_in_ms": np.mean(v) for k, v in self.timers.items()}


class SCASeq2SeqTrainer(Seq2SeqTrainer):
    # NOTE(xiaoke): Modified. Based on transformers v4.30.2
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.use_legacy_prediction_loop is True:
            raise ValueError(
                f"Not support legacy `prediction loop` for {self.__class__.__name__}! "
                "As I do not override it for region caption task."
            )

        # NOTE: change length_penalty, and apply its to both the model and the language model.
        # NOTE: with 10 samples, length_penalty leads to loss of CIDER
        # generation_config = self.model.generation_config.to_dict()
        # generation_config["length_penalty"] = 0.0
        # generation_config.pop("_from_model_config")
        # self.model.generation_config = GenerationConfig(**generation_config)
        # self.model.language_model.generation_config = self.model.generation_config
        # logger.info(f"generation_config: {self.model.generation_config}")

        self.function_timers = FunctionTimers()
        self._prepare_inputs = self.function_timers.get_timer(self._prepare_inputs)
        self.compute_loss = self.function_timers.get_timer(self.compute_loss)
        self._do_backward = self.function_timers.get_timer(self._do_backward)
        self.training_step = self.function_timers.get_timer(self.training_step)

        # NOTE: define the compute_metric_func
        # NOTE: compute_metrics = None triggers the default `prediction_loss_only=True`
        # NOTE: compute_metrics should be a function, but we define the function in the trainer, so we use bool here to indicate the usage.
        # NOTE: only world process zero compute the metrics, otherwise it may leads to download error.
        if self.compute_metrics is True and self.is_world_process_zero():
            import evaluate

            self.compute_metrics_func = evaluate.load("meteor")
        else:
            self.compute_metrics_func = None

        # NOTE: bump transformers from 4.30.2 to 4.36.2
        if not hasattr(self, "is_fsdp_xla_enabled"):
            self.is_fsdp_xla_enabled = False

    # Copied from `Trainer`
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            param_group_key = ["full"] + list(self.args.custom_param_lrs.keys())
            param_group_values = self._get_learning_rate()
            # NOTE: only keep the even idxs, because each group is divided into two sub ones, e.g., (lr_w_wd, lr_wo_wd).
            param_group_values = [v for idx, v in enumerate(param_group_values) if idx % 2 == 0]
            for k, v in zip(param_group_key, param_group_values):
                logs[f"learning_rate/{k}"] = v

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            logs.update(self.function_timers.report())
            self.function_timers.clear()
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
                # NOTE: add metric loss for best ckpt saving.
                metrics_loss = {k: v for k, v in metrics.items() if k.startswith("eval_") and k.endswith("_loss")}
                metrics["eval_loss"] = sum(metrics_loss.values()) / len(metrics_loss)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # NOTE: to handel empty batch during training due to LSJ augmentation.
        # We set `inputs` to None in `training_step` when the batch is empty.
        if inputs is None:
            logger.error("The inputs shouldn't be None in training! Thus we skip this batch of data.")
            return torch.tensor(torch.nan)

        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self._do_backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def _do_backward(self, loss):
        # NOTE: bump transformers from 4.30.2 to 4.36.2
        # sharded_ddp for fairseq was deprecated.
        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

    # NOTE: START OF INFERENCE CODE
    # Call order:
    # 1. inference
    # 2. inference_loop
    # 3. inference_step

    # use generate and save the outputs
    def inference(
        self,
        inference_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "inference",
        **gen_kwargs,
    ):
        # NOTE(xiaoke): Modified. Check the tokenizer and the unk_token_id first
        # We do not want to encounter the error after all the predicions are generated
        if self.tokenizer is None:
            raise ValueError("You need to specify a tokenizer in Trainer!")
        if self.tokenizer.unk_token_id is None:
            raise ValueError(f"Check the tokenizer! unk_token_id is None! {self.tokenizer}")

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(inference_dataset)
        start_time = time.time()

        output = self.inference_loop(
            eval_dataloader,
            description="Inference",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            skip_predcition_loss_after_generate=True,
        )

        (
            batch_num_regions,
            gt_captions,
            pred_captions,
            metadata_with_num_regions_length,
            logits_with_num_regions_length,
        ) = self._decode_inference_outputs(output)

        self._save_inference_json(
            metric_key_prefix,
            batch_num_regions,
            gt_captions,
            pred_captions,
            metadata_with_num_regions_length,
            logits_with_num_regions_length,
        )

    def _decode_inference_outputs(self, output):
        # NOTE(xiaoke): Modified. Dispatch the logits.
        #   - `generated_tokens`: (batch_size, num_regions, num_heads, token_max_length)
        #   - `iou_scores`: (batch_size, num_regions, num_heads)
        # Remove metrics update

        logits = output.logits  # Dict[str, (batch_num_regions, num_heads, ...)]
        label_ids = output.label_ids  # (batch_num_regions, token_max_length)
        metadata = output.metadata  # Dict[str, (batch_num_regions, ...)]

        # NOTE: generated_tokens is removed from logits, we only have `iou_scores` left
        generate_ids = logits.pop("generated_tokens")  # (batch_num_regions, num_heads, token_max_length)

        # NOTE(xiaoke): since we pad the labels with -100, we need to cast them back to unk_token_id
        # we believe there is always a tokenizer.unk_token_id in the tokenizer
        # Avoid error OverflowError: out of range integral type conversion attempted
        # https://github.com/huggingface/transformers/issues/22634#issuecomment-1500429811
        generate_ids = self._change_loss_token_to_unk_token(generate_ids, unk_token_id=self.tokenizer.unk_token_id)
        label_ids = self._change_loss_token_to_unk_token(label_ids, unk_token_id=self.tokenizer.unk_token_id)
        # NOTE(xiaoke): process generate_ids
        batch_num_regions, num_heads, token_max_length = generate_ids.shape
        generate_ids = generate_ids.reshape(batch_num_regions * num_heads, token_max_length)
        pred_captions = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        # NOTE(xiaoke): process label_ids
        if batch_num_regions != label_ids.shape[0]:
            raise ValueError(f"batch_num_regions {batch_num_regions} != label_ids.shape[0] {label_ids.shape[0]}")
        gt_captions = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_captions = np.array(pred_captions, dtype=object).reshape(batch_num_regions, num_heads).tolist()
        # NOTE(xiaoke): we asuume there is only ONE gt caption for each region
        gt_captions = np.array(gt_captions, dtype=object).reshape(batch_num_regions, 1).tolist()

        metadata_with_num_regions_length = {}
        for k, v in metadata.items():
            if len(v) != batch_num_regions:
                logger.warning(
                    f"metadata {k} has length {len(v)}, but batch_num_regions is {batch_num_regions}, so skip it"
                )
            else:
                metadata_with_num_regions_length[k] = v.tolist()  # json does not support numpy type object
        logits_with_num_regions_length = {}
        for k, v in logits.items():
            if len(v) != batch_num_regions:
                logger.warning(f"logits {k} has length {len(v)}, but batch_num_regions is {batch_num_regions}")
            else:
                logits_with_num_regions_length[k] = v.tolist()  # json does not support numpy type object

        return (
            batch_num_regions,
            gt_captions,
            pred_captions,
            metadata_with_num_regions_length,
            logits_with_num_regions_length,
        )

    def _save_inference_json(
        self,
        metric_key_prefix,
        batch_num_regions,
        gt_captions,
        pred_captions,
        metadata_with_num_regions_length,
        logits_with_num_regions_length,
    ):
        # NOTE(xiaoke): the output json follows the format of https://github.com/CannyLab/vdtk
        output_json = []
        for idx in range(batch_num_regions):
            output_json.append(
                {
                    "_id": idx,
                    "split": "inference",
                    "references": gt_captions[idx],
                    "candidates": pred_captions[idx],
                    "metadata": {k: v[idx] for k, v in metadata_with_num_regions_length.items()},
                    "logits": {k: v[idx] for k, v in logits_with_num_regions_length.items()},
                }
            )

        import json

        infer_json_dir = os.path.join(self.args.output_dir, "infer")
        os.makedirs(infer_json_dir, exist_ok=True)
        infer_json_file = os.path.join(infer_json_dir, f"infer-{metric_key_prefix}.json")
        # TODO: only the very first process will write the file
        if self.is_world_process_zero():
            with open(infer_json_file, "w") as f:
                json.dump(output_json, f, indent=4)

    @staticmethod
    def _change_loss_token_to_unk_token(tokens, unk_token_id, padding_index=-100):
        tokens[tokens == padding_index] = unk_token_id
        return tokens

    def inference_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        skip_predcition_loss_after_generate: Optional[bool] = None,
    ) -> InferenceLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.model_wrapped is self.model:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # NOTE: otherwise we will get the OOM due to fp32.
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Num examples for process ({self.args.process_index}) = {len(dataloader) * batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        # NOTE(xiaoke): Modified. We need to save the inputs for ids
        metadata_host = None
        batch_num_regions_shape_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # NOTE(xiaoke): Modified. We need to save the inputs for ids
        all_metadata = None
        all_batch_num_regions_shape = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0

        # Main inference loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # NOTE(xiaoke): Modified. We need to save the inputs for ids
            metadata = None
            for k, v in inputs.items():
                if k.startswith("metadata_") and isinstance(v, torch.Tensor):
                    if metadata is None:
                        metadata = {}
                    # metadata[k] = v.flatten(0, 1) if len(v.shape) > 1 else v
                    metadata[k] = v
            metadata = self._prepare_input(metadata)

            # NOTE: skip_predcition_loss_after_generate=True
            # Prediction step
            loss, logits, batch_num_regions_shape, labels = self.inference_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
                skip_predcition_loss_after_generate=skip_predcition_loss_after_generate,
            )
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # NOTE(xiaoke): While our outputs has four dim `(PADDED_batch_size, PADDED_num_regions, token_length, ...)`,
            # and we squash the first two dims. In the `gather` function, we assume the tensors has the same shape.
            # Thus, we need to `unsqueeze` one dim ahead, and recover the results with the batch_size-num_regions number pair.

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                # # NOTE(xiaoke): Modified. PRETEND its shape is (batch_size, token_length, ...) which the trainer expects.
                # # NOTE(xiaoke): we do not add the `num_heads` dim, since they are the same across the batch.
                # # Thus taking their mean is the same as with multiple heads.
                # assert len(batch_num_regions_shape) == 1
                # losses = loss.repeat(batch_num_regions_shape[0].tolist())
                # losses = self._pad_across_processes(losses)
                # losses = self._nested_gather(losses)
                # # NOTE(xiaoke): Modified. We need to pad the `token_length` dim, since they may be different across batches
                # losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)

                # NOTE: compute batch-wise average loss
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                # NOTE: bump transformers from 4.30.2 to 4.36.2
                # labels = self._pad_across_processes(labels)
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                # NOTE: bump transformers from 4.30.2 to 4.36.2
                # inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                # NOTE: bump transformers from 4.30.2 to 4.36.2
                # logits = self._pad_across_processes(logits)
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            # NOTE(xiaoke): Modified. We need to save the inputs for ids
            if metadata is not None:
                # NOTE: bump transformers from 4.30.2 to 4.36.2
                # metadata = self._pad_across_processes(metadata)
                metadata = self.accelerator.pad_across_processes(metadata, dim=1, pad_index=-100)
                metadata = self._nested_gather(metadata)
                metadata_host = (
                    metadata if metadata_host is None else nested_concat(metadata_host, metadata, padding_index=-100)
                )
            # NOTE(xiaoke): Modified. We need to save the batch-num_regions shape to recover the results
            if batch_num_regions_shape is not None:
                batch_num_regions_shape = self._nested_gather(batch_num_regions_shape)
                batch_num_regions_shape_host = (
                    batch_num_regions_shape
                    if batch_num_regions_shape_host is None
                    else torch.concat((batch_num_regions_shape_host, batch_num_regions_shape), dim=0)
                )
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                # NOTE(xiaoke): Modified. We need to save the inputs for ids
                if metadata_host is not None:
                    metadata = nested_numpify(metadata_host)
                    all_metadata = (
                        metadata if all_metadata is None else nested_concat(all_metadata, metadata, padding_index=-100)
                    )
                # NOTE(xiaoke): Modified. We need to save the batch-num_regions shape to recover the results
                if batch_num_regions_shape_host is not None:
                    batch_num_regions_shape = nested_numpify(batch_num_regions_shape_host)
                    all_batch_num_regions_shape = (
                        batch_num_regions_shape
                        if all_batch_num_regions_shape is None
                        else torch.concat(all_batch_num_regions_shape, batch_num_regions_shape, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None
                # NOTE(xiaoke): Modified. We need to save the inputs for ids
                metadata_host = None
                batch_num_regions_shape_host = None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        # NOTE(xiaoke): Modified. We need to save the inputs for ids
        if metadata_host is not None:
            metadata = nested_numpify(metadata_host)
            all_metadata = (
                metadata if all_metadata is None else nested_concat(all_metadata, metadata, padding_index=-100)
            )
        # NOTE(xiaoke): Modified. We need to save the batch-num_regions shape to recover the results
        if batch_num_regions_shape_host is not None:
            batch_num_regions_shape = nested_numpify(batch_num_regions_shape_host)
            all_batch_num_regions_shape = (
                batch_num_regions_shape
                if all_batch_num_regions_shape is None
                else nested_concat(all_batch_num_regions_shape, batch_num_regions_shape, padding_index=-100)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                # NOTE(xiaoke): Modified. Log wrong number of samples
                logger.warning(
                    f"Your dataset doesn't implement `__len__`. Use dataloader instead, Inference will not check all elements."
                )
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                # NOTE(xiaoke): Modified. Log wrong number of samples
                logger.warning(
                    f"Your dataset doesn't implement `__len__`. Use one process observed data. Inference will not check all elements."
                )
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # NOTE(xiaoke): Modified. In region caption task, the prediction has two batch dimensions,
        # the first one is the batch size, the second one is the number of regions.
        # we need to truncate the results based on both dims.
        #   - all_batch_num_regions_shape: (batch_steps, 2), one batch_step has a batch of data
        #   - all_losses:  (PADDED_batch_size, PADDED_num_regions)
        #   - all_preds:   (PADDED_batch_size, PADDED_num_regions, num_heads, PADDED_token_length), a.k.a., all_generate_ids
        #   - all_labels:  (PADDED_batch_size, PADDED_num_regions, PADDED_token_length)
        #   - all_metadata (PADDED_batch_size, PADDED_num_regions, ...)

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        # NOTE(xiaoke): Modified. We truncate the results based both the batch size and the number of regions
        # (batch_size, PADDED_num_regions)
        if all_losses is not None:
            # all_losses = nested_two_dims_truncate_and_flatten(all_losses, all_batch_num_regions_shape, num_samples)
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_two_dims_truncate_and_flatten(all_preds, all_batch_num_regions_shape, num_samples)
        if all_labels is not None:
            all_labels = nested_two_dims_truncate_and_flatten(all_labels, all_batch_num_regions_shape, num_samples)
        if all_inputs is not None:
            all_inputs = nested_two_dims_truncate_and_flatten(all_inputs, all_batch_num_regions_shape, num_samples)
        # NOTE(xiaoke): Modified. We need to save the inputs for ids
        if all_metadata is not None:
            all_metadata = nested_two_dims_truncate_and_flatten(all_metadata, all_batch_num_regions_shape, num_samples)

        # Metrics!
        metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # NOTE(xiaoke): Modified. Skip the metric computation
        return InferenceLoopOutput(
            logits=all_preds,
            label_ids=all_labels,
            metadata=all_metadata,
            batch_num_regions_shape=all_batch_num_regions_shape,
            metrics=metrics,
            num_samples=num_samples,
        )

    def inference_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        skip_predcition_loss_after_generate: Optional[bool] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        # NOTE: `prediction_loss_only` is always False in inference
        if not self.args.predict_with_generate or prediction_loss_only:
            # TODO(xiaoke): replace `super().inference_step` with batch-region `region_caption_prediction_step`
            # we need `batch_num_regions_shape` to truncate the results
            # remember to add a todo in loss computation, as the mask loss is not added!
            loss, logits, labels = super(Seq2SeqTrainer, self).prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
            batch_num_regions_shape = torch.tensor(inputs["input_ids"].shape[:2]).unsqueeze(0).to(device=loss.device)
            return loss, logits, batch_num_regions_shape, labels
            # return super().prediction_step(
            #     model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            # )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in inputs
            and "decoder_input_ids" in inputs
            and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        # TODO(xiaoke): the generate should return both the generated tokens and the masks
        # We need to change both this `*_step` and the `*_loop`
        # FIXME(xiaoke): the genearte is not warpped by self.compute_loss_context_manager()
        # which could cause problem in sharded distributed inference. The `prediction_step` used in `*_loop` is affected too.
        # NOTE(xiaoke): Modified. Adapt for region caption task and chunk inference to reduce memory consumption.
        inputs = self._prepare_input_dtype(inputs, self.model.dtype)  # NOTE: for fp16 inference
        generated_outputs = self._generate_in_inference_step(inputs, gen_kwargs)
        generated_tokens = generated_outputs.sequences
        iou_scores = generated_outputs.iou_scores
        pred_masks = generated_outputs.pred_masks

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded

        # NOTE(xiaoke): Modified. For region caption task, the shape of the generated tokens
        # is (batch_size, num_regions, num_heads, token_max_length), we use the modified `_pad_tensors_to_max_len`
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        # NOTE: Compute loss after generate
        with torch.no_grad():
            if has_labels and skip_predcition_loss_after_generate is not True:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        # NOTE(xiaoke): Modified. We record the batch size and num_regions
        # for truncation of distributed evaluation.
        batch_num_regions_shape = torch.tensor(generated_tokens.shape[:2]).unsqueeze(0).to(generated_tokens)

        if self.args.prediction_loss_only:
            return loss, None, batch_num_regions_shape, None

        if has_labels:
            labels = inputs["labels"]
            # NOTE(xiaoke): Modified. For region caption task, the shape of
            # the labels is (batch_size, num_regions, token_max_length)
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        # TODO(xiaoke): compute and return the metrics for vision tasks here.
        # the dense logits have shape of (batch_size, num_regions, num_heads, ...)
        # which is very memory consuming. e.g., 230k*3*256*256=45GB
        logits = dict(generated_tokens=generated_tokens, iou_scores=iou_scores)
        return loss, logits, batch_num_regions_shape, labels

    PROMPT_TYPES_TO_ABLATE_ON_VG = ["center_point_in_box", "random_point_in_box", "random_point_in_mask", None]
    SAM_IMAGE_PROCESSOR = None

    def _generate_in_inference_step(self, inputs, gen_kwargs):
        prompt_types_to_ablate_on_vg = getattr(self.args, "prompt_types_to_ablate_on_vg", None)

        if prompt_types_to_ablate_on_vg not in self.PROMPT_TYPES_TO_ABLATE_ON_VG:
            raise ValueError(
                f"prompt_types_to_ablate_on_vg is {prompt_types_to_ablate_on_vg}. It should be one of {self.PROMPT_TYPES_TO_ABLATE_ON_VG}"
            )

        if prompt_types_to_ablate_on_vg == "center_point_in_box":
            logger.debug("prompt types is [center_point_in_box] to ablate on VG")
            input_boxes = inputs["input_boxes"]

            center_points_x = input_boxes[:, :, [0, 2]].mean(dim=-1)
            center_points_y = input_boxes[:, :, [1, 3]].mean(dim=-1)
            center_points = torch.stack((center_points_x, center_points_y), dim=-1)
            center_points = center_points.unsqueeze(-2)

            inputs["input_points"] = center_points
            inputs["input_boxes"] = None

        elif prompt_types_to_ablate_on_vg == "random_point_in_box":
            logger.debug("prompt types is [random_point_in_box] to ablate on VG")
            input_boxes = inputs["input_boxes"]

            # NOTE: Uniformly sample a point in the box, the shape of the box is (batch_size, num_regions, 4), the coordinate are xyxy.
            # NOTE: the shape of the point is (batch_size, num_regions, 1, 2), the coordinates are xy.
            random_points = torch.rand(input_boxes.shape[:2] + (2,), device=input_boxes.device)
            # NOTE: the shape of the point is (batch_size, num_regions, 2)
            random_points = input_boxes[:, :, [0, 1]] + random_points * (
                input_boxes[:, :, [2, 3]] - input_boxes[:, :, [0, 1]]
            )
            random_points = random_points.unsqueeze(-2)

            inputs["input_points"] = random_points
            inputs["input_boxes"] = None

        elif prompt_types_to_ablate_on_vg == "random_point_in_mask":
            logger.debug("prompt types is [random_point_in_mask] to ablate on VG")
            if self.SAM_IMAGE_PROCESSOR is None:
                from src.models.sam.image_processing_sam import SamImageProcessor

                self.SAM_IMAGE_PROCESSOR = SamImageProcessor()

            # NOTE: generate the binary mask
            generated_outputs = self.model.generate(
                generate_chunk_size=getattr(self.args, "generate_chunk_size"), **inputs, **gen_kwargs
            )
            iou_scores = generated_outputs.iou_scores  # (batch_size, num_regions, 3)
            iou_scores_max_head = iou_scores.argmax(dim=-1)  # (batch_size, num_regions)
            pred_masks = generated_outputs.pred_masks  # (batch_size, num_regions, 3, H, W)

            # NOTE: A list of binary masks, List[torch.Tensor]]: list shape (batch_size), bool tensor shape (num_regions, num_heads, H, W)
            masks = self.SAM_IMAGE_PROCESSOR.post_process_masks(
                pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
            )

            # NOTE: Sample random point in each masks
            input_boxes = inputs["input_boxes"]
            dtype = input_boxes.dtype
            center_points_x = input_boxes[:, :, [0, 2]].mean(dim=-1)
            center_points_y = input_boxes[:, :, [1, 3]].mean(dim=-1)
            center_points = torch.stack((center_points_x, center_points_y), dim=-1)

            random_points = []
            for batch_idx, batch_masks in enumerate(masks):
                resized_scale = inputs["reshaped_input_sizes"][batch_idx] / inputs["original_sizes"][batch_idx]

                batch_iou_scores_max_head = iou_scores_max_head[batch_idx]  # (num_regions)
                batch_masks  # (num_regions, num_heads, H, W)
                max_indices = batch_iou_scores_max_head.view(-1, 1, 1, 1).expand(
                    -1, 1, batch_masks.size(2), batch_masks.size(3)
                )
                # NOTE: gather do not support multi-dim indexing, so we need to flatten the first dim
                max_confidence_masks = batch_masks.gather(1, max_indices).squeeze(1)

                # NOTE: for debug
                # for i in range(len(max_confidence_masks)):
                #     assert torch.allclose(max_confidence_masks[i], batch_masks[i, batch_iou_scores_max_head[i]])

                batch_random_points = []
                for region_id, mask in enumerate(max_confidence_masks):
                    # NOTE: Find the indices of all True values in the mask
                    # NOTE: the index is yx, we need to flip it
                    true_indices = mask.nonzero(as_tuple=False).to(dtype=dtype)  # Shape: [num_true_points, 2]
                    true_indices = torch.flip(true_indices, dims=[-1])  # Shape: [num_true_points, 2]

                    if len(true_indices) > 0:
                        selected_index = true_indices[torch.randint(0, len(true_indices), ())]
                        # NOTE: scale it as `input_boxes` and `input_points` are scaled to 1024 in the image preprocessor of SAM.
                        selected_index = selected_index * resized_scale

                        batch_random_points.append(selected_index)
                    else:
                        # In case there are no True values in the mask, append None or a placeholder
                        logger.error("No True values in the mask!")
                        batch_random_points.append(center_points[batch_idx, region_id])
                batch_random_points = torch.stack(batch_random_points, dim=0)
                random_points.append(batch_random_points)

            random_points = torch.stack(random_points, dim=0)
            random_points = random_points.unsqueeze(-2)

            inputs["input_points"] = random_points
            inputs["input_boxes"] = None

        else:
            logger.debug("prompt types is [null] to ablate on VG")

        generated_outputs = self.model.generate(
            generate_chunk_size=getattr(self.args, "generate_chunk_size"), **inputs, **gen_kwargs
        )
        return generated_outputs

    # NOTE: END OF INFERENCE CODE

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # NOTE(xiaoke): Modified. Check the shape, at least 1D
        # FIXME(xiaoke): use `atleast_1d` maybe better
        if len(tensor.shape) < 1:
            raise ValueError("Cannot pad tensors with fewer than one dimension")

        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        # NOTE(xiaoke): Modified. The original function taks tensor with shape (batch_size, number of tokens)
        # However, for region caption task, the shape of the tensor is (batch_size, num_regions, num_heads, number of tokens)
        # we need to pad the tensor along the last dim.
        # NOTE(xiaoke): This is a GENERALIZED version of the original function
        tensor_shape = tensor.shape
        padded_tensor = pad_token_id * torch.ones(
            (*tensor_shape[:-1], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[..., : tensor.shape[-1]] = tensor
        return padded_tensor

    # NOTE: START OF EVALUATION CODE
    # `Seq2SeqTrainer` is mostly about **`predict_with_generate`**. We set `predict_with_generate=True` in config by default.
    # The call order:
    # 1. `Seq2SeqTrainer.evaluate` add generate args like `max_length` and `num_beams`
    # 2. `Trainer.evaluate`. `prediction_loss_only=True` if self.compute_metrics is None, else `None` leads to `self.args.prediction_loss_only` is False,
    # 3. `Seq2SeqTrainer.prediction_step`.To reduce the call stack, use `super(Seq2SeqTrainer, self).prediction_step`, which is `Trainer.prediction_step`
    # 4. `Trainer.prediction_step`, due to `prediction_loss_only=True`.

    # NOTE: START OF CUSTOM EVALUATION CODE WITH INFERENCE

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ):
        # NOTE(xiaoke): Modified. Check the tokenizer and the unk_token_id first
        # We do not want to encounter the error after all the predicions are generated
        if self.tokenizer is None:
            raise ValueError("You need to specify a tokenizer in Trainer!")
        if self.tokenizer.unk_token_id is None:
            raise ValueError(f"Check the tokenizer! unk_token_id is None! {self.tokenizer}")

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.inference_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            prediction_loss_only=True if self.compute_metrics is None else None,
            # prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            skip_predcition_loss_after_generate=False,
        )

        # Metrics!
        if self.compute_metrics is not None and output.logits is not None:
            (
                batch_num_regions,
                gt_captions,
                pred_captions,
                metadata_with_num_regions_length,
                logits_with_num_regions_length,
            ) = self._decode_inference_outputs(output)
            num_heads = max(len(gt_captions[0]), len(pred_captions[0]))

            def _repeat_and_flatten(list_, num_heads):
                ret_list = []
                for sub_list in list_:
                    sub_list += [sub_list[-1]] * (num_heads - len(sub_list))
                    ret_list += sub_list
                return ret_list

            gt_captions = _repeat_and_flatten(gt_captions, num_heads)
            pred_captions = _repeat_and_flatten(pred_captions, num_heads)

            if self.compute_metrics_func is not None:
                # NOTE: only the world process zero evaluate the metrics
                metrics = self.compute_metrics_func.compute(predictions=pred_captions, references=gt_captions)
                # To be JSON-serializable, we need to remove numpy types or zero-d tensors
                metrics = denumpify_detensorize(metrics)
                # Prefix all keys with metric_key_prefix + '_'
                for key in list(metrics.keys()):
                    if not key.startswith(f"{metric_key_prefix}_"):
                        metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                output.metrics.update(metrics)

        # Copy from: Trainer.evaluate
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    # NOTE: END OF EVALUATION CODE

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        # NOTE(xiaoke): Modified. mtime are all the same when running on Azure Sigunlarity.
        # On Azure AMLK8s, they work well.
        super()._rotate_checkpoints(use_mtime=False, output_dir=output_dir)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = []

            custom_param_lrs = self.args.custom_param_lrs
            logger.debug(f"[Optimizer] default param ls: {self.args.learning_rate}")
            optimizer_grouped_parameters += self._create_grouped_parameters(
                opt_model, self.args.learning_rate, custom_param_lrs
            )
            for filtered_param, lr in custom_param_lrs.items():
                logger.debug(f"[Optimizer] param {filtered_param} will use lr {lr}")
                optimizer_grouped_parameters += self._create_grouped_parameters(
                    get_parameter_by_name(opt_model, filtered_param), lr
                )

            num_params_each_group = [len(g["params"]) for g in optimizer_grouped_parameters]
            all_optimizable_params = list(filter(lambda p: p.requires_grad, opt_model.parameters()))
            if sum(num_params_each_group) != len(all_optimizable_params):
                raise ValueError(
                    f"num_params_each_group != all_optimizable_params ({sum(num_params_each_group)} vs. {len(all_optimizable_params)}), which should not happened."
                )

            logger.info(
                f"[Optimizer] num of param groups: {len(optimizer_grouped_parameters)}, these group has {num_params_each_group} params"
            )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # NOTE: bump transformers from 4.30.2 to 4.36.2
            # NOTE: deprecate fairscale's ShardedDDP, https://github.com/huggingface/transformers/pull/24825
            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def _create_grouped_parameters(self, opt_model, lr, filter_keys=None):
        full_parameters = list(opt_model.named_parameters())
        if (filter_keys is None) or len(filter_keys) == 0:
            logger.debug(f"[Optimizer] no filter keys, using all {len(full_parameters)} params")
            filtered_parameters = []
        else:
            filtered_parameters = get_parameters_names_by_keys(opt_model, filter_keys)
            logger.debug(f"[Optimizer] filtered out {len(filtered_parameters)} from {len(full_parameters)} params")

        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in full_parameters
                    if (n in decay_parameters and p.requires_grad and n not in filtered_parameters)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in full_parameters
                    if (n not in decay_parameters and p.requires_grad and n not in filtered_parameters)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

        return optimizer_grouped_parameters

    def _get_learning_rate(self) -> List[float]:
        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = [g["lr"] for g in self.optimizer.param_groups]
            else:
                last_lr = self.lr_scheduler.get_last_lr()
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr

    def _prepare_input_dtype(self, data: Union[torch.Tensor, Any], dtype) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input_dtype(v, dtype) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input_dtype(v, dtype) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            elif torch.is_floating_point(data) or torch.is_complex(data):
                kwargs.update({"dtype": dtype})
            return data.to(**kwargs)
        return data

    # NOTE: to handel empty batch during training due to LSJ augmentation.
    # We set `inputs` to None in `training_step` when the batch is empty.
    # When encounter the None inputs, we keep the step
    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # NOTE: bump transformers from 4.30.2 to 4.36.2
        # NOTE: deprecate fairscale's ShardedDDP, https://github.com/huggingface/transformers/pull/24825
        # delay_optimizer_creation = (
        #     self.sharded_ddp is not None
        #     and self.sharded_ddp != ShardedDDPOption.SIMPLE
        #     or is_sagemaker_mp_enabled()
        #     or self.fsdp is not None
        # )
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        # NOTE: bump transformers from 4.30.2 to 4.36.2
        # The arguments for on_step_end are moved from `args` to `state`
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # NOTE: bump transformers from 4.30.2 to 4.36.2
        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            try:
                if args.gradient_checkpointing_kwargs is None:
                    gradient_checkpointing_kwargs = {}
                else:
                    gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except AttributeError:
                self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            # NOTE: bump transformers from 4.30.2 to 4.36.2
            # if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            #     train_dataloader.sampler.set_epoch(epoch)
            # elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
            #     train_dataloader.dataset.set_epoch(epoch)
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                # NOTE: Modified here. We set `inputs` to None when the batch is empty. Skip this step.
                if inputs is None:
                    logger.warning("The inputs shouldn't be None in training! Thus we skip this batch of data.")
                    continue

                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        # NOTE: bump transformers from 4.30.2 to 4.36.2
                        # sharded_ddp for fairseq was deprecated.
                        # if self.do_grad_scaling:
                        #     # Reduce gradients first for XLA
                        #     if is_torch_tpu_available():
                        #         gradients = xm._fetch_gradients(self.optimizer)
                        #         xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                        #     # AMP: gradients need unscaling
                        #     self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        # NOTE: bump transformers from 4.30.2 to 4.36.2
                        # sharded_ddp for fairseq was deprecated.
                        # if self.do_grad_scaling:
                        #     self.scaler.step(self.optimizer)
                        #     self.scaler.update()
                        # else:
                        xm.optimizer_step(self.optimizer)
                    # elif self.do_grad_scaling:
                    #     scale_before = self.scaler.get_scale()
                    #     self.scaler.step(self.optimizer)
                    #     self.scaler.update()
                    #     scale_after = self.scaler.get_scale()
                    #     optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(
                            self.lr_scheduler,
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                        ):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _save_checkpoint(self, model, trial, metrics=None):
        # NOTE: Temporay fix multi-node saving bugs: https://github.com/huggingface/transformers/issues/27925#issuecomment-1869331349
        try:
            super()._save_checkpoint(model, trial, metrics=metrics)
        except FileNotFoundError:
            pass

        # NOTE: it is possible for partial saving which cannot be read.
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        open(os.path.join(output_dir, SAVING_FINISHED_FLAG), "a").close()

    # NOTE: Fix the resume of DS optimizer + HF scheduler. https://github.com/huggingface/transformers/pull/25863/files
    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
            return

        super()._load_optimizer_and_scheduler(checkpoint)


def nested_two_dims_truncate_and_flatten(tensors, batch_num_regions_shape, limits) -> List[torch.Tensor]:
    # NOTE(xiaoke): Modified. In region caption task, the prediction has two batch dimensions,
    # the first one is the batch size, the second one is the number of regions.
    # we need to truncate the results based on both dims.
    #   - all_batch_num_regions_shape: (batch_steps, 2), one batch_step has a batch of data
    #   - all_losses:  (PADDED_batch_size, PADDED_num_regions)
    #   - all_preds:   (PADDED_batch_size, PADDED_num_regions, num_heads, PADDED_token_length), a.k.a., all_generate_ids
    #   - all_labels:  (PADDED_batch_size, PADDED_num_regions, PADDED_token_length)
    #   - all_metadata (PADDED_batch_size, PADDED_num_regions, ...)
    "Truncate `tensors` at `limit` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_two_dims_truncate_and_flatten(t, batch_num_regions_shape, limits) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_two_dims_truncate_and_flatten(t, batch_num_regions_shape, limits) for k, t in tensors.items()}
        )

    if len(batch_num_regions_shape.shape) != 2:
        raise ValueError(f"batch_num_regions_shape should have two dims, got {batch_num_regions_shape.shape}")
    if batch_num_regions_shape[:, 0].sum() != len(tensors):
        raise ValueError(
            f"batch_num_regions_shape[:, 0].sum() should be equal to the length of tensors, "
            f"got {batch_num_regions_shape[:, 0].sum()} and {len(tensors)}"
        )
    list_tensors = []
    sample_start_idx = 0
    for num_samples, num_regions in batch_num_regions_shape:
        tensor = tensors[sample_start_idx : sample_start_idx + num_samples, :num_regions]
        tensor = tensor.reshape(-1, *tensor.shape[2:])
        list_tensors.append(tensor)
        sample_start_idx += num_samples

    return np.concatenate(list_tensors[:limits], axis=0)


def get_parameter_by_name(model, parameter_name):
    """
    Get the parameter object in a PyTorch model given its name.

    Args:
        model (nn.Module): The PyTorch model containing the parameter.
        parameter_name (str): The name of the parameter as a string, with dot notation.

    Returns:
        nn.Parameter: The parameter object.
    """
    parameter_name_parts = parameter_name.split(".")
    parameter_obj = model

    for part in parameter_name_parts:
        if part == "":
            continue
        parameter_obj = getattr(parameter_obj, part)

    return parameter_obj


def get_parameters_names_by_keys(opt_model, keys):
    return [name for name, _ in opt_model.named_parameters() if any(key in name for key in keys)]
