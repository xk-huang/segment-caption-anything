# NOTE(xiaoke): Copy from gisting:src/integrations.py
"""Custom wandb integrations"""


import dataclasses
import os
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

import wandb
from transformers.integrations import TrainerCallback, WandbCallback
from transformers.utils import is_torch_tpu_available, logging
from omegaconf import OmegaConf

from .arguments import Arguments

logger = logging.get_logger(__name__)


class CustomWandbCallBack(WandbCallback):
    def __init__(self, custom_args: Arguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_args = custom_args

    def setup(self, args, state, model, **kwargs):
        # NOTE(xiaoke): Copy from gisting:src/integrations.py
        # NOTE(xiaoke): Copy from transformers/integrations.py, version 4.30.2
        del args
        args = self._custom_args

        if self._wandb is None:
            return

        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )

            # NOTE(xiaoke): Used in training sweep I guess. Should be removed.
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args["name"] = trial_name
                init_args["group"] = args.run_name

            # NOTE: use generate_id to identify each run https://github.com/wandb/wandb/issues/335#issuecomment-493284910
            if args.wandb.id is not None:
                logger.info(f"Resuming wandb run with {args.wandb.id}")
                id_ = args.wandb.id
            else:
                run_id_path = os.path.join(args.training.output_dir, "wandb_id")
                if not os.path.exists(run_id_path):
                    id_ = wandb.util.generate_id()
                    with open(os.path.join(run_id_path), "w") as f:
                        f.write(id_)
                    logger.info(f"Creating wandb run with {id_} and saving to {run_id_path}")
                else:
                    with open(os.path.join(run_id_path), "r") as f:
                        id_ = f.read()
                    logger.info(f"Resuming wandb run with {id_} from {run_id_path}")

            if self._wandb.run is None:
                self._wandb.init(
                    project=args.wandb.project,
                    group=args.wandb.group,
                    name=args.wandb.name,
                    config=OmegaConf.to_container(args),
                    dir=args.training.output_dir,
                    resume=args.wandb.resume,
                    id=id_,
                )

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            _watch_model = os.getenv("WANDB_WATCH", "false")
            if not is_torch_tpu_available() and _watch_model in ("all", "parameters", "gradients"):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, args.logging_steps))


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        # NOTE(xiaoke)
        if state.global_step == 0:
            control.should_evaluate = True


# NOTE: The logging system of transformers is incompatible with wandb.
# So we need to write a custom callback to log the metrics to our log files.
class LoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)


class EvalLossCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return super().on_evaluate(args, state, control, **kwargs)
