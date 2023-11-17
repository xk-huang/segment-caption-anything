import os
import os.path as osp
from pathlib import Path
import sys
import json
import logging
from pprint import pformat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_trainer_state_path(train_state_path):
    with open(train_state_path, "r") as f:
        train_state = json.load(f)
    global_step = int(train_state["global_step"])
    try:
        best_metric = train_state["best_metric"]
        best_model_checkpoint = train_state["best_model_checkpoint"]
        best_model_step = int(Path(best_model_checkpoint).stem.split("-")[-1])
        return dict(
            global_step=global_step,
            best_metric=best_metric,
            best_model_step=best_model_step,
            best_model_checkpoint=best_model_checkpoint,
        )
    except Exception as e:
        logger.error(f"Error: {e}. So we use global_step as best_model_step.")
        return dict(global_step=global_step, best_metric=None, best_model_step=global_step, best_model_checkpoint=None)


def get_model_step(output_dir, ckpt_type):
    if ckpt_type not in ["best", "last"]:
        raise ValueError("ckpt_type must be one of [best, last], but got {}".format(ckpt_type))

    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise ValueError("Output dir does not exist: {}".format(output_dir))

    trainer_state_paths = list(output_dir.glob("*/trainer_state.json"))
    # NOTE: to solve the problem, when there are multiple stages. Thus starge-1/trainer_state.json and stage-2/trainer_state.json, the latter one will be used.
    trainer_state_paths = sorted(trainer_state_paths, reverse=True)
    if len(trainer_state_paths) == 0:
        raise ValueError("No trainer_state.json found in {}".format(output_dir))

    trainer_state_paths_mapping = {}
    for trainer_state_path in trainer_state_paths:
        trainer_state_paths_mapping[trainer_state_path] = parse_trainer_state_path(trainer_state_path)

    if ckpt_type == "last":
        last_trainer_state_path = max(
            trainer_state_paths_mapping, key=lambda x: int(trainer_state_paths_mapping[x]["global_step"])
        )
        last_trainer_state = trainer_state_paths_mapping[last_trainer_state_path]
        logger.info(
            f"Last trainer state path: {last_trainer_state_path}\nlast trainer state: {pformat(last_trainer_state)}"
        )
        last_step = last_trainer_state["global_step"]
        return last_step
    elif ckpt_type == "best":
        best_trainer_state_path = max(
            trainer_state_paths_mapping, key=lambda x: int(trainer_state_paths_mapping[x]["best_model_step"])
        )  # NOTE: the later best model is the global best one.
        best_trainer_state = trainer_state_paths_mapping[best_trainer_state_path]
        logger.info(
            f"Best trainer state path: {best_trainer_state_path}\nbest trainer state: {pformat(best_trainer_state)}"
        )
        best_step = best_trainer_state["best_model_step"]
        return best_step
    else:
        raise ValueError("ckpt_type must be one of [best, last], but got {}".format(ckpt_type))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise ValueError("Usage: python get_model_name_from_trainer_state.py <output_dir> <ckpt_type> {best, last}")
    output_dir = sys.argv[1]
    ckpt_type = sys.argv[2]
    best_step = get_model_step(output_dir, ckpt_type)
    print(best_step)
