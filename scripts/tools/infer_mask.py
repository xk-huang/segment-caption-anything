import sys

sys.path.append(".")

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
)
from src.models.sam_captioner import SAMCaptionerConfig, SAMCaptionerModel, SAMCaptionerProcessor
from src.sca_seq2seq_trainer import SCASeq2SeqTrainer, get_parameter_by_name
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
)
from src.integrations import CustomWandbCallBack, EvaluateFirstStepCallback, LoggerCallback, EvalLossCallback
import src.models.sca
import src.utils

from transformers.trainer_utils import _re_checkpoint
from transformers import set_seed
import json
from src.train import prepare_datasets, prepare_data_transform
from transformers import SamModel
import torch
from collections.abc import Mapping
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pycocotools.mask
import sqlite3
from contextlib import closing
import multiprocessing as mp

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(args: DictConfig) -> None:
    # NOTE(xiaoke): follow https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

    logger.info(OmegaConf.to_yaml(args))
    args, training_args, model_args = global_setup(args)

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

    DataCollatorClass = None
    if isinstance(model_args, SAMCaptionerModelArguments):
        DataCollatorClass = SamCaptionerDataCollator
    elif isinstance(model_args, SCAModelBaseArguments):
        DataCollatorClass = SCADataCollator
    collate_fn = DataCollatorClass(processor.tokenizer)

    compute_metrics = training_args.compute_metrics
    if compute_metrics is not True:
        # NOTE: compute_metrics = None triggers the default `prediction_loss_only=True`
        # NOTE: compute_metrics should be a function, but we define the function in the trainer, so we use bool here to indicate the usage.
        compute_metrics = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model_name_or_path = model_args.sam_model_name_or_path
    # NOTE: Official SAM (4.29) only support BS=1, otherwise the masks are wrong.
    model = SamModel.from_pretrained(sam_model_name_or_path, cache_dir=model_args.cache_dir).to(device)
    logger.info(f"Load sam model from {sam_model_name_or_path}")

    # NOTE: max_samples=10 is for debugging
    max_samples = os.getenv("MAX_SAMPLES", None)
    if max_samples is not None:
        max_samples = int(max_samples)

    if training_args.do_eval or training_args.do_inference:
        for eval_dataset_name, eval_dataset_ in eval_dataset.items():
            saving_dir = os.path.join(training_args.output_dir, eval_dataset_name)
            os.makedirs(saving_dir, exist_ok=True)

            db_file = os.path.join(saving_dir, "results.db")

            # Initialize the SQLite database and start the save_results process
            init_database(db_file)

            # Create a queue to store the results and start the saving process
            result_queue = mp.Queue(maxsize=50)
            save_process = mp.Process(target=save_results, args=(result_queue, db_file))
            save_process.start()

            eval_dataloader = get_dataloader(eval_dataset_, collate_fn, training_args.dataloader_num_workers)
            for image_cnt, inputs in enumerate(tqdm.tqdm(eval_dataloader, desc="Evaluating")):
                if max_samples is not None and image_cnt == max_samples:
                    break
                image_id = inputs["metadata_image_id"][0][0].item()
                region_ids = inputs["metadata_region_id"][0].numpy()
                if all(result_exists(db_file, image_id, region_id) for region_id in region_ids):
                    continue
                inputs = _prepare_input(inputs, device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    masks = processor.sam_processor.image_processor.post_process_masks(
                        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
                    )
                    scores = outputs.iou_scores

                # NOTE: dim: image, region, ...
                # fig, axes = show_masks_on_image(inputs["images"][0], masks[0][0], scores[0][0])

                masks = masks[0].permute(0, 2, 3, 1).cpu().numpy()
                scores = scores[0].cpu().numpy()
                input_boxes = inputs["metadata_input_boxes"][0].cpu().numpy()
                gt_captions = inputs["metadata_captions"][0]

                result_queue.put(
                    dict(
                        image_cnt=image_cnt,
                        region_ids=region_ids,
                        image_id=image_id,
                        masks=masks,
                        scores=scores,
                        gt_captions=gt_captions,
                        input_boxes=input_boxes,
                    )
                )

            # Signal the saving process to finish
            result_queue.put(None)

            # Wait for the saving process to complete and get the results
            save_process.join()

            # This line is for debug.
            # save_results(result_queue, db_file)
            # json_file = os.path.join(saving_dir, "results.json")
            # convert_db_to_json(db_file, json_file)


def get_dataloader(dataset, collate_fn, num_workers):
    logger.info(f"Creating dataloader: num_workers: {num_workers}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return dataloader


def _prepare_input(data, device):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    return data


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach()
        axes[i].imshow(np.array(raw_image))
        show_mask(mask, axes[i])
        axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
        axes[i].axis("off")
    return fig, axes


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def save_results(queue, db_file):
    with closing(sqlite3.connect(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(region_cnt) FROM results")
        max_id = cursor.fetchone()[0]
        if max_id is None:
            region_cnt = 0
        else:
            region_cnt = max_id + 1

        while True:
            batch = queue.get()
            if batch is None:
                break

            image_cnt = batch["image_cnt"]
            region_ids = batch["region_ids"]
            if isinstance(region_ids, np.ndarray):
                region_ids = region_ids.tolist()
            image_id = batch["image_id"]
            masks = batch["masks"]
            scores = batch["scores"]
            gt_captions = batch["gt_captions"]
            input_boxes = batch["input_boxes"]
            if isinstance(input_boxes, np.ndarray):
                input_boxes = input_boxes.tolist()

            # region_masks = masks[0]
            # region_scores = scores[0]
            # rle_region_masks = pycocotools.mask.encode(np.asfortranarray(region_masks))
            # pycocotools.mask.decode(rle_region_masks)
            for region_id, masks_, scores_, gt_caption, input_box in zip(
                region_ids, masks, scores, gt_captions, input_boxes
            ):
                rle_region_masks = pycocotools.mask.encode(np.asfortranarray(masks_))
                for m in rle_region_masks:
                    m["counts"] = m["counts"].decode("ascii")
                scores_ls = scores_.tolist()

                result = (
                    region_cnt,
                    image_cnt,
                    region_id,
                    image_id,
                    json.dumps(rle_region_masks),
                    json.dumps(scores_ls),
                    json.dumps(input_box),
                    json.dumps(gt_caption),
                )
                region_cnt += 1

                conn.execute(
                    """  
                    INSERT INTO results ( 
                        region_cnt, image_cnt, region_id, image_id, masks, scores, input_box, gt_caption
                    ) VALUES (?, ?, ?,?,?,?,?,?)
                """,
                    result,
                )

            # NOTE: commit after each batch
            conn.commit()


def result_exists(db_file, image_id, region_id):
    with closing(sqlite3.connect(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """  
            SELECT COUNT(*) FROM results
            WHERE image_id = ? AND region_id = ?
        """,
            (image_id, region_id),
        )
        count = cursor.fetchone()[0]
    return count > 0


def init_database(db_file):
    REWRITE_DB = os.getenv("REWRITE_DB", None)
    if REWRITE_DB is not None and os.path.exists(db_file):
        os.remove(db_file)
        logger.info(f"Remove existing db file: {db_file}")

    with closing(sqlite3.connect(db_file)) as conn:
        # NOTE: in sqlite3, mind the comma!
        with conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS results (
                    region_cnt INTEGER PRIMARY KEY,
                    image_cnt INTEGER,
                    region_id INTEGER,
                    image_id INTEGER,
                    masks TEXT,
                    scores TEXT,
                    input_box TEXT,
                    gt_caption TEXT)"""
            )


def convert_db_to_json(db_file, json_file):
    with closing(sqlite3.connect(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """  
            SELECT region_cnt, image_cnt, region_id, image_id, masks, scores, input_box, gt_caption
            FROM results
        """
        )
        results = cursor.fetchall()
    results = [
        dict(
            region_cnt=region_cnt,
            image_cnt=image_cnt,
            region_id=region_id,
            image_id=image_id,
            masks=json.loads(masks),
            scores=json.loads(scores),
            input_box=json.loads(input_box),
            gt_caption=json.loads(gt_caption),
        )
        for region_cnt, image_cnt, region_id, image_id, masks, scores, input_box, gt_caption in results
    ]
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
