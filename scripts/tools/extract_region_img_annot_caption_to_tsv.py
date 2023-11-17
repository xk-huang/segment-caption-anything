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
from hydra.utils import instantiate
import numpy as np
import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.core.utils import configure_log
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import pycocotools.mask
from utils.git_utils import TSVWriter
from src.arguments import Arguments
from src.train import prepare_datasets
import torch

logger = logging.getLogger(__name__)


def ensure_correct_bbox(x, y, x2, y2, image_w, image_h):
    x = max(0, x)
    y = max(0, y)
    x2 = min(image_w - 1, x2)
    y2 = min(image_h - 1, y2)
    return x, y, x2, y2


def gen_rows(dataset, img_enc_type="PNG", num_rows=None):
    region_cnt = 0
    sample_cnt = 0
    sent_cnt = 0

    dataset = next(iter(dataset.values()))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=lambda x: x[0]
    )
    tbar = tqdm.tqdm(dataloader)
    for sample_idx, sample in enumerate(tbar):
        if num_rows is not None and sample_idx >= num_rows:
            break
        image = sample["image"]
        image = np.array(image)
        image_id = sample["image_id"]
        image_h, image_w = image.shape[:2]

        for annot_idx, region in enumerate(sample["regions"]):
            mask = region.get("mask", None)
            if mask is None:
                area = -1
            else:
                area = pycocotools.mask.area(mask)

            seg_id = region["region_id"]
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            x2, y2 = x + w, y + h
            try:
                x, y, x2, y2 = ensure_correct_bbox(x, y, x2, y2, image_w, image_h)
                if x >= x2 or y >= y2:
                    raise ValueError(
                        f"Invalid bbox: {x, y, x2, y2} (x, y, x2, y2), at {sample_idx}-{image_id}-{annot_idx}-{seg_id} (sample_idx-image_id-annot_idx-seg_id)"
                    )
                # sub_mask = mask[y:y2, x:x2]
                sub_image = image[y:y2, x:x2].copy()
                # sub_image[~sub_mask] = 255
            except Exception as e:
                logger.error(f"[Crop] Error cropping image: {e}")
                continue

            # prepare tsv row: sub_image_name, img_base64, json_dumped_annot
            sub_image_name = f"{image_id}-{annot_idx}-{seg_id}"
            buffer = io.BytesIO()
            Image.fromarray(sub_image).save(buffer, format=img_enc_type)
            bytes = buffer.getvalue()
            base64_bytes = base64.b64encode(bytes)

            # NOTE: `cap` is a List[str]
            cap = region["phrases"]
            sent_cnt += len(cap)
            region_cnt += 1

            yield sub_image_name, base64_bytes, cap, int(area), [int(image_h), int(image_w)]

        sample_cnt += 1
        tbar.set_description(f"Already processing {sample_cnt} samples and {region_cnt} regions.")

    logger.info(f"Total samples: {sample_cnt}, total regions: {region_cnt}, total sents: {sent_cnt}")


# NOTE(xiaoke): the config_path is `src/conf`
# NOTE(xiaoke): Add additional args:
#   - num_rows
#   - img_enc_type
#   - split
@hydra.main(version_base="1.3", config_path="../../src/conf", config_name="conf")
def main(cfg: Arguments):
    output_dir = cfg.training.output_dir
    if output_dir is None:
        raise ValueError("Output dir is None")
    output_dir = osp.join(output_dir, "region_img_annot_caption")
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    _, eval_dataset = prepare_datasets(cfg)
    eval_data = cfg.eval_data

    if len(eval_data) > 1:
        raise ValueError(f"Only one eval dataset is allowed, got {cfg.eval_data}")

    num_rows = cfg.get("num_rows", None)
    img_enc_type = cfg.get("img_enc_type", "PNG")
    force_overwrite = cfg.get("force_overwrite", False)

    data_path = eval_data[0].path
    data_path = osp.basename(data_path)
    data_name = eval_data[0].name
    data_split = eval_data[0].split

    if data_split is None:
        raise ValueError(
            "Data split is None. "
            f"Please specify the split in: {list(eval_dataset.keys())}. "
            f"e.g., +split={list(eval_dataset.keys())[0]}"
        )

    extract_from_one_split(
        output_dir, eval_dataset, num_rows, img_enc_type, data_split, data_path, data_name, force_overwrite
    )


def extract_from_one_split(
    output_dir, dataset, num_rows, img_enc_type, data_split, data_path, data_name, force_overwrite
):
    output_name = f"{data_path}-{data_name}-{data_split}"
    output_img_path = f"{output_dir}/{output_name}.region_img.tsv"
    output_cap_path = f"{output_dir}/{output_name}.region_cap.tsv"
    output_annot_path = f"{output_dir}/{output_name}.region_annot.tsv"

    logger.info(f"Output dir: {output_dir}, Data split: {data_split}")
    logger.info(f"\tOutput img path: {output_img_path}")
    logger.info(f"\tOutput cap path: {output_cap_path}")
    logger.info(f"\tOutput annot path: {output_annot_path}")

    if osp.exists(output_img_path) or osp.exists(output_cap_path) or osp.exists(output_annot_path):
        if force_overwrite is True:
            logger.info(
                f"Force overwrite output tsv files:\n\t{output_img_path},\n\t{output_cap_path},\n\t{output_annot_path}"
            )
        else:
            raise ValueError(
                f"Output tsv files exist. Skip:\n\t{output_img_path},\n\t{output_cap_path},\n\t{output_annot_path}"
            )

    with TSVWriter(output_img_path) as img_tsv_writer, TSVWriter(output_cap_path) as cap_tsv_writer, TSVWriter(
        output_annot_path
    ) as annot_tsv_writer:
        for row in gen_rows(dataset, img_enc_type=img_enc_type, num_rows=num_rows):
            sub_image_name, base64_bytes, cap, area, image_size = row
            cap = json.dumps([dict(caption=cap)])
            img_tsv_writer.write((sub_image_name, base64_bytes))
            cap_tsv_writer.write((sub_image_name, cap))
            annot = json.dumps(dict(area=area, image_size=image_size))
            annot_tsv_writer.write((sub_image_name, annot))


if __name__ == "__main__":
    main()
