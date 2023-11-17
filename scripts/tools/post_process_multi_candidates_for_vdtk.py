import json
import os
import os.path as osp
import click
import numpy as np
import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@click.command()
@click.option("--input_file", "-i", type=str, help="input file")
@click.option("--output_file", "-o", type=str, help="output file", default=None)
def main(input_file, output_file):
    if osp.splitext(input_file)[1] != ".json":
        raise ValueError("input file must be json file")

    if output_file is None:
        output_file = osp.splitext(input_file)[0] + ".post_process.json"
    output_dir = osp.dirname(output_file)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Input file path: {input_file}")
    logger.info(f"Output file path: {output_file}")

    with open(input_file, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("input file must be a list of dict")
    if not all(isinstance(d, dict) for d in data):
        raise ValueError("input file must be a list of dict")
    # Specialized check for the output of VDTK and the region caption task.
    if not all(check_keys_in_dict(d) for d in data):
        logger.warning(
            "[WARNING] input file must be a list of dict with keys: logits, candidates. "
            f"We directly copy the file ({output_file}) due to the error."
        )
    else:
        for d in tqdm.tqdm(data):
            process_dict(d)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def check_keys_in_dict(d: dict) -> bool:
    # NOTE(xiaoke): This function is specially designed for the output of VDTK and the region caption task.
    is_ok = d.get("logits") and d.get("logits").get("iou_scores")
    is_ok = is_ok and len(d.get("logits").get("iou_scores")) == len(d.get("candidates"))
    return is_ok


def process_dict(d: dict) -> None:
    try:
        # NOTE(xiaoke): This function is specially designed for the output of VDTK and the region caption task.
        iou_scores = d.get("logits").get("iou_scores")
        candidates = d.get("candidates")
        max_iou_idx = np.argmax(iou_scores)
        d["candidates"] = candidates[max_iou_idx : max_iou_idx + 1]
    except Exception as e:
        logger.warning(f"[WARNING] {e}")
        if d.get("candidates", None) is not None and len(d["candidates"]) > 1:
            logger.warning(
                f"[WARNING] multiple candidates are found, but we only keep the first one as we miss the `logits.iou_scores` key."
            )
            d["candidates"] = d["candidates"][:1]


if __name__ == "__main__":
    main()
