import sys
import os

sys.path.append(".")
import os

import torch
from src.arguments import global_setup, SAMCaptionerModelArguments, SCAModelBaseArguments
from src.models.sam_captioner import SAMCaptionerProcessor
from src.models.sca import ScaProcessor

import numpy as np
import pandas as pd
from src.train import prepare_datasets, prepare_data_transform
import pycocotools.mask
from PIL import Image

from hydra import initialize, compose
import json
import tqdm
import hashlib
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
import json
from PIL import Image
import io
import base64
import pycocotools.mask
import sqlite3

os.getcwd()

from flask import Flask, render_template, request, send_file


DATASET = "vg-densecap-local"
with initialize(version_base="1.3", config_path="../../src/conf"):
    args = compose(
        config_name="conf",
        overrides=[
            f"train_data=[{DATASET}]",
            f"eval_data=[{DATASET}]",
            "+model=base_sam_captioner",
            "training.output_dir=tmp/visualization"
            # "training.do_train=True",
            # "training.do_eval=True",
        ],
    )


args, training_args, model_args = global_setup(args)
os.makedirs(training_args.output_dir, exist_ok=True)


# Initialize our dataset and prepare it
with initialize(version_base="1.3", config_path="../../src/conf"):
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

train_dataset, eval_dataset = prepare_data_transform(training_args, model_args, train_dataset, eval_dataset, processor)


# [NOTE] Used to restore the image tensor after transformed
# Use global to avoid passing too many arguments
global image_mean, image_std
image_mean, image_std = (
    processor.sam_processor.image_processor.image_mean,
    processor.sam_processor.image_processor.image_std,
)


REWRITE_MAPPING = False
image_id_to_dataset_id_mapping_file = os.path.join(training_args.output_dir, "image_id_to_dataset_id_mapping.json")


def find_json_file_with_md5(json_file):
    json_file_name, json_file_ext = os.path.splitext(json_file)
    json_file_blob = f"{json_file_name}-*{json_file_ext}"
    return glob.glob(json_file_blob)


def get_md5_from_json(json_file):
    with open(json_file, "r") as f:
        content = f.read()
    return hashlib.md5(content.encode()).hexdigest()


def get_md5_from_pyobj(pyobj):
    bytes_data = pyobj.encode()
    readable_hash = hashlib.md5(bytes_data).hexdigest()
    return readable_hash


def save_dict_to_json_with_md5(json_file, dict_data):
    # Convert to json and bytes
    json_data = json.dumps(dict_data)
    json_data_md5 = get_md5_from_pyobj(json_data)
    json_file_name, json_file_ext = os.path.splitext(json_file)
    json_file_with_md5 = f"{json_file_name}-{json_data_md5}{json_file_ext}"
    with open(json_file_with_md5, "w") as f:
        f.write(json_data)
    return json_file_with_md5


# Initialize our dataset and prepare it
with initialize(version_base="1.3", config_path="../../src/conf"):
    args_no_image = compose(
        config_name="conf",
        overrides=[
            f"train_data=[{DATASET}]",
            f"eval_data=[{DATASET}]",
            "+model=base_sam_captioner",
            "training.output_dir=tmp/visualization"
            # "training.do_train=True",
            # "training.do_eval=True",
        ],
    )
    args_no_image.train_data_overrides = ["data.with_image=False"]
    args_no_image.eval_data_overrides = ["data.with_image=False"]
    train_dataset_no_image, eval_dataset_no_image = prepare_datasets(args_no_image)

json_file_with_md5_ls = find_json_file_with_md5(image_id_to_dataset_id_mapping_file)
if len(json_file_with_md5_ls) > 1:
    raise ValueError(f"find more than one json file with md5, {json_file_with_md5_ls}")
if REWRITE_MAPPING is False and len(json_file_with_md5_ls) == 1:
    image_id_to_dataset_id_mapping_file = json_file_with_md5_ls[0]
    md5_in_name = os.path.splitext(image_id_to_dataset_id_mapping_file)[0].split("-")[-1]
    assert md5_in_name == get_md5_from_json(
        image_id_to_dataset_id_mapping_file
    ), f"md5 not match for {image_id_to_dataset_id_mapping_file}"

    with open(image_id_to_dataset_id_mapping_file, "r") as f:
        image_id_to_dataset_id_mapping = json.load(f)
    print(f"Load mapping from {image_id_to_dataset_id_mapping_file}")
else:
    image_id_to_dataset_id_mapping = {
        "train": dict(),
        **{k: dict() for k in eval_dataset_no_image.keys()},
    }
    for sample_cnt, sample in enumerate(tqdm.tqdm(train_dataset_no_image)):
        image_id_to_dataset_id_mapping["train"][sample["image_id"]] = sample_cnt
    for eval_dataset_name, eval_dataset_ in eval_dataset_no_image.items():
        for sample_cnt, sample in enumerate(tqdm.tqdm(eval_dataset_)):
            image_id_to_dataset_id_mapping[eval_dataset_name][sample["image_id"]] = sample_cnt
    image_id_to_dataset_id_mapping_file = save_dict_to_json_with_md5(
        image_id_to_dataset_id_mapping_file, image_id_to_dataset_id_mapping
    )
    print(f"save mapping to {image_id_to_dataset_id_mapping_file}")


def hex_to_rgb(hex_color):
    return tuple([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)])


# hex_colors = ["#EB1E2CFF", "#FD6F30FF", "#F9A729FF", "#F9D23CFF", "#5FBB68FF", "#64CDCCFF", "#91DCEAFF", "#A4A4D5FF", "#BBC9E5FF"]
hex_colors = ["#F1F5F9FF", "#FFE7E8FF", "#EEFDEEFF", "#FFF9C8FF", "#FED5B2FF", "#F9E8D2FF", "#E2E3F9FF", "#DCFEFFFF"]

rgb_colors = [
    hex_to_rgb(color[:-2]) for color in hex_colors
]  # '[:-2]' is to remove the 'FF' at the end of each color code, which represents the alpha channel in ARGB format
harmonious_colors = rgb_colors


def draw_bbox(pil_image, bbox, color=(255, 0, 0), thickness=2):
    cv_image = np.array(pil_image)
    cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return Image.fromarray(cv_image)


def draw_mask(pil_image, mask_array, color=(255, 0, 0), alpha=0.1):
    cv_image = np.array(pil_image)
    cv_image[mask_array == 1] = cv_image[mask_array == 1] * (1 - alpha) + np.array(color) * alpha
    return Image.fromarray(cv_image)


def draw_mask_boundary(pil_image, mask_array, color=(255, 0, 0), thickness=1):
    cv_image = np.array(pil_image)
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cv_image, contours, -1, color, thickness)
    return Image.fromarray(cv_image)


def resize_image(image, height=None, width=None):
    """
    Resizes an image given the desired height and/or width.
    If only one of height or width is provided, the other dimension is scaled proportionally.
    If both height and width are provided, the image is resized to the exact dimensions.
    """
    if height is None and width is None:
        return image

    original_width, original_height = image.size

    if height is not None and width is not None:
        new_size = (width, height)
    elif height is not None:
        new_size = (int(original_width * height / original_height), height)
    else:
        new_size = (width, int(original_height * width / original_width))

    return image.resize(new_size)


def draw_captions(
    pil_image,
    captions,
    font_path="tmp/Arial.ttf",
    font_size=20,
    font_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    margin_size=10,
    captions_color=None,
):
    font = ImageFont.truetype(font_path, font_size)
    # Calculate the total height of the padding for the captions
    text_height = font.getbbox("My")[-1]
    total_height = 0
    for caption in captions:
        total_height += text_height + margin_size

    # Create a new image with padding at the bottom for the captions
    new_image = Image.new("RGB", (pil_image.width, pil_image.height + total_height), bg_color)
    new_image.paste(pil_image, (0, 0))

    draw = ImageDraw.Draw(new_image)
    # Draw each caption
    y_position = pil_image.height
    for caption_id, caption in enumerate(captions):
        _, _, text_width, _ = font.getbbox(caption)
        if captions_color is not None:
            text_bbox = (0, y_position, text_width, y_position + text_height)
            fill_color = captions_color[caption_id]
            draw.rectangle(text_bbox, fill=fill_color, width=0)
        draw.text((0, y_position), caption, fill=font_color, font=font)
        y_position += text_height + margin_size

    return new_image


def plot_bbox_and_captions(
    pil_image,
    bbox=None,
    captions=None,
    mask=None,
    font_path="tmp/Arial.ttf",
    font_size=20,
    font_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    margin_size=0,
    captions_color=None,
    plot_mask=False,
    plot_mask_boundary=False,
    plot_bbox=False,
):
    if bbox is not None and plot_bbox is True:
        pil_image = draw_bbox(pil_image, bbox)
    if mask is not None and plot_mask_boundary is True:
        pil_image = draw_mask_boundary(pil_image, mask)
    if mask is not None and plot_mask is True:
        pil_image = draw_mask(pil_image, mask)

    pil_image = resize_image(pil_image, height=512)

    if captions is not None:
        pil_image = draw_captions(
            pil_image, captions, font_path, font_size, font_color, bg_color, margin_size, captions_color=captions_color
        )
    return pil_image


# Load the infer json
infer_json_path_dict = {
    "sam_cap-git_large": "amlt/111523.exp.sam_captioner/infer_sam_captioner_region_chunkify/microsoft/git-large/infer-post_processed/infer-visual_genome-densecap-local-densecap-test.json.post.json",
    "sam_cap-blip_large": "amlt/111523.exp.sam_captioner/infer-sam_captioner-region_chunkify-eval_suite/Salesforce/blip-image-captioning-large/vg-densecap-region_descriptions/infer-post_processed/infer-visual_genome-region_descriptions_v1.2.0-test.json.post.json",
    "sam_cap-blip2_opt_2_7b": "amlt/111523.exp.sam_captioner/infer-sam_captioner-region_chunkify-eval_suite/Salesforce/blip2-opt-2.7b/infer-post_processed/infer-visual_genome-densecap-local-densecap-test.json.post.json",
    "grit": "amlt/111523.exp.grit/infer-promptable-grit/infer-post_processed/infer-visual_genome-densecap-local-densecap-test.json.post.json",
    "vg-gpt2l-bs_32-lsj": "/home/t-yutonglin/xiaoke/segment-caption-anything-v2/amlt/111423.exp-only_vg-finetune_vg/111323.infer-train-sca-ablat-lsj-scale_lr-110423.4x8_fin-16x4_unfin.pre/best-gpt2-large-lsj-1xlr.110423.octo-4x8-v100-16g-no_pre/vg-densecap-region_descriptions/infer-post_processed/infer-visual_genome-region_descriptions_v1.2.0-test.json",
    "vg-ollm3bv2-bs_32-lsj": "/home/t-yutonglin/xiaoke/segment-caption-anything-v2/amlt/110723.exp.ablat-lsj-scale_lr-running-2/infer-train-sca-ablat-lsj-scale_lr-110423-110723.running-2/best-fp16-ollm3bv2-large-lsj-1xlr.110423.octo-4x8-v100-16g-no_pre/vg-densecap-region_descriptions/infer-post_processed/infer-visual_genome-region_descriptions_v1.2.0-test.json",
    "o365_vg-gpt2l-bs_64-lsj": "/home/t-yutonglin/xiaoke/segment-caption-anything-v2/amlt/111423.exp-only_vg-finetune_vg/111323.infer-train-sca.finetune_lsj_scale_lr-o365_1e_4_1xlr_lsj.111023.4x8_fin-16x4_unfin.pre/best-111223.rr1-4x8-v100-32g-pre.fintune-gpt2_large-lr_1e_4-1xlr-lsj-bs_2-o365_1e_4_no_lsj_bs_64/vg-densecap-region_descriptions/infer-post_processed/infer-visual_genome-region_descriptions_v1.2.0-test.json",
}
mask_db_file = "tmp/sam_mask_db/visual_genome-densecap-local-densecap-test/results.db"

for job_name, json_path in tqdm.tqdm(infer_json_path_dict.items()):
    print(f"[infer json] job_name: {job_name}")
    print(f"[infer json] is exists: {os.path.exists(json_path)}")
    assert os.path.exists(json_path), f"{json_path} not exists"


class MultiInferJson(torch.utils.data.Dataset):
    def __init__(self, infer_json_path_dict, mask_db_file=None):
        self.infer_json_path_dict = infer_json_path_dict
        self.infer_json_dict = dict()
        for job_name, json_path in self.infer_json_path_dict.items():
            with open(json_path, "r") as f:
                self.infer_json_dict[job_name] = json.load(f)

        # check their length
        first_key = next(iter(self.infer_json_dict))
        for job_name, infer_json in self.infer_json_dict.items():
            assert len(infer_json) == len(self.infer_json_dict[first_key]), f"length not match for {job_name}"
        self._len = len(self.infer_json_dict[first_key])

        self.mask_db_file = mask_db_file
        if mask_db_file is not None:
            conn = sqlite3.connect(self.mask_db_file)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM results
            """
            )
            count = cursor.fetchone()[0]
            if count != self._len:
                print(f"mask_db_file length not match, {count} != {self._len}")
            conn.close()

    def __len__(self):
        return self._len

    def make_sure_db_open(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.mask_db_file)

    def __getitem__(self, idx):
        masks, scores = None, None
        if self.mask_db_file is not None:
            conn = sqlite3.connect(self.mask_db_file)
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT * FROM results WHERE region_cnt = {idx}
            """,
            )
            results = cursor.fetchone()
            conn.close()

            if results is not None:
                rle_masks = results[4]
                scores = results[5]
                rle_masks = json.loads(rle_masks)
                scores = json.loads(scores)
                masks = pycocotools.mask.decode(rle_masks)
            else:
                breakpoint()
                # import IPython; IPython.embed()
        mask_dict = {"masks": masks, "scores": scores}

        ret_dict = {}
        for job_name, infer_json in self.infer_json_dict.items():
            ret_dict[job_name] = infer_json[idx]
            ret_dict[job_name].update(mask_dict)
        return ret_dict


infer_json_dataset = MultiInferJson(infer_json_path_dict, mask_db_file=mask_db_file)


def check_region_id_image_id(infer_json_dataset):
    dataloader = torch.utils.data.DataLoader(
        infer_json_dataset, batch_size=1, shuffle=False, num_workers=10, collate_fn=lambda x: x[0]
    )
    for sample in tqdm.tqdm(dataloader, desc="Check region_id and image_id"):
        first_key = next(iter(sample))
        image_id = sample[first_key]["metadata"]["metadata_image_id"]
        region_id = sample[first_key]["metadata"]["metadata_region_id"]
        for job_name, region_pred in sample.items():
            assert image_id == region_pred["metadata"]["metadata_image_id"], f"image_id not match for {job_name}"
            assert region_id == region_pred["metadata"]["metadata_region_id"], f"region_id not match for {job_name}"


if os.getenv("CHECK_REGION_ID_IMAGE_ID", None) is not None:
    print("CHECK_REGION_ID_IMAGE_ID is set, check region_id and image_id")
    check_region_id_image_id(infer_json_dataset)


def plot_one_region(
    infer_json_dataset, region_cnt, plot_mask=False, plot_mask_boundary=False, plot_bbox=False, selected_job_names=None
):
    samples = infer_json_dataset[region_cnt]
    first_key = next(iter(samples))
    EVAL_DATASET_SPLIT = "visual_genome-densecap-local-densecap-test"

    first_sample = samples[first_key]

    image_id = first_sample["metadata"]["metadata_image_id"]
    region_id = first_sample["metadata"]["metadata_region_id"]
    input_boxes = first_sample["metadata"]["metadata_input_boxes"]

    sample_cnt = image_id_to_dataset_id_mapping[EVAL_DATASET_SPLIT][str(image_id)]
    sample = eval_dataset[EVAL_DATASET_SPLIT][sample_cnt]
    image = sample["image"]

    references = first_sample["references"]

    candidates = []

    candidate_captions_colors = []
    num_colors = len(harmonious_colors)
    for i, (job_name, region_pred) in enumerate(samples.items()):
        if selected_job_names is not None and job_name not in selected_job_names:
            continue
        candidates.extend(region_pred["candidates"])
        candidate_captions_colors.append(harmonious_colors[i % num_colors])

    font_path = "tmp/Arial.ttf"

    captions = candidates + references
    captions_color = candidate_captions_colors + [(255, 255, 255)]

    masks = first_sample["masks"]
    scores = first_sample["scores"]
    if masks is not None and scores is not None:
        # NOTE: masks from pycoco is in (h, w, n) format
        max_mask_score = np.argmax(scores)
        mask = masks[..., max_mask_score]
    else:
        mask = None
    pil_img_with_bbox_and_captions = plot_bbox_and_captions(
        image,
        bbox=input_boxes,
        mask=mask,
        captions=captions,
        captions_color=captions_color,
        font_path=font_path,
        margin_size=5,
        plot_mask=plot_mask,
        plot_mask_boundary=plot_mask_boundary,
        plot_bbox=plot_bbox,
    )
    return pil_img_with_bbox_and_captions, f"{region_cnt}-{sample_cnt}-{region_id}-{image_id}"


# region_cnt = 0
# pil_img_with_bbox_and_captions, pil_img_with_bbox_and_captions_path = plot_one_region(infer_json_dataset, region_cnt)
# pil_img_with_bbox_and_captions.save(os.path.join(training_args.output_dir, pil_img_with_bbox_and_captions_path))
# pil_img_with_bbox_and_captions


def _add_prefix_suffix_to_path(path: str, prefix: str, suffix: str) -> str:
    base_dir, filename = os.path.split(path)
    return os.path.join(base_dir, prefix + filename + suffix)


score_json_path_dict = {}
# CIDEr-D-scores.infer-visual_genome-region_descriptions_v1.2.0-test.json.json
SCORE_PREFIX = "CIDEr-D-scores."
SCORE_SUFFIX = ".json"

for k, v in infer_json_path_dict.items():
    score_json_path_dict[k] = _add_prefix_suffix_to_path(v, SCORE_PREFIX, SCORE_SUFFIX)
for job_name, score_json_path in tqdm.tqdm(score_json_path_dict.items()):
    print(f"[score json] job_name: {job_name}")
    print(f"[score json] is exists: {os.path.exists(score_json_path)}")
    if not os.path.exists(score_json_path):
        print(f"{score_json_path} not exists")


score_json_dict = {}
for k, v in score_json_path_dict.items():
    try:
        with open(v, "r") as f:
            score_json_dict[k] = json.load(f)
    except FileNotFoundError:
        print(f"{v} not found")


def build_score_df(score_json_dict):
    return pd.DataFrame.from_dict({k: v for k, v in score_json_dict.items()})


score_df = build_score_df(score_json_dict)
score_df


app = Flask(__name__)
app.secret_key = "your secret key"

# NOTE: global variables to save the state of the page
region_cnt_ls = None
job_names = None


@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    global region_cnt_ls
    start_image_id = int(request.form.get("start_image_id", 0))
    num_images = int(request.form.get("num_images", 10))
    images_per_row = int(request.form.get("images_per_row", 5))

    plot_mask = "plot_mask" in request.form
    plot_mask_boundary = "plot_mask_boundary" in request.form
    plot_bbox = "plot_bbox" in request.form

    mode = request.form.get("mode", "random_generate")

    selected_job_names = request.form.getlist("job_names")

    prefix = request.form.get("prefix", "")

    region_cnt_ls = request.form.get("region_cnt_ls", None)

    if start_image_id < 0 or start_image_id >= len(infer_json_dataset):
        start_image_id = 0  # Reset to default if out of range
    if num_images < 1 or start_image_id + num_images > len(infer_json_dataset):
        num_images = 10  # Reset to default if out of range
        num_images = min(num_images, len(infer_json_dataset) - start_image_id)
    if images_per_row < 1:
        images_per_row = 5  # Reset to default if less than 1

    # Depending on the chosen mode, we generate the image list differently
    if mode == "random_generate":
        region_cnt_ls = np.random.randint(0, len(infer_json_dataset), num_images)
    elif mode == "random_start_id":
        start_image_id = np.random.randint(0, len(infer_json_dataset))
        region_cnt_ls = list(range(start_image_id, start_image_id + num_images))
    elif mode == "chosen_id":
        region_cnt_ls = list(range(start_image_id, start_image_id + num_images))
    elif mode == "given_ids":
        region_cnt_ls = [int(i) for i in region_cnt_ls.split(",")]

    return render_page(
        start_image_id,
        num_images,
        images_per_row,
        region_cnt_ls,
        mode,
        plot_mask,
        plot_mask_boundary,
        plot_bbox,
        selected_job_names,
        prefix,
    )


@app.route("/re_visualize", methods=["GET", "POST"])
def re_visualize():
    global region_cnt_ls
    start_image_id = int(request.form.get("start_image_id", 0))
    num_images = int(request.form.get("num_images", 10))
    images_per_row = int(request.form.get("images_per_row", 5))

    plot_mask = "plot_mask" in request.form
    plot_mask_boundary = "plot_mask_boundary" in request.form
    plot_bbox = "plot_bbox" in request.form

    mode = request.form.get("mode", "random_generate")

    selected_job_names = request.form.getlist("job_names")

    prefix = request.form.get("prefix", "")

    return render_page(
        start_image_id,
        num_images,
        images_per_row,
        region_cnt_ls,
        mode,
        plot_mask,
        plot_mask_boundary,
        plot_bbox,
        selected_job_names,
        prefix,
    )


@app.route("/", methods=["GET", "POST"])
def render_home():
    global job_names
    start_image_id = int(request.form.get("start_image_id", 0))
    num_images = int(request.form.get("num_images", 10))
    images_per_row = int(request.form.get("images_per_row", 5))
    plot_bbox = True
    plot_mask_boundary = True
    plot_mask = False

    samples = infer_json_dataset[0]
    job_names = [job_name for job_name in samples.keys()]

    mode = "random_generate"
    return render_template(
        "home.html",
        images_per_row=images_per_row,
        start_image_id=start_image_id,
        num_images=num_images,
        mode=mode,
        plot_mask=plot_mask,
        plot_mask_boundary=plot_mask_boundary,
        plot_bbox=plot_bbox,
        job_names=job_names,
        selected_job_names=job_names,
    )


def render_page(
    start_image_id,
    num_images,
    images_per_row,
    regions_cnt_ls,
    mode,
    plot_mask,
    plot_mask_boundary,
    plot_bbox,
    selected_job_names,
    prefix,
):
    images = []

    print("regions_cnt_ls", regions_cnt_ls)
    pbar = tqdm.tqdm(regions_cnt_ls)
    for i in pbar:
        pil_img, pil_img_note = plot_one_region(
            infer_json_dataset,
            i,
            plot_mask=plot_mask,
            plot_mask_boundary=plot_mask_boundary,
            plot_bbox=plot_bbox,
            selected_job_names=selected_job_names,
        )  # Assuming dataset[i] returns a tuple of (image, caption)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        pbar.write(
            f"image size: {pil_img.size}, plot_bbox_and_captions: plot_mask={plot_mask}, plot_mask_boundary={plot_mask_boundary}, plot_bbox={plot_bbox}"
        )
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        images.append((img_base64, pil_img_note))

    samples = infer_json_dataset[regions_cnt_ls[0]]
    model_name = [job_name for job_name in samples.keys()]
    num_colors = len(harmonious_colors)
    selected_colors = [harmonious_colors[i % num_colors] for i in range(len(model_name))]
    model_color_fig = draw_captions(Image.new("RGB", (256, 0)), model_name, captions_color=selected_colors)
    buf = io.BytesIO()
    model_color_fig.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    images.append((img_base64, "model_name.png"))

    print(f"len of images: {len(images)}")

    print("html is ready!")
    return render_template(
        "home.html",
        images=images,
        images_per_row=images_per_row,
        start_image_id=start_image_id,
        num_images=num_images,
        mode=mode,
        plot_mask=plot_mask,
        plot_mask_boundary=plot_mask_boundary,
        plot_bbox=plot_bbox,
        job_names=job_names,
        selected_job_names=selected_job_names,
        prefix=prefix,
        region_cnt_ls=",".join(map(str, regions_cnt_ls)),
    )


if __name__ == "__main__":
    debug = os.getenv("DEBUG", None) is not None
    print(f"debug: {debug}")
    app.run(debug=debug)
