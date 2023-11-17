"""
example from gradio: https://www.gradio.app/guides/running-background-tasks
"""
import gradio as gr
from dataclasses import dataclass
import sqlite3
import random
from utils.git_utils.tsv_io import TSVFile
import json
from PIL import Image
import io
import base64
import pandas as pd
import datetime
import os

# Connect to the SQLite database or create a new one if it doesn't exist
db_path = "tmp/annotations.db"
# Load the images and captions from the TSV files
images_tsv_path = "/home/v-xiaokhuang/sa1b_cropper/sa1b_data/sa1b-sub_image_w_bg-tsv/sa_000000.tar.tsv"
captions_tsv_path = "/home/v-xiaokhuang/sa1b_cropper/sa1b_data/sa1b-git_caption-tsv/model_iter_0007189.pt.TaxXiaokeV2.test0.crop384.crpPct1.fp16.gen.lenP0.6.beam4.predict.tsv"
annots_tsv_path = "/home/v-xiaokhuang/sa1b_cropper/sa1b_data/sa1b-annot-tsv/sa_000000.tar.tsv"
clip_scores_tsv_path = "/home/v-xiaokhuang/segment-caption-anything/out/sa1b-cap-0.clip-truncation.tsv"

images = TSVFile(images_tsv_path)
captions = TSVFile(captions_tsv_path)
if len(images) != len(captions):
    raise ValueError("Number of images and captions do not match.")
num_rows = len(images)


def b64_to_bin(base64_bin_str: str) -> bytes:
    """
    Decodes a base64 binary string to a binary string.
    """
    return io.BytesIO(base64.b64decode(base64_bin_str))


# Function to display a random image and its caption
def display_image_and_caption(gr_vars):
    images = gr_vars.images_tsv
    captions = gr_vars.captions_tsv
    annots = gr_vars.annots_tsv
    clip_scores = gr_vars.clip_scores_tsv
    num_rows = gr_vars.num_rows

    random_index = random.randint(0, num_rows - 1)
    identifier, base64_bytes = images[random_index]
    _identifier, caption_json_string = captions[random_index]
    if identifier != _identifier:
        raise ValueError(f"Image and caption identifiers do not match, {identifier} != {_identifier}")
    _identifier, annot_json_string = annots[random_index]
    if identifier != _identifier:
        raise ValueError(f"Image and annotation identifiers do not match, {identifier} != {_identifier}")
    _identifier, clip_scores_json_string = clip_scores[random_index]
    if identifier != _identifier:
        raise ValueError(f"Image and clip scores identifiers do not match, {identifier} != {_identifier}")

    image = Image.open(b64_to_bin(base64_bytes))

    caption_json = json.loads(caption_json_string)
    caption = caption_json[0]["caption"]  # TODO(xiaoke): now only use the first caption
    conf = caption_json[0].get("conf", -1)

    annot_json = json.loads(annot_json_string)
    area = annot_json["area"]
    image_size = annot_json["image_size"]
    image_area = image_size[0] * image_size[1]

    clip_scores_json = json.loads(clip_scores_json_string)
    clip_score = clip_scores_json[0]["clip_score"]  # TODO(xiaoke): now only use the first clip score
    # TODO(xiaoke): add aspect ratio

    image_id, region_cnt, region_id = list(map(int, identifier.split("-")))
    return (
        image,
        caption,
        dict(
            caption=caption,
            conf=conf,
            area=area,
            image_area=image_area,
            clip_score=clip_score,
            image_id=image_id,
            region_cnt=region_cnt,
            region_id=region_id,
        ),
    )


def get_latest_reviews(db: sqlite3.Connection):
    reviews = db.execute("SELECT * FROM annotations ORDER BY created_at DESC LIMIT 10").fetchall()
    num_reviews = db.execute("Select COUNT(image_id) from annotations").fetchone()[0]
    # caption, conf, area, image_area, clip_score, image_id, region_cnt, region_id, is_acceptable, created_at
    reviews = pd.DataFrame(
        reviews,
        columns=[
            "caption",
            "conf",
            "area",
            "image_area",
            "clip_score",
            "image_id",
            "region_cnt",
            "region_id",
            "is_acceptable",
            "created_at",
        ],
    )
    return reviews, num_reviews


def load_tables(gt_vars):
    db_path = gt_vars.db_path
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    reviews, num_reviews = get_latest_reviews(conn)
    conn.close()
    return reviews, num_reviews


# Function to handle user annotation
def _annotate(gr_vars, sample_dict, is_acceptable):
    db_path = gr_vars.db_path
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ks = list(sample_dict.keys()) + ["is_acceptable"]
    vs = list(sample_dict.values()) + [int(is_acceptable)]
    cur.execute(
        f"INSERT INTO annotations ({', '.join(ks)}) VALUES ({', '.join(['?'] * len(ks))})",
        vs,
    )
    conn.commit()
    reviews, num_reviews = get_latest_reviews(conn)
    conn.close()
    return f"[{datetime.datetime.now()}] Annotation saved.", reviews, num_reviews


def annotate_yes(gr_vars, sample_dict):
    return _annotate(gr_vars, sample_dict, True)


def annotate_no(gr_vars, sample_dict):
    return _annotate(gr_vars, sample_dict, False)


@dataclass
class GradioVariables:
    db_path: str
    images_tsv: TSVFile = None
    captions_tsv: TSVFile = None
    annots_tsv: TSVFile = None
    clip_scores_tsv: TSVFile = None
    num_rows: int = None


def files_setup(
    gr_vars: GradioVariables,
    db_path,
    image_tsv_path,
    caption_tsv_path,
    annot_tsv_path,
    clip_scores_tsv_path,
):
    # Connect to the SQLite database or create a new one if it doesn't exist
    gr_vars.db_path = db_path
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create the annotations table if it doesn't exist
    # caption, conf, area, image_area, clip_score, image_id, region_cnt, region_id, is_acceptable, created_at
    print("Creating annotations table...")
    cur.execute(
        """
CREATE TABLE IF NOT EXISTS annotations (
    caption TEXT,
    conf REAL,
    area REAL,
    image_area REAL,
    clip_score REAL,
    image_id INTEGER,
    region_cnt INTEGER,
    region_id INTEGER,
    is_acceptable INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)  
"""
    )

    # Load the images and captions from the TSV files
    for path in [image_tsv_path, caption_tsv_path, annot_tsv_path, clip_scores_tsv_path]:
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist.")
    gr_vars.images_tsv = TSVFile(image_tsv_path)
    gr_vars.captions_tsv = TSVFile(caption_tsv_path)
    gr_vars.annots_tsv = TSVFile(annot_tsv_path)
    gr_vars.clip_scores_tsv = TSVFile(clip_scores_tsv_path)
    if len(gr_vars.images_tsv) != len(gr_vars.captions_tsv):
        raise ValueError("Number of images and captions do not match.")
    if len(gr_vars.images_tsv) != len(gr_vars.annots_tsv):
        raise ValueError("Number of images and annotations do not match.")
    if len(gr_vars.images_tsv) != len(gr_vars.clip_scores_tsv):
        raise ValueError("Number of images and CLIP scores do not match.")
    gr_vars.num_rows = len(gr_vars.images_tsv)


# Gradio UI components
with gr.Blocks() as iface:
    gr_vars = GradioVariables(db_path)
    gr_vars = gr.Variable(gr_vars)

    with gr.Accordion(label="Files Setup", open=False) as file_setup:
        db_path = gr.Textbox(label="Database Path", value=db_path)
        image_tsv_path = gr.Textbox(label="Image TSV Path", value=images_tsv_path)
        caption_tsv_path = gr.Textbox(label="Caption TSV Path", value=captions_tsv_path)
        annot_tsv_path = gr.Textbox(label="Annotation TSV Path", value=annots_tsv_path)
        clip_scores_tsv_path = gr.Textbox(label="CLIP Scores TSV Path", value=clip_scores_tsv_path)
        files_setup_button = gr.Button(value="Reload Files")
    files_setup_button.click(
        files_setup, inputs=[gr_vars, db_path, image_tsv_path, caption_tsv_path, annot_tsv_path, clip_scores_tsv_path]
    )
    iface.load(
        files_setup, inputs=[gr_vars, db_path, image_tsv_path, caption_tsv_path, annot_tsv_path, clip_scores_tsv_path]
    )

    intro = gr.Markdown(
        value="""## Dataset Annotator
This tool is used to annotate the dataset.
1. If the region and caption are related and acceptable, click "Yes".
    1. The "in the whitebackground" region is ok.
2. If the region and caption are not related or neither of them are not acceptable, click "No".

"""
    )

    with gr.Row() as image_row:
        image = gr.Image(height=500)
        with gr.Column() as caption_column:
            caption = gr.Textbox(label="Caption")
    sample_dict = gr.Variable({})

    with gr.Row() as choice_row:
        yes_button = gr.Button(value="Yes")
        no_button = gr.Button(value="No")
        # TODO(xiaoke) add a button to skip the current image
    db_output = gr.Textbox(label="Log")

    with gr.Accordion(label="Latest annotations from the database", open=True) as accordion:
        count = gr.Number(label="Total number of rows")
        data = gr.Dataframe(label="Most recently created 10 rows")

    iface_handle = iface.load(display_image_and_caption, inputs=[gr_vars], outputs=[image, caption, sample_dict])
    iface_handle.then(load_tables, inputs=[gr_vars], outputs=[data, count])

    yes_button_handle = yes_button.click(annotate_yes, inputs=[gr_vars, sample_dict], outputs=[db_output, data, count])
    yes_button_handle = yes_button_handle.then(
        display_image_and_caption, inputs=[gr_vars], outputs=[image, caption, sample_dict]
    )

    no_button_handle = no_button.click(annotate_no, inputs=[gr_vars, sample_dict], outputs=[db_output, data, count])
    no_button_handle.then(display_image_and_caption, inputs=[gr_vars], outputs=[image, caption, sample_dict])

iface.launch()
