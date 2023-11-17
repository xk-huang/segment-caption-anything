import click
from utils.git_utils.tsv_io import TSVFile
import json
import tqdm
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_mapping_for_json(json_data):
    mapping = defaultdict(dict)
    for data in tqdm.tqdm(json_data):
        if data.get("metadata", None) is None:
            raise ValueError(
                "Metadata is not found in JSON data, we need it to build image-region idx to image mapping"
            )
        metadata = data["metadata"]
        if metadata.get("metadata_image_id", None) is None:
            raise ValueError(
                "Image ID is not found in JSON data, we need it to build image-region idx to image mapping"
            )
        if metadata.get("metadata_region_id", None) is None:
            raise ValueError(
                "Region ID is not found in JSON data, we need it to build image-region idx to image mapping"
            )
        image_id = metadata["metadata_image_id"]
        region_id = metadata["metadata_region_id"]
        mapping[image_id][region_id] = data
    return mapping


@click.command()
@click.option("--tsv_path", "-t", help="Path to TSV file")
@click.option("--json_path", "-j", help="Path to JSON file")
@click.option("--output_path", "-o", help="Path to output JSON file")
def main(tsv_path, json_path, output_path):
    """we build the json file with images based on the image tsv file.

    Args:
        tsv_path (_type_): _description_
        json_path (_type_): _description_
        output_path (_type_): _description_
    """
    tsv_data = TSVFile(tsv_path)
    with open(json_path, "r") as f:
        json_data = json.load(f)

    tsv_data_len = len(tsv_data)
    json_data_len = len(json_data)

    mapping = build_mapping_for_json(json_data)

    if tsv_data_len != json_data_len:
        logger.warning(f"Lengths of img TSV and JSON data are not equal: {tsv_data_len} != {json_data_len}")

    with open(output_path, "w") as f:
        is_first = True
        for tsv_sample in tqdm.tqdm(tsv_data):
            tsv_ident, media_b64 = tsv_sample
            image_id, region_cnt, region_id = list(map(int, tsv_ident.split("-")))
            json_sample = mapping[image_id][region_id]
            json_sample["media_b64"] = media_b64
            if is_first:
                string = json.dumps(json_sample)
                f.write("[" + string)
                is_first = False
            else:
                string = json.dumps(json_sample)
                f.write("\n," + string)
        f.write("]")


if __name__ == "__main__":
    main()
