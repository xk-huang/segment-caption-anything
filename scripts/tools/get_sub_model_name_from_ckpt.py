import sys
import json
import os
import os.path as osp


def get_sub_model_name(ckpt_path):
    ckpt_json_path = osp.join(ckpt_path, "config.json")

    with open(ckpt_json_path, "r") as f:
        ckpt_json = json.load(f)

    return ckpt_json


def parse_sub_model(ckpt_json, sub_model_type):
    if sub_model_type not in ["sam", "lm"]:
        raise ValueError("sub_model_type must be one of [sam, lm], but got {}".format(sub_model_type))

    if sub_model_type == "sam":
        return ckpt_json["_name_or_path"]
    elif sub_model_type == "lm":
        return ckpt_json["text_config"]["_name_or_path"]


if __name__ == "__main__":
    ckpt_path = sys.argv[1]
    sub_model_type = sys.argv[2]
    ckpt_json = get_sub_model_name(ckpt_path)
    sub_model_name = parse_sub_model(ckpt_json, sub_model_type)
    print(sub_model_name)
