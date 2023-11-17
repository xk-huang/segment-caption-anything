import base64
import glob
import io
import json
import os
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import DATA_PATH_TABLE
from utils.misc import get_rank, get_world_size


class ConceptualCaptions12M(Dataset):
    def __init__(self):
        self.data_path = DATA_PATH_TABLE["cc12m"]

        self.ann_file = glob.glob(os.path.join(self.data_path, "split_*.json"))
        self.ann_file.sort()
        self.database = self._build_database()
        self.dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

    def _build_database(self):
        print("Creating database for Conceptual Captions 12M dataset...")
        global_rank = get_rank()
        total_gpu = get_world_size()
        tsv_per_gpu = len(self.ann_file) // total_gpu
        fix_length = 358657 if tsv_per_gpu == 4 else (179213 if tsv_per_gpu == 2 else (717545 if tsv_per_gpu == 8 else 89491))
        tsvpergpu_to_fixlength = {
            1:89491,
            2:179213,
            4:358657,
            8:717545,
            16:1435090,
            32:2870180,
            64:5740360,
            128:11480720
        } # tsv_per_gpu from 1 to 128
        full_info_tmp = []
        for i in range(tsv_per_gpu):
            with open(os.path.join(self.data_path, f'split_{tsv_per_gpu*global_rank + i}.json')) as f:
                _full_info = json.load(f)
            full_info_tmp.extend(list(_full_info.values()))
        fix_length = tsvpergpu_to_fixlength[tsv_per_gpu]
        print("Truncate to fix length: ", fix_length, " for each GPU out of ", len(full_info_tmp), " images.")
        full_info_tmp = full_info_tmp[:fix_length]

        database=[]
        for info in full_info_tmp:
            database.append(
                {
                    "caption": info["caption"], # should be List[str]
                    "image": f'split_{info["img_location"]}.tsv/{info.get("lineidx_ptr","null")}',
                }
            )
        # filter out the images that cannot be loaded
        print("Loaded {} images".format(len(database)))
        database = [d for d in database if d["image"].split("/")[-1] != "null"]
        print("Filtered {} images".format(len(database)))
        return database

    def deconfusing(self, string):
        confuse_head = r"this is a head".encode("utf-8")
        if string.startswith(confuse_head):
            confuse_code = b"\xff\xdb\x00C\x00\x02\x01"
            string = string[len(confuse_head) :]
            result = re.search(b"\xff\xda", string)
            startofscan = result.span()[0]
            return string[: startofscan - len(confuse_code)] + string[startofscan:]
        else:
            # no confusing for Laion dataset
            return string

    def _load_image(self, path):
        tsv_name, lineidx = path.split(".tsv/")
        _fp = open(tsv_name + ".tsv", "r")
        _fp.seek(int(lineidx))
        _, img = [s.strip() for s in _fp.readline().split("\t")]
        img = base64.b64decode(img)
        img = self.deconfusing(img)
        img = Image.open(io.BytesIO(img))
        _fp.close()
        img = img.convert("RGB")
        return img

    def __getitem__(self, index):
        idb = self.database[index]

        try:
            image = self._load_image(os.path.join(self.data_path, idb["image"]))
        except Exception as e:
            print(
                "Failed to load image {} due to {}, use zero image!".format(
                    idb["image"], e
                )
            )
            image = self.dummy
        ret = {"image": image, "captions": idb["caption"]}
        return ret

    def __len__(self):
        return len(self.database)


if __name__ == "__main__":
    dataset = ConceptualCaptions12M()
    print(len(dataset))
    print(dataset[0])
