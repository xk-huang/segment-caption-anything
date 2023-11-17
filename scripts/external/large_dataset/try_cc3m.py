import re
from PIL import Image
import base64
import io


def deconfusing(string):
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


data_tsv_path = "split_0.tsv"
with open(data_tsv_path, "r") as f:
    for line in f.readlines():
        line = line.strip()
        name, img = line.split("\t")
        img = base64.b64decode(img)
        img = deconfusing(img)
        img = Image.open(io.BytesIO(img))
        print(img.size)
