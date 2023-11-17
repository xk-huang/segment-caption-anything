import torch
from torch.utils.data import Dataset
import io
import re
import os
import json
import base64
from PIL import Image
import random
import numpy as np

import torch.distributed as dist
from .utils import DATA_PATH_TABLE


class Laion400M(Dataset):
    def __init__(self):
        # create dataset
        self.database = []
        self.data_path = DATA_PATH_TABLE["laion400m"]
        self.load_dataset_anno('Laion-PerGPU')
            
    def load_dataset_anno(self, data_set_type):
        print("Creating database for Laion 400M dataset...")
        if data_set_type == 'Laion-PerGPU':
            # tsv -version
            global_rank = int(dist.get_rank())
            total_gpu = int(dist.get_world_size())
            tsv_per_gpu = 2048 // total_gpu
            full_info_tmp = []
            full_info = []
            fix_length = 197000
            self.fix_length = fix_length
            for i in range(tsv_per_gpu):
                _full_info = json.load(open(os.path.join(self.data_path, f'split_{tsv_per_gpu*global_rank + i}.json')))
                full_info_tmp = list(_full_info.values())[:fix_length]
                if len(full_info_tmp) < fix_length:
                    print(f' warning :: laion only have {len(full_info_tmp)}, need {fix_length}, resample to it!!')
                    pad_idx = list(range(len(full_info_tmp)))
                    random.shuffle(pad_idx)
                    pad_idx = pad_idx[:fix_length-len(full_info_tmp)]
                    pad_info = []
                    for _i in pad_idx:
                        pad_info.append(full_info_tmp[_i])
                    full_info_tmp = full_info_tmp + pad_info
                for info in full_info_tmp:
                    self.database.append([info['caption'], os.path.join(self.data_path, f'split_{tsv_per_gpu*global_rank + i}.tsv/{info["lineidx_ptr"]}')])               
            dist.barrier()
        else:
            pass
    
    def deconfusing(self, string):
        confuse_head = r'this is a head'.encode('utf-8')
        if string.startswith(confuse_head):
            confuse_code = b'\xff\xdb\x00C\x00\x02\x01'
            string = string[len(confuse_head):]
            result = re.search(b'\xff\xda', string)
            startofscan = result.span()[0]
            return string[:startofscan-len(confuse_code)] + string[startofscan:]
        else:
            # no confusing at all
            return string

    def _load_image(self, path):
        if '.tsv/' in path:
            tsv_name, lineidx = path.split('.tsv/')
            _fp = open(tsv_name+'.tsv', 'r')
            _fp.seek(int(lineidx))
            _, img = [s.strip() for s in _fp.readline().split('\t')]
            img = base64.b64decode(img)
            img = self.deconfusing(img)
            img = Image.open(io.BytesIO(img))
            _fp.close()
            return img.convert("RGB")

    def __getitem__(self, index):
        index = index
        idb = self.database[index]
        img = self._load_image(idb[1])
        sentence = idb[0]
        
        if sentence is not None:
            sentence = [sentence]
        else:
            print("!!caption is empty, use 'an image' to replace.")
            sentence = ["an image"]

        return {"image": img, "captions": sentence}

    def __len__(self):
        return len(self.database) * int(dist.get_world_size())
