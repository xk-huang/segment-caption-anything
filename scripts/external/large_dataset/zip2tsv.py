import os
import io
import re
import json
import base64
import zipfile
import jsonlines
from PIL import Image

from tqdm import tqdm

class ZipReader(object):
    zip_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        if path in zip_bank:
            return zip_bank[path]
        else:
            print("creating new zip_bank")
            zfile = zipfile.ZipFile(path, 'r')
            zip_bank[path] = zfile
            return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_zip_at = path.index('.zip@')
        if pos_zip_at == len(path):
            print("character '@' is not found from the given path '%s'" % (path))
            assert 0
        pos_at = pos_zip_at + len('.zip@') - 1

        zip_path = path[0: pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path
    
    @staticmethod
    def list_folder(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        folder_list = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and \
               len(os.path.splitext(file_foler_name)[-1]) == 0 and \
               file_foler_name != folder_path:
                if len(folder_path) == 0:
                    folder_list.append(file_foler_name)
                else:
                    folder_list.append(file_foler_name[len(folder_path)+1:])

        return folder_list

    @staticmethod
    def list_files(path, extension=['.*']):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            file_foler_name = str.strip(file_foler_name, '/')
            if file_foler_name.startswith(folder_path) and str.lower(os.path.splitext(file_foler_name)[-1]) in extension:
                if len(folder_path) == 0:
                    file_lists.append(file_foler_name)
                else:
                    file_lists.append(file_foler_name[len(folder_path)+1:])

        return file_lists

    @staticmethod
    def list_files_fullpath(path, extension=['.*']):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)

        zfile = ZipReader.get_zipfile(zip_path)
        file_lists = []
        for file_foler_name in zfile.namelist():
            if file_foler_name.startswith(folder_path) and str.lower(os.path.splitext(file_foler_name)[-1]) in extension:
                file_lists.append(file_foler_name)

        return file_lists

    @staticmethod
    def imread(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        im = Image.open(io.BytesIO(data))
        return im

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data

anno = list(jsonlines.open('train_frcnn_oscar.json'))
per_tsv = 22239
cur_count = 0

from torch.utils.data import Dataset, SequentialSampler, DataLoader
class dataset(Dataset): 
    def __init__(self, anno, idx):
        self.zipreader = ZipReader()
        self.anno_dump = {}
        self.anno = []
        count = 0
        for i in anno:
            id = i['image'].split('/')[-1].split('.')[0]
            i.pop('frcnn')
            i['image_id'] = id
            i['img_location'] = idx
            self.anno_dump[id] = i
            self.anno.append(i)
            count += 1
        json.dump(self.anno_dump, open(f'split_{idx}.json', 'w'))
        self.idx = idx

    def confuse(self, string):
        confuse_head = r'this is a head'.encode('utf-8')
        confuse_code = b'\xff\xdb\x00C\x00\x02\x01'
        result = re.search(b'\xff\xda', string)
        sos = result.span()[0]
        return confuse_head + string[:sos] + confuse_code + string[sos:]

    def __getitem__(self, idx):
        cur_anno = self.anno[idx]
        img_path = cur_anno['image']
        try:
            img = self.zipreader.imread(img_path)
            img = img.convert('RGB')
        except:
            return 0
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=95)
        img_bytes = output.getvalue()
        encoded_string = base64.b64encode(self.confuse(img_bytes))
        with open(f'split_{self.idx}.tsv', 'ab+')  as f:
            f.write((str(cur_anno['image_id'])+'\t').encode('utf-8')+encoded_string+'\n'.encode('utf-8'))
        return 1
   
    def __len__(self):
        return len(self.anno)

for i in tqdm(range(128)):
    cur_anno = anno[i * per_tsv: i * per_tsv + per_tsv]
    ds = dataset(cur_anno, i)
    sampler = SequentialSampler(ds)
    loader = DataLoader(ds, sampler=sampler, batch_size=24, num_workers=24)
    for item in tqdm(loader):
        pass
    #os.remove(f'image_split_{i}_part0.zip')
    #os.remove(f'image_split_{i}_part1.zip')


