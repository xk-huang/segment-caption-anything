# Data

By default, we cache the data in `.data.cache/`.

## About datasets loading

We use `datasets` to load data.

For `.zip` files (e.g., VG, RefCOCOs), the streaming fetching is extremely slow due to data access via random indexes.

In contrast, loading `.tar` or `.tsv` files is faster as the data are accessed by order.

As a result, we only use `streaming=True` in when loading `SA1B-Cap` due to its huge memory consumption, whereas for VG and RefCOCOs, we set `streaming=False`.

TODO: use webdatasets for openimage (and sa1b).

## About Data preprocessing

`data/transforms.py`: take each sample, process all the regions inside it:
1. image: using SAM processor to resize and pad images to 1024x1024.
2. region box/ point / mask: use SAM processor to process the prompts.
3. region captions: Use LM processor to do tokenization; For SCA, we need to add "virtual" <BOS> and true <EOS>.

`data/collator.py`: take in multiple processed samples, and form tensors in the batch format:
1. If the number of regions is not the same among the samples, we chunk each of them to the minimum number of regions.
2. For captions, we need to pad the <PAD> tokens during batchifying.

### Code dev

1. `src/data/transforms`
2. Add arguments in `src/arguments.py` 
3. Add arguments in the function in `src/train.py`

The problem: generting random number with numpy in multi process data loader
- https://pytorch.org/docs/stable/notes/faq.html#my-data-loader-workers-return-identical-random-numbers
```
transformers/trainer_utils.py
detectron2/data/build.py
```
However, we use `datasets`'s `map`, which do not use sub-processes.


## Visual Genome

Editted from https://huggingface.co/datasets/visual_genome/blob/main/visual_genome.py, we can load the data stored on Azure.

- the broken links are fixed in https://huggingface.co/datasets/visual_genome/discussions/3#649d99c26a066a00a087b80d (as of 06/30/2023)

if all parameters in `src/conf/data/vg_densecap.yaml` are set to `null`, the loading scripts will use the default urls.
If you want to load data from Azure, you **MUST UPDATE THE SAS KEY**.

## RefCOCO series

Use refer2 for referring expression generation. The paper is SLR.
- https://github.com/lichengunc/refer2
- https://arxiv.org/abs/1612.09542
- Thanks to [easy-to-understand-REG](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2) which points out the data evolving problem, and upload the evaluation sentences.

refcoco, location
refcoco+, no location
refcocog, with or without location
"testA" and "testB" sets in RefCoco and RefCoco+ contain only people and only non-people respectively.

## SA1B-Cap

###  The implementation of streaming loading in `datasets`

### Load with azcopy

Firstly, Each tar or tsv file is downloaded to local host with `azcopy` to a temporary dictory `/tmp/$PRFIX-$HASH_OF_URL`.

After all file loading handles are release, the file will be removed.
.

After all file loading handles are release, the file will be removed.

### Legacy solution

The `open` function of Python is extened with streaming loading from the Internet by `xopen` in [`datasets.download.streaming_download_manager`](https://github.com/huggingface/datasets/blob/029227a116c14720afca71b9b22e78eb2a1c09a6/src/datasets/download/streaming_download_manager.py#L471).

After that, `xopen` is futher patched into `open` by [`datasets.streaming`](https://github.com/huggingface/datasets/blob/029227a116c14720afca71b9b22e78eb2a1c09a6/src/datasets/streaming.py#L80).

There is an attribute called `is_streaming` in `dl_manager` object in data scripts which can indicate the whether the data are loaded with streaming mode or not.


## OpenImages

### Webdataset and pytorch-dalle

There are V6 (maybe) in webdataset format (i.e., `tar`)
https://webdataset.github.io/webdataset/gettingstarted/ and https://github.com/lucidrains/DALLE-pytorch

```
cd ~
mkdir webdataset-openimages
cd webdataset-openimages
# for i in http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar; do
for i in {000000..000554}; do
echo $i
wget http://storage.googleapis.com/nvdata-openimages/openimages-train-$i.tar
done
cd ..
```

Train split: 523 GB

### Fiftyone

Openimages v6 and v7

(Use Fiftyone to load the 'train' split of Openimages is extremely slow, as it loads the data into memory, which takes about 3 hours.)

https://docs.voxel51.com/integrations/open_images.html
https://docs.voxel51.com/api/fiftyone.zoo.datasets.base.html#fiftyone.zoo.datasets.base.OpenImagesV7Dataset

Full split stats:
- Train split: 1,743,042 images (513 GB)
- Test split: 125,436 images (36 GB)
- Validation split: 41,620 images (12 GB)

Download OpenImagesV7 detections from fiftyone:

```python
import fiftyone as fo
import fiftyone.zoo as foz


validation_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
)
test_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="test",
    label_types=["detections"],
)
train_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
)
```


## Detection data: COCO instance, Objects365, v3det

The default task_type is `recognition`.

If you want to activate the task tokens for `caption`, please use `*task_type_caption*.yaml`

Also see [./MODEL.md#multitaskv2](./MODEL.md#multitaskv2).

## Panoptic Segmentation Data: COCO Panoptic, ADE20k panoptic

From Mask2Former: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md
- It provides code to convert data to panoptic format of detectron2.
- It requires `Detectron2` and `git+https://github.com/cocodataset/panopticapi.git@7bb4655` to preprocess the data to detectron2 format.

### COCO panoptic

https://cocodataset.org/#download

```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip panoptic_annotations_trainval2017.zip
unzip annotations/panoptic_train2017.zip
unzip annotations/panoptic_val2017.zip

DETECTRON2_DATASETS= python datasets/prepare_coco_semantic_annos_from_panoptic_annos.py
```

### ADE20k Panopitc

http://sceneparsing.csail.mit.edu/

```
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
cd ADEChallengeData2016

wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xvf annotations_instance.tar

DETECTRON2_DATASETS= python datasets/prepare_ade20k_sem_seg.py
DETECTRON2_DATASETS= python datasets/prepare_ade20k_pan_seg.py
DETECTRON2_DATASETS= python datasets/prepare_ade20k_ins_seg.py

DETECTRON2_DATASETS=/home/t-yutonglin/xiaoke/segment-caption-anything-v2/tmp/data/mask2former_data python datasets/prepare_ade20k_ins_seg.py
```

The format should be in https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
Usage:
1. Add the custom dataset class in `DatasetCatalog`;
2. Add mapper to convert the arbitary custom dataset to the standard format (load images from paths, augment images, and convert images to tensors);
3. `MetadataCatalog` contains info that is shared for all samples, like class labels.

Check data registator
Then check how the data is load with built-in function
Check mapper

## Compare The data loading (image) between [[detectron 2]] and [[hugging face - datasets library]]

From [[hugging face - datasets library]], they are similar:

1. A like, the data script is the dataset that provides image paths and labels (load a json)
  1. Difference: The **difference** is that we merge different dataset here. We should merge latter
2. Then we use a transform function to load and process images and labels
3. We define a collator for dataloader
  1. Improvement: Here is the place to merge multiple dataset, by merging the dataloader. In [[OpenSEED]], it return `{"coco": coco_batch, "o365": o365_batch}`
