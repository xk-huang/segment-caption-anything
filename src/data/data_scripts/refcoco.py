import json
import os
import pickle
import logging

import datasets
import pycocotools.mask as mask
import dotenv

logger = logging.getLogger(__name__)


# Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{'{a} }r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

# Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
COCO is a large-scale object detection, segmentation, and captioning dataset.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = "http://cocodataset.org/#home"

# Add the licence for the dataset here if you can find it
_LICENSE = ""

# Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

# This script is supposed to work with local (downloaded) COCO dataset.
_URLs = {}

_BASE_REGION_FEATURES = {
    "region_id": datasets.Value("int64"),
    "image_id": datasets.Value("int32"),
    "phrases": [datasets.Value("string")],
    "x": datasets.Value("int32"),
    "y": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
}

_BASE_MASK_FEATURES = {
    "size": [datasets.Value("int32")],
    "counts": datasets.Value("string"),
}

_BASE_MASK_REGION_FEATURES = {
    "region_id": datasets.Value("int64"),
    "image_id": datasets.Value("int32"),
    "phrases": [datasets.Value("string")],
    "x": datasets.Value("int32"),
    "y": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
    "mask": _BASE_MASK_FEATURES,
}

_ANNOTATION_FEATURES = {
    "region_descriptions": {"regions": [_BASE_REGION_FEATURES]},
    "mask_region_descriptions": {"regions": [_BASE_MASK_REGION_FEATURES]},
}

_BASE_IMAGE_METADATA_FEATURES = {
    "image_id": datasets.Value("int32"),
    # "caption_id": datasets.Value("int64"),
    # "caption": datasets.Value("string"),
    "height": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "file_name": datasets.Value("string"),
    "coco_url": datasets.Value("string"),
    # "image_path": datasets.Value("string"),
    "task_type": datasets.Value("string"),
}


_SPLIT_BYS = {
    "refclef": ["unc", "berkeley"],
    # NOTE: use refer2 by UNC authors
    # "refcoco": ["unc", "google"],
    "refcoco": ["unc"],
    "refcoco+": ["unc"],
    "refcocog": ["umd", "google"],
}
_SPLITS = {
    "refclef-unc": ["train", "val", "testA", "testB", "testC"],
    "refclef-berkeley": ["train", "val", "test"],
    # **{f"refcoco-{_split_by}": ["train", "val", "test"] for _split_by in _SPLIT_BYS["refcoco"]},
    # **{f"refcoco+-{_split_by}": ["train", "val", "test"] for _split_by in _SPLIT_BYS["refcoco+"]},
    **{f"refcoco-{_split_by}": ["train", "val", "testA", "testB"] for _split_by in _SPLIT_BYS["refcoco"]},
    **{f"refcoco+-{_split_by}": ["train", "val", "testA", "testB"] for _split_by in _SPLIT_BYS["refcoco+"]},
    **{f"refcocog-{_split_by}": ["train", "val"] for _split_by in _SPLIT_BYS["refcocog"]},
}
datasets.Split("testA")
datasets.Split("testB")


class RefCOCOBuilderConfig(datasets.BuilderConfig):
    def __init__(
        self,
        name,
        splits,
        with_image=True,
        with_mask=True,
        base_url=None,
        sas_key=None,
        task_type="caption",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.splits = splits
        self.dataset_name = name.split("-")[0]
        self.split_by = name.split("-")[-1]
        self.with_image = with_image
        self.with_mask = with_mask
        self.base_url = base_url
        self.sas_key = sas_key
        self.task_type = task_type

    @property
    def features(self):
        annoation_type = "mask_region_descriptions" if self.with_mask else "region_descriptions"
        logger.info(f"Using annotation type: {annoation_type} due to with_mask={self.with_mask}")
        return datasets.Features(
            {
                **({"image": datasets.Image()} if self.with_image else {}),
                **_BASE_IMAGE_METADATA_FEATURES,
                **_ANNOTATION_FEATURES[annoation_type],
            }
        )


# Name of the dataset usually match the script name with CamelCase instead of snake_case
class RefCOCODataset(datasets.GeneratorBasedBuilder):
    """An example dataset script to work with the local (downloaded) COCO dataset"""

    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = RefCOCOBuilderConfig
    BUILDER_CONFIGS = [RefCOCOBuilderConfig(name=name, splits=splits) for name, splits in _SPLITS.items()]

    DEFAULT_CONFIG_NAME = "refcoco-unc"
    config: RefCOCOBuilderConfig

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = self.config.features

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # NOTE: we use base_url instead of data_dir
        # When we use data_dir, all the paths are relative to the data_dir.
        base_url = self.config.base_url
        if base_url is None:
            raise ValueError(
                "This script is supposed to work with local or remote RefCOCO dataset. It is either a local path or remote url. The argument `base_url` in `load_dataset()` is required."
            )
        logger.info(f"Using base_url: {base_url}")

        # _DL_URLS = {
        #     "train": os.path.join(data_dir, "train2017.zip"),
        #     "val": os.path.join(data_dir, "val2017.zip"),
        #     "test": os.path.join(data_dir, "test2017.zip"),
        #     "annotations_trainval": os.path.join(data_dir, "annotations_trainval2017.zip"),
        #     "image_info_test": os.path.join(data_dir, "image_info_test2017.zip"),
        # }
        _DL_URLS = {}
        if self.config.dataset_name in ["refcoco", "refcoco+", "refcocog"]:
            _DL_URLS["image_dir"] = os.path.join(base_url, "train2014.zip")
        elif self.config.dataset_name == "refclef":
            _DL_URLS["image_dir"] = os.path.join(base_url, "saiapr_tc-12.zip")
        else:
            raise ValueError(f"Unknown dataset name: {self.config.dataset_name}")
        _DL_URLS["annotation_dir"] = os.path.join(base_url, f"{self.config.dataset_name}.zip")

        sas_key = self.config.sas_key
        if sas_key is None:
            # NOTE(xiaoke): load sas_key from .env
            logger.info(f"Try to load sas_key from .env file: {dotenv.load_dotenv('.env')}.")
            sas_key = os.getenv("REFCOCO_SAS_KEY")
        if sas_key is not None and not os.path.exists(base_url):
            logger.info(f"Using sas_key: {sas_key}")
            _DL_URLS = {k: f"{v}{sas_key}" for k, v in _DL_URLS.items()}

        if dl_manager.is_streaming is True:
            raise ValueError(
                "dl_manager.is_streaming is True, which is very slow due to the random access inside zip files with streaming loading."
            )

        archive_path = dl_manager.download_and_extract(_DL_URLS)

        # NOTE(xiaoke): prepare data for index generation
        with open(
            os.path.join(archive_path["annotation_dir"], self.config.dataset_name, f"refs({self.config.split_by}).p"),
            "rb",
        ) as fp:
            refs = pickle.load(fp)
        with open(
            os.path.join(archive_path["annotation_dir"], self.config.dataset_name, f"instances.json"),
            "r",
            encoding="UTF-8",
        ) as fp:
            instances = json.load(fp)
        self.data = {}
        self.data["dataset"] = self.config.dataset_name
        self.data["refs"] = refs
        self.data["images"] = instances["images"]
        self.data["annotations"] = instances["annotations"]
        self.data["categories"] = instances["categories"]
        self.createIndex()
        print(f"num refs: {len(self.Refs)}")

        splits = []
        for split in self.config.splits:
            if split == "train":
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    # gen_kwargs={
                    #     "json_path": os.path.join(
                    #         archive_path["annotations_trainval"], "annotations", "captions_train2017.json"
                    #     ),
                    #     "image_dir": os.path.join(archive_path["train"], "train2017"),
                    #     "split": "train",
                    # },
                    gen_kwargs={
                        "image_dir": archive_path["image_dir"],
                        "split": split,
                    },
                )
            elif split in ["val"]:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    # gen_kwargs={
                    #     "json_path": os.path.join(
                    #         archive_path["annotations_trainval"], "annotations", "captions_val2017.json"
                    #     ),
                    #     "image_dir": os.path.join(archive_path["val"], "val2017"),
                    #     "split": "valid",
                    # },
                    gen_kwargs={
                        "image_dir": archive_path["image_dir"],
                        "split": split,
                    },
                )
            elif split == "test":
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    # gen_kwargs={
                    #     "json_path": os.path.join(
                    #         archive_path["image_info_test"], "annotations", "image_info_test2017.json"
                    #     ),
                    #     "image_dir": os.path.join(archive_path["test"], "test2017"),
                    #     "split": "test",
                    # },
                    gen_kwargs={
                        "image_dir": archive_path["image_dir"],
                        "split": split,
                    },
                )
            elif split in ["testA", "testB", "testC"]:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split(split),
                    # These kwargs will be passed to _generate_examples
                    # gen_kwargs={
                    #     "json_path": os.path.join(
                    #         archive_path["image_info_test"], "annotations", "image_info_test2017.json"
                    #     ),
                    #     "image_dir": os.path.join(archive_path["test"], "test2017"),
                    #     "split": "test",
                    # },
                    gen_kwargs={
                        "image_dir": archive_path["image_dir"],
                        "split": split,
                    },
                )
            else:
                raise ValueError(f"Unknown split name: {split}")

            splits.append(dataset)

        return splits

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self,
        image_dir,
        split,
    ):
        """Yields examples as (key, example) tuples."""
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        ref_ids = self.getRefIds(split=split)
        img_ids = self.getImgIds(ref_ids=ref_ids)

        logger.info(f"Generating examples from {len(ref_ids)} refs and {len(img_ids)} images in split {split}...")

        if self.config.dataset_name in ["refcoco", "refcoco+", "refcocog"]:
            image_dir_name = "train2014"
        elif self.config.dataset_name == "refclef":
            image_dir_name = "saiapr_tc-12"
        else:
            raise ValueError(f"Unknown dataset name: {self.config.dataset_name}")

        for idx, img_id in enumerate(img_ids):
            img = self.Imgs[img_id]
            image_metadata = {
                "coco_url": img.get("coco_url", None),
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
            }
            image_dict = (
                {"image": os.path.join(image_dir, image_dir_name, img["file_name"])} if self.config.with_image else {}
            )

            annotation = []

            img_to_refs = self.imgToRefs[img_id]
            for img_to_ref in img_to_refs:
                ref_to_ann = self.refToAnn[img_to_ref["ref_id"]]
                x, y, width, height = ref_to_ann["bbox"]
                # NOTE: we need to convert float to int
                annotation_dict = {
                    "image_id": img_to_ref["image_id"],
                    "region_id": img_to_ref["ref_id"],
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height),
                }
                annotation_dict["phrases"] = [sent["sent"] for sent in img_to_ref["sentences"]]

                if self.config.with_mask:
                    if type(ref_to_ann["segmentation"][0]) == list:
                        rle = mask.frPyObjects(ref_to_ann["segmentation"], img["height"], img["width"])
                    else:
                        rle = ref_to_ann["segmentation"]
                    mask_dict = rle[0]  # should be a dict, rather a list
                    annotation_dict["mask"] = {
                        "size": mask_dict["size"],
                        "counts": mask_dict["counts"].decode("utf-8"),  # NOTE: otherwise, it leads to core dump error.
                    }
                annotation.append(annotation_dict)
            annotation = {"regions": annotation}
            yield idx, {**image_dict, **image_metadata, **annotation, "task_type": self.config.task_type}

        """
        {
            'coco_url': Value(dtype='string', id=None),
            'file_name': Value(dtype='string', id=None),
            'height': Value(dtype='int32', id=None),
            'image': Image(decode=True, id=None),
            'image_id': Value(dtype='int32', id=None),
            'regions': [{
                'height': Value(dtype='int32', id=None),
                'image_id': Value(dtype='int32', id=None),
                'mask': {
                    'counts': Value(dtype='string', id=None),
                    'size': [Value(dtype='int32', id=None)]
                },
                'phrases': [Value(dtype='string', id=None)],
                'region_id': Value(dtype='int32', id=None),
                'width': Value(dtype='int32', id=None),
                'x': Value(dtype='int32', id=None),
                'y': Value(dtype='int32', id=None)
                }],
            'width': Value(dtype='int32', id=None)
        }
        """

        # _features = [
        #     "image_id",
        #     "caption_id",
        #     "caption",
        #     "height",
        #     "width",
        #     "file_name",
        #     "coco_url",
        #     "image_path",
        #     "id",
        # ]
        # features = list(_features)

        # if split in "valid":
        #     split = "val"

        # with open(json_path, "r", encoding="UTF-8") as fp:
        #     data = json.load(fp)

        # # list of dict
        # images = data["images"]
        # entries = images

        # # build a dict of image_id -> image info dict
        # d = {image["id"]: image for image in images}

        # # list of dict
        # if split in ["train", "val"]:
        #     annotations = data["annotations"]

        #     # build a dict of image_id ->
        #     for annotation in annotations:
        #         _id = annotation["id"]
        #         image_info = d[annotation["image_id"]]
        #         annotation.update(image_info)
        #         annotation["id"] = _id

        #     entries = annotations

        # for id_, entry in enumerate(entries):
        #     entry = {k: v for k, v in entry.items() if k in features}

        #     if split == "test":
        #         entry["image_id"] = entry["id"]
        #         entry["id"] = -1
        #         entry["caption"] = -1

        #     entry["caption_id"] = entry.pop("id")
        #     entry["image_path"] = os.path.join(image_dir, entry["file_name"])

        #     entry = {k: entry[k] for k in _features if k in entry}

        #     yield str((entry["image_id"], entry["caption_id"])), entry

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs:          {ref_id: ref}
        # 2)  Anns:          {ann_id: ann}
        # 3)  Imgs:             {image_id: image}
        # 4)  Cats:          {category_id: category_name}
        # 5)  Sents:         {sent_id: sent}
        # 6)  imgToRefs:     {image_id: refs}
        # 7)  imgToAnns:     {image_id: anns}
        # 8)  refToAnn:      {ref_id: ann}
        # 9)  annToRef:      {ann_id: ref}
        # 10) catToRefs:     {category_id: refs}
        # 11) sentToRef:     {sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        logger.info(f"creating index for {self.config.name}...")
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data["annotations"]:
            Anns[ann["id"]] = ann
            imgToAnns[ann["image_id"]] = imgToAnns.get(ann["image_id"], []) + [ann]
        for img in self.data["images"]:
            Imgs[img["id"]] = img
        for cat in self.data["categories"]:
            Cats[cat["id"]] = cat["name"]

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data["refs"]:
            # ids
            ref_id = ref["ref_id"]
            ann_id = ref["ann_id"]
            category_id = ref["category_id"]
            image_id = ref["image_id"]

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref["sentences"]:
                Sents[sent["sent_id"]] = sent
                sentToRef[sent["sent_id"]] = ref
                sentToTokens[sent["sent_id"]] = sent["tokens"]

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        logger.info("index created.")
        """
        Dataset Statistic:
        refcoco-unc
        Refs 50000
        Anns 196771
        Imgs 19994
        Cats 80
        Sents 142210
        imgToRefs 19994
        imgToAnns 19994
        refToAnn 50000
        annToRef 50000
        catToRefs 78
        sentToRef 142210
        sentToTokens 142210
        """

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=""):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data["refs"]
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data["refs"]
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref["category_id"] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref["ref_id"] in ref_ids]
            if not len(split) == 0:
                if split in ["testA", "testB", "testC"]:
                    # we also consider testAB, testBC, ...
                    refs = [ref for ref in refs if split[-1] in ref["split"]]
                elif split in ["testAB", "testBC", "testAC"]:
                    # rarely used I guess...
                    refs = [ref for ref in refs if ref["split"] == split]
                elif split == "test":
                    refs = [ref for ref in refs if "test" in ref["split"]]
                elif split == "train" or split == "val":
                    refs = [ref for ref in refs if ref["split"] == split]
                else:
                    raise ValueError("No such split [%s]" % split)
        ref_ids = [ref["ref_id"] for ref in refs]
        return ref_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]["image_id"] for ref_id in ref_ids]))
        else:
            image_ids = list(self.Imgs.keys())
        return image_ids
