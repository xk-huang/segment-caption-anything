import json
import os
import datasets
from datasets.download.download_manager import DownloadManager
import dotenv
from urllib.parse import urlparse
import re

logger = datasets.logging.get_logger(__name__)


_BASE_IMAGE_METADATA_FEATURES = {
    "image_id": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
    "file_name": datasets.Value("string"),
    "coco_url": datasets.Value("string"),
    "task_type": datasets.Value("string"),
}

_BASE_REGION_FEATURES = {
    # NOTE: one of them is 900100184613, which is out of the range of int32
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
    # "area": datasets.Value("int32"),
    # "phrase_conf": datasets.Value("float32"),
}


_ANNOTATION_FEATURES = {
    "region_descriptions": {"regions": [_BASE_REGION_FEATURES]},
    "mask_region_descriptions": {"regions": [_BASE_MASK_REGION_FEATURES]},
}

import json
import os
import datasets
import dotenv
from pycocotools.coco import COCO

logger = datasets.logging.get_logger(__name__)


_BASE_IMAGE_METADATA_FEATURES = {
    "image_id": datasets.Value("int32"),
    "width": datasets.Value("int32"),
    "height": datasets.Value("int32"),
    "file_name": datasets.Value("string"),
    "coco_url": datasets.Value("string"),
    "task_type": datasets.Value("string"),
}

_BASE_REGION_FEATURES = {
    # NOTE: one of them is 900100184613, which is out of the range of int32
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
    # "area": datasets.Value("int32"),
    # "phrase_conf": datasets.Value("float32"),
}


_ANNOTATION_FEATURES = {
    "region_descriptions": {"regions": [_BASE_REGION_FEATURES]},
    "mask_region_descriptions": {"regions": [_BASE_MASK_REGION_FEATURES]},
}


class VisualGenomeDensecapLocalConfig(datasets.BuilderConfig):
    def __init__(
        self,
        name,
        splits,
        with_image: bool = True,
        with_mask: bool = False,
        base_dir: str = None,
        base_annotation_dir: str = None,
        task_type: str = "caption",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.splits = splits
        self.with_image = with_image
        self.with_mask = with_mask
        self.base_dir = base_dir
        self.base_annotation_dir = base_annotation_dir
        self.task_type = task_type

    @property
    def features(self):
        if self.with_mask is True:
            raise ValueError("with_mask=True is not supported yet in COCO caption.")

        annoation_type = "mask_region_descriptions" if self.with_mask else "region_descriptions"
        logger.info(f"Using annotation type: {annoation_type} due to with_mask={self.with_mask}")
        return datasets.Features(
            {
                **({"image": datasets.Image()} if self.with_image else {}),
                **_BASE_IMAGE_METADATA_FEATURES,
                **_ANNOTATION_FEATURES[annoation_type],
            }
        )


class VisualGenomeDensecapLocalDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = VisualGenomeDensecapLocalConfig
    BUILDER_CONFIGS = [
        # NOTE: we do not need test as it lacks visual promptsc
        # COCOBuilderConfig(name="2017", splits=["train", "valid", "test"]),
        VisualGenomeDensecapLocalConfig(name="densecap", splits=["train", "test"]),
    ]
    DEFAULT_CONFIG_NAME = "densecap"
    config: VisualGenomeDensecapLocalConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=self.config.features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
            version=self.config.version,
        )

    def _split_generators(self, dl_manager: DownloadManager):
        """
        The data file structure:
        base_dir:
        - VG_100K/%d.jpg
        - VG_100K_2/%d.jpg

        annotation_dir:
        - image_data.json (the image meta data)
        - densecap_splits.json (from densecap github repo)
        - region_descriptions.json (from region descriptions)
        - {train,test}.json (from grit github repo)

        NOTE: Compare `grit` and `densecap`:
        1. Image split: the `test` is the same, the `val` is not used, for `train` the `densecap` has two more images than `grit`: `{1650, 1684}`
        2. Texts are different, the `densecap` is raw, the `grit` is processed
        3. The `image_id` is the same as that in `file url` or `file path`

        Args:
            dl_manager (DownloadManager): _description_

        Returns:
            _type_: _description_
        """
        base_dir = self.config.base_dir
        base_annotation_dir = self.config.base_annotation_dir

        if base_dir is None:
            raise ValueError("base_dir is not provided.")
        if base_annotation_dir is None:
            raise ValueError("vg_annotation_dir is not provided.")

        PATHS = {
            "base_dir": base_dir,
            "base_annotation_dir": base_annotation_dir,
        }
        split_kwargs_ls = []
        splits = self.config.splits
        for split in splits:
            if split not in ["train", "test"]:
                raise ValueError(f"Split {split} is not supported yet.")
            if split == "train":
                dataset = datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={**PATHS, "split": "train"})
            elif split == "test":
                dataset = datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={**PATHS, "split": "test"})
            split_kwargs_ls.append(dataset)
        return split_kwargs_ls

    def _generate_examples(
        self,
        base_dir,
        base_annotation_dir,
        split,
    ):
        densecap_split_file = os.path.join(base_annotation_dir, "densecap_splits.json")
        densecap_annot_file = os.path.join(base_annotation_dir, "region_descriptions.json")
        densecap_image_meta_file = os.path.join(base_annotation_dir, "image_data.json")

        with open(densecap_split_file, "r") as f:
            densecap_split = json.load(f)
        with open(densecap_annot_file, "r") as f:
            densecap_annot = json.load(f)
        with open(densecap_image_meta_file, "r") as f:
            densecap_image_meta = json.load(f)

        densecap_img_id_to_region = self.build_densecap_img_id_to_region(densecap_annot)
        densecap_img_id_to_img = self.build_densecap_img_id_to_img(densecap_image_meta)
        densecap_split_img_id = densecap_split[split]

        for idx, img_id in enumerate(densecap_split_img_id):
            img = densecap_img_id_to_img[img_id]
            image_metadata = {
                "coco_url": img["url"],
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "image_id": img["image_id"],
            }

            image_dict = {"image": os.path.join(base_dir, img["file_name"])} if self.config.with_image else {}

            regions = []
            for ann in densecap_img_id_to_region[img_id]:
                region_id = ann["region_id"]
                phrase = ann["phrase"]
                region = {
                    "region_id": region_id,
                    "image_id": img_id,
                    "phrases": [phrase],
                    "x": ann["x"],
                    "y": ann["y"],
                    "width": ann["width"],
                    "height": ann["height"],
                }
                if self.config.with_mask is True:
                    region["mask"] = {
                        "size": ann["mask"]["size"],
                        "counts": ann["mask"]["counts"],
                    }
                regions.append(region)

            yield idx, {
                **image_dict,
                **image_metadata,
                "regions": regions,
                "task_type": self.config.task_type,
            }

    @staticmethod
    def build_densecap_img_id_to_region(densecap_annot):
        img_id_to_region = {}
        for img in densecap_annot:
            img_id = img["id"]
            regions = img["regions"]
            img_id_to_region[img_id] = regions
        return img_id_to_region

    def build_densecap_img_id_to_img(self, densecap_image_meta):
        img_id_to_img = {}
        for img in densecap_image_meta:
            img_id = img["image_id"]
            img["file_name"] = self.convert_url_to_path(img["url"])
            img_id_to_img[img_id] = img
        return img_id_to_img

    @staticmethod
    def convert_url_to_path(img_url):
        """Obtain image folder given an image url.

        For example:
        Given `https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg` as an image url, this method returns the local path for that image.
        """
        matches = re.fullmatch(
            r"^https://cs.stanford.edu/people/rak248/(VG_100K(?:_2)?)/([0-9]+\.jpg)$",
            img_url,
        )
        assert matches is not None, f"Got img_url: {img_url}, matched: {matches}"
        folder, filename = matches.group(1), matches.group(2)
        return os.path.join(folder, filename)


_CITATION = """\
@inproceedings{krishnavisualgenome,
  title={Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations},
  author={Krishna, Ranjay and Zhu, Yuke and Groth, Oliver and Johnson, Justin and Hata, Kenji and Kravitz, Joshua and Chen, Stephanie and Kalantidis, Yannis and Li, Li-Jia and Shamma, David A and Bernstein, Michael and Fei-Fei, Li},
  year = {2016},
  url = {https://arxiv.org/abs/1602.07332},
}
"""

_DESCRIPTION = """\
Visual Genome enable to model objects and relationships between objects.
They collect dense annotations of objects, attributes, and relationships within each image.
Specifically, the dataset contains over 108K images where each image has an average of 35 objects, 26 attributes, and 21 pairwise relationships between objects.
"""

_HOMEPAGE = "https://homes.cs.washington.edu/~ranjay/visualgenome/"

_LICENSE = "Creative Commons Attribution 4.0 International License"
