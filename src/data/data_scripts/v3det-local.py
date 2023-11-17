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


class V3DetBuilderConfig(datasets.BuilderConfig):
    def __init__(
        self,
        name,
        splits,
        with_image: bool = True,
        with_mask: bool = True,
        v3det_base_dir: str = None,
        v3det_base_annotations_dir: str = None,
        task_type: str = "recognition",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.splits = splits
        self.with_image = with_image
        self.with_mask = with_mask
        self.v3det_base_dir = v3det_base_dir
        self.v3det_base_annotations_dir = v3det_base_annotations_dir
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
class V3DetDataset(datasets.GeneratorBasedBuilder):
    """An example dataset script to work with the local (downloaded) V3Det dataset"""

    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = V3DetBuilderConfig
    BUILDER_CONFIGS = [
        # NOTE: we do not need test as it lacks visual promptsc
        # V3DetBuilderConfig(name="2017", splits=["train", "valid", "test"]),
        V3DetBuilderConfig(name="v1", splits=["train", "valid"]),
    ]
    DEFAULT_CONFIG_NAME = "v1"
    config: V3DetBuilderConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        if self.config.with_mask is True:
            raise ValueError("This v3det does not support `with_mask=True`.")

        # data_dir = self.config.data_dir
        v3det_base_dir = self.config.v3det_base_dir
        v3det_base_annotations_dir = self.config.v3det_base_annotations_dir
        if v3det_base_dir is None:
            raise ValueError(
                "This script is supposed to work with local (downloaded) objects356 dataset. The argument `v3det_base_dir` in `load_dataset()` is required."
            )
        if v3det_base_annotations_dir is None:
            raise ValueError(
                "This script is supposed to work with local (downloaded) objects356 dataset. The argument `v3det_base_annotations_dir` in `load_dataset()` is required."
            )

        # NOTE: the config is from https://?.blob.core.windows.net/onemodel/?/V3Det/, which is provided by XK Huang.
        _DL_URLS = {
            "train": v3det_base_dir,
            "val": v3det_base_dir,
            "annotations_train": os.path.join(v3det_base_annotations_dir, "v3det_2023_v1_train.json"),
            "annotations_val": os.path.join(v3det_base_annotations_dir, "v3det_2023_v1_val.json"),
        }

        archive_path = _DL_URLS

        splits = []
        for split in self.config.splits:
            if split == "train":
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": archive_path["annotations_train"],
                        "image_dir": archive_path["train"],
                        "split": "train",
                    },
                )
            elif split in ["val", "valid", "validation", "dev"]:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": archive_path["annotations_val"],
                        "image_dir": archive_path["val"],
                        "split": "val",
                    },
                )
            else:
                continue

            splits.append(dataset)

        return splits

    # NOTE: There are 6 urls are expired. (09/06/23)
    MISSING_IMAGES = {
        "images/n11708857/9_80_2421297739_7c2a68f404_c.jpg",
        "images/n11794139/10_47_5109432138_4201649ce4_c.jpg",
        "images/a00012462/11_256_9645439415_6f50f1599f_c.jpg",
        "images/Q494417/34_1125_10108827626_78b07f281b_c.jpg",
        "images/Q3216816/10_1372_8165953126_129e378cc2_c.jpg",
        "images/a00006662/34_145_18056000502_e12645da0a_c.jpg",
    }

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self,
        json_path,
        image_dir,
        split,
    ):
        """Yields examples as (key, example) tuples."""

        coco = COCO(json_path)
        img_ids = coco.getImgIds()
        for idx, img_id in enumerate(img_ids):
            img = coco.imgs[img_id]
            image_metadata = {
                "coco_url": "",
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
            }
            # NOTE: the config is from https://?.blob.core.windows.net/?/data_raw/V3Det/, which is provided by XK Huang.
            image_dict = {"image": os.path.join(image_dir, img["file_name"])} if self.config.with_image else {}

            if img["file_name"] in self.MISSING_IMAGES:
                logger.warning(f"Image {img['file_name']} in V3Det is missing. Skipping.")
                continue

            if img_id not in coco.imgToAnns:
                continue

            annotation = []
            for ann in coco.imgToAnns[img_id]:
                x, y, width, height = ann["bbox"]
                x, y, width, height = int(x), int(y), int(width), int(height)
                annotation_dict = {
                    # NOTE: one of them is 900100184613, which is out of the range of int32
                    "region_id": ann["id"],
                    "image_id": ann["image_id"],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }

                phrases = []
                category_id = ann["category_id"]
                category = coco.cats[category_id]
                phrases.append(category["name"])
                # TODO: add supercategory
                # phrases.append(category["supercategory"])
                annotation_dict["phrases"] = phrases

                if self.config.with_mask:
                    mask_dict = coco.annToRLE(ann)
                    mask_dict = {
                        "size": mask_dict["size"],
                        "counts": mask_dict["counts"].decode("utf-8"),  # NOTE: otherwise, it leads to core dump error.
                    }
                    annotation_dict["mask"] = mask_dict

                annotation.append(annotation_dict)
            annotation = {"regions": annotation}

            yield idx, {**image_dict, **image_metadata, **annotation, "task_type": self.config.task_type}
