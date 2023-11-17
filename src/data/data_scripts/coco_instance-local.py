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


class COCOBuilderConfig(datasets.BuilderConfig):
    def __init__(
        self,
        name,
        splits,
        with_image: bool = True,
        with_mask: bool = True,
        coco_zip_url: str = None,
        coco_annotations_zip_url: str = None,
        task_type: str = "recognition",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.splits = splits
        self.with_image = with_image
        self.with_mask = with_mask
        self.coco_zip_url = coco_zip_url
        self.coco_annotations_zip_url = coco_annotations_zip_url
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
class COCODataset(datasets.GeneratorBasedBuilder):
    """An example dataset script to work with the local (downloaded) COCO dataset"""

    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = COCOBuilderConfig
    BUILDER_CONFIGS = [
        # NOTE: we do not need test as it lacks visual promptsc
        # COCOBuilderConfig(name="2017", splits=["train", "valid", "test"]),
        COCOBuilderConfig(name="2017", splits=["train", "valid"]),
    ]
    DEFAULT_CONFIG_NAME = "2017"
    config: COCOBuilderConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # data_dir = self.config.data_dir
        coco_zip_url = self.config.coco_zip_url
        coco_annotations_zip_url = self.config.coco_annotations_zip_url
        if coco_zip_url is None:
            raise ValueError(
                "This script is supposed to work with local (downloaded) COCO dataset. The argument `coco_zip_url` in `load_dataset()` is required."
            )
        if coco_annotations_zip_url is None:
            raise ValueError(
                "This script is supposed to work with local (downloaded) COCO dataset. The argument `coco_annotations_zip_url` in `load_dataset()` is required."
            )

        # NOTE(xiaoke): load sas_key from .env
        logger.info(f"Try to load sas_key from .env file: {dotenv.load_dotenv('.env')}.")
        coco_zip_url_sas_key = os.getenv("COCO_ZIP_URL_SAS_KEY", "")
        coco_annotations_zip_url_sas_key = os.getenv("COCO_ANNOTATIONS_ZIP_URL_SAS_KEY", "")

        _DL_URLS = {
            "train": os.path.join(coco_zip_url, "train2017"),
            "val": os.path.join(coco_zip_url, "val2017"),
            "test": os.path.join(coco_zip_url, "test2017"),
            "annotations_trainval": coco_annotations_zip_url,
            # "image_info_test": os.path.join(coco_annotations_zip_url, "image_info_test2017.zip")
        }

        archive_path = _DL_URLS

        splits = []
        for split in self.config.splits:
            if split == "train":
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": os.path.join(
                            archive_path["annotations_trainval"], "annotations", "instances_train2017.json"
                        ),
                        # "image_dir": os.path.join(archive_path["train"], "train2017"),
                        "image_dir": archive_path["train"],
                    },
                )
            elif split in ["val", "valid", "validation", "dev"]:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": os.path.join(
                            archive_path["annotations_trainval"], "annotations", "instances_val2017.json"
                        ),
                        # "image_dir": os.path.join(archive_path["val"], "val2017"),
                        "image_dir": archive_path["val"],
                    },
                )
            # NOTE: we do not need test as it lacks visual prompts
            # elif split == "test":
            #     dataset = datasets.SplitGenerator(
            #         name=datasets.Split.TEST,
            #         # These kwargs will be passed to _generate_examples
            #         gen_kwargs={
            #             "json_path": os.path.join(
            #                 archive_path["image_info_test"], "annotations", "image_info_test2017.json"
            #             ),
            #             "image_dir": os.path.join(archive_path["test"], "test2017"),
            #             "split": "test",
            #         },
            #     )
            else:
                continue

            splits.append(dataset)

        return splits

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self,
        json_path,
        image_dir,
    ):
        """Yields examples as (key, example) tuples."""

        coco = COCO(json_path)
        img_ids = coco.getImgIds()
        for idx, img_id in enumerate(img_ids):
            img = coco.imgs[img_id]
            image_metadata = {
                "coco_url": img["coco_url"],
                "file_name": img["file_name"],
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
            }
            image_dict = {"image": os.path.join(image_dir, img["file_name"])} if self.config.with_image else {}

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
