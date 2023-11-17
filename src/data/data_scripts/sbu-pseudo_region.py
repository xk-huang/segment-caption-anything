import json
import os
import datasets
import dotenv

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
    # "mask_region_descriptions": {"regions": [_BASE_MASK_REGION_FEATURES]},
}


class SBUBuilderConfig(datasets.BuilderConfig):
    def __init__(
        self,
        name,
        splits,
        with_image: bool = True,
        with_mask: bool = False,
        base_dir: str = None,
        base_annotations_dir: str = None,
        task_type: "str" = "caption",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.splits = splits
        self.with_image = with_image
        self.with_mask = with_mask
        self.base_dir = base_dir
        self.base_annotations_dir = base_annotations_dir
        self.task_type = task_type

    @property
    def features(self):
        # annoation_type = (
        #     "mask_region_descriptions" if self.with_mask else "region_descriptions"
        # )
        annoation_type = "region_descriptions"
        logger.info(f"Using annotation type: {annoation_type} due to with_mask={self.with_mask}")
        return datasets.Features(
            {
                **({"image": datasets.Image()} if self.with_image else {}),
                **_BASE_IMAGE_METADATA_FEATURES,
                **_ANNOTATION_FEATURES[annoation_type],
            }
        )


class SBUPseudoRegionDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = SBUBuilderConfig
    BUILDER_CONFIGS = [
        SBUBuilderConfig(name="pseudo_region", splits=["train"]),
    ]
    DEFAULT_CONFIG_NAME = "pseudo_region"
    config: SBUBuilderConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        if self.config.with_mask is True:
            raise ValueError("This sbu does not support `with_mask=True`.")

        base_dir = self.config.base_dir
        base_annotations_dir = self.config.base_annotations_dir
        if base_dir is None:
            raise ValueError("base_dir must be specified.")
        if base_annotations_dir is None:
            raise ValueError("base_annotations_dir must be specified.")

        _DL_URLS = {
            "train": os.path.join(base_dir, "images.zip"),
            "annotations_train": os.path.join(base_annotations_dir, "sbu.json"),
        }

        # NOTE(xiaoke): load sas_key from .env
        logger.info(f"Try to load sas_key from .env file: {dotenv.load_dotenv('.env')}.")
        sbu_url_sas_key = os.getenv("SBU_URL_SAS_KEY", "")
        sbu_annotations_url_sas_key = os.getenv("SBU_ANNOTATIONS_URL_SAS_KEY", "")

        if not os.path.exists(_DL_URLS["train"]):
            _DL_URLS["train"] += sbu_url_sas_key
        if not os.path.exists(_DL_URLS["annotations_train"]):
            _DL_URLS["annotations_train"] += sbu_annotations_url_sas_key

        _DL_URLS = dl_manager.download_and_extract(_DL_URLS)
        logger.warning(f"Downloaded to {_DL_URLS}.")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_dir": _DL_URLS["train"],
                    "annotations_path": _DL_URLS["annotations_train"],
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, image_dir, annotations_path, split):
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        from PIL import Image

        failed_to_load = 0
        for i, annotation in enumerate(annotations):
            # NOTE: the annotation file "sbu.json" is from LAVIS download script.
            image_path = os.path.join(image_dir, annotation["image"])
            url = annotation["url"]
            try:
                image = Image.open(image_path)
            except Exception as e:
                logger.debug(f"Failed to open image {image_path} with url {url}: {e}")
                failed_to_load += 1
                continue
            width, height = image.size
            image_metadata = {
                "image_id": i,
                "width": width,
                "height": height,
                "file_name": annotation["image"],
                "coco_url": url,
            }

            image_dict = {"image": image_path} if self.config.with_image else {}

            annotation = {
                "regions": [
                    {
                        "region_id": i,
                        "image_id": i,
                        "x": 0,
                        "y": 0,
                        "width": width,
                        "height": height,
                        "phrases": [annotation["caption"]],
                    }
                ]
            }

            yield i, {
                **image_metadata,
                **image_dict,
                **annotation,
                "task_type": self.config.task_type,
            }
        logger.info(
            f"Total images: {len(annotations)}, successfully loaded: {len(annotations) - failed_to_load}, failed to load: {failed_to_load}"
        )
