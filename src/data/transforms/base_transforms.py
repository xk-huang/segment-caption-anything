import logging
from ...models.sam.processing_sam import SamProcessor
from transformers import PreTrainedTokenizer
from typing import Optional, Union, List, Dict, Any
import torch
import pycocotools.mask
from collections import defaultdict
from dataclasses import dataclass
from PIL import Image
import random
from . import detectron2_transforms as T
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SCADataTransformedFeature:
    # num_regions = 35
    images: Image
    pixel_values: torch.Tensor  # (3, 1024, 1024)
    original_sizes: torch.Tensor  # (2,)
    reshaped_input_sizes: torch.Tensor  # (2,)
    input_ids: List[List[int]]  # (num_regions, DIFFERENT_length)
    attention_mask: List[List[int]]  # (num_regions, DIFFERENT_length)
    input_boxes: torch.Tensor  # (num_regions,, 4)
    metadata_input_boxes: torch.Tensor  # (num_regions,, 4)
    metadata_captions: List[str]  # (num_regions,)
    metadata_image_id: torch.Tensor  # (num_regions,)
    metadata_region_id: torch.Tensor  # (num_regions,)


# NOTE(xiaoke): used in collator to batchify the regions
# based on the minimum number of regions in the batch
REGION_KEYS = (
    "input_points",
    "input_labels",
    "input_boxes",
    "metadata_input_boxes",
    "metadata_image_id",
    "metadata_region_id",
)

# NOTE(xiaoke): used in `generate()` to remove unused keys
UNUSED_KEYS_IN_GENERATE = (
    "metadata_captions",
    "metadata_input_boxes",
    "metadata_image_id",
    "metadata_region_id",
    "task_type",
)


class SamCaptionerDataTransform:
    def __init__(
        self,
        sam_processor: SamProcessor,
        tokenizer: PreTrainedTokenizer,
        split: Optional[str] = None,
        num_masks_per_sample: Optional[int] = None,
        data_transforms: Optional[Any] = None,
    ):
        if split is None:
            raise ValueError("split in DataTransfom must be provided")
        if split not in ["train", "inference"]:
            raise ValueError(f"split in DataTransfom must be one of ['train', 'inference'], got {split}")

        if num_masks_per_sample is None and split == "train":
            num_masks_per_sample = 64
            logger.info(f"num_masks_per_sample not provided, defaulting to {num_masks_per_sample}")

        self.sam_processor = sam_processor
        self.tokenizer = tokenizer
        self.split = split
        self.num_masks_per_sample = num_masks_per_sample

        logger.info(f"[{self.__class__}] this is split: {split}")
        max_length = self.tokenizer.model_max_length
        logger.info(f"max_length is {max_length} for caption tokens")

        if data_transforms is None:
            logger.info(f"data_transforms not provided, defaulting to not using it.")

        self.augmentation = None  # NOTE: either split=inference or data_transforms=None.
        if data_transforms is not None and split == "train":
            logger.info("Applying data augmentations during training")
            logger.info(f"data_transforms is {data_transforms}")
            min_scale = data_transforms.min_scale
            max_scale = data_transforms.max_scale
            image_size = data_transforms.image_size

            augmentation = []
            augmentation.append(T.RandomFlip(horizontal=True))
            augmentation.append(
                T.ResizeScale(
                    min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
                )
            )
            # NOTE: use pad_value=0 to normlized images (with imagenet mean and std)
            augmentation.append(T.FixedSizeCrop(crop_size=(image_size, image_size), pad_value=0))
            self.augmentation = augmentation
            logger.info(f"augmentation is {augmentation}")

            image_mean, image_std = (
                sam_processor.image_processor.image_mean,
                sam_processor.image_processor.image_std,
            )
            image_mean = np.array(image_mean).reshape(1, 1, 3)
            image_std = np.array(image_std).reshape(1, 1, 3)

            self._normalize_np_image = lambda x: (x / 255 - image_mean) / image_std
            self._denormalize_np_image = lambda x: ((x * image_std + image_mean) * 255).astype(np.uint8)

    def __call__(self, examples):
        """_summary_

        Args:
            examples (_type_): _description_

        Returns:
            dict: tensors
                # sam related
                - images: (batch_size, 3, H, W)
                - pixel_values: (batch_size, 3, H, W)
                - input_points
                - input_labels
                - input_boxes

                # used to crop and post-processing
                - original_sizes (batch_size, 2)
                - reshaped_input_sizes (batch_size, 2)

                # caption related
                - input_ids
                - attention_mask
                - labels

                # used for post-processing, or evaluation
                # maybe need to be removed for `.generate()`
                - metadata_input_boxes (batch_size, num_regions, 4)
                - metadata_captions: List[List[str]]
                - metadata_image_id: (batch_size,)
                - metadata_region_id: (batch_size, num_regions)
        """
        batch_image = examples["image"]
        batch_transforms = None
        # NOTE: Apply augmentations to images from detectron2
        if self.augmentation is not None:
            # NOTE: convert image to RGB, since there may be gray images.
            # transformers/image_transforms.py
            # transformers/models/sam/image_processing_sam.py
            # NOTE: convert to np array and normalize with image_mean and image_std
            np_image = [self._normalize_np_image(np.array(image_.convert("RGB"))) for image_ in batch_image]
            batch_aug_image = []
            batch_aug_transforms = []
            for image_ in np_image:
                aug_image_, aug_transforms_ = T.apply_transform_gens(self.augmentation, image_)
                batch_aug_image.append(self._denormalize_np_image(aug_image_))
                batch_aug_transforms.append(aug_transforms_)
            batch_image = batch_aug_image
            batch_transforms = batch_aug_transforms

        # NOTE: Process images with huggingface's preprocessor
        encoding_sam_processor = self.sam_processor(batch_image, return_tensors="pt")
        original_sizes = encoding_sam_processor["original_sizes"]

        encoding_region_processor = defaultdict(list)
        batch_regions = examples["regions"]
        batch_image_id = examples["image_id"]
        for idx, regions in enumerate(batch_regions):
            if batch_transforms is not None:
                transforms = batch_transforms[idx]
            else:
                transforms = None
            _encoding_region_processor = self.process_regions(
                original_sizes[idx : idx + 1], regions, batch_image_id[idx], transforms
            )
            for k, v in _encoding_region_processor.items():
                encoding_region_processor[k].append(v)

        return {
            "images": batch_image,
            **encoding_sam_processor,
            **encoding_region_processor,
            "task_type": examples["task_type"],
        }

    def process_regions(self, original_sizes, regions, image_id_to_be_checked, transforms=None):
        if self.split == "train":
            num_regions = len(regions)
            region_index_ls = torch.randperm(num_regions)
        else:
            region_index_ls = torch.arange(len(regions))

        # boxes, original_caption_ls, mask_ls = self.prepare_regions(regions, region_index_ls)
        # TODO: support mask and point. Now only support box
        original_boxes, original_caption_ls = self.load_regions_by_index_ls(regions, region_index_ls)

        # NOTE: Apply augmentations to regions from detectron2
        if transforms is not None:
            boxes = transforms.apply_box(original_boxes)
            # NOTE: boxes are transformed in np in detectron2, so we need to convert it back to tensor
            boxes = torch.from_numpy(boxes)
        else:
            boxes = original_boxes

        # NOTE: clip and clamp, de-empty the boxes
        # TODO: support mask and point. Now only support box
        # NOTE: the input shape is from the transformed images. We normalize the boxes upon the transformed images.
        # https://github.com/UX-Decoder/Semantic-SAM/blob/e3b9e758ba8832eebdf30e00bc4fe4b45963f010/datasets/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py#L144
        # NOTE: `original_sizes` are the image size after transformed if `transforms` is not None.
        # NOTE: filter boxes, region_index_ls, original_boxes, original_caption_ls
        boxes, region_index_ls, original_boxes, original_caption_ls = normalize_and_filter_transformed_boxes(
            original_sizes, boxes, region_index_ls, original_boxes, original_caption_ls, mode=self.split
        )

        if self.split == "train":
            # NOTE: sample regions with `num_masks_per_sample`
            # NOTE: the regions are not enough, from 3 to 250+. We futher handle this in collator for not enough regions (< num_masks_per_sample).
            boxes = boxes[: self.num_masks_per_sample]
            region_index_ls = region_index_ls[: self.num_masks_per_sample]
            original_boxes = original_boxes[: self.num_masks_per_sample]
            original_caption_ls = original_caption_ls[: self.num_masks_per_sample]

        # NOTE: Process regions with huggingface's preprocessor
        input_boxes = boxes.unsqueeze(0)  # (batch_size (1), num_regions, 4)
        assert len(original_sizes) == len(input_boxes)
        # NOTE: sam's processor only convert prompts to tensors and scale it to 1024.
        # It does NOT clip, clamp, or de-empty the points, boxes, and masks.
        # NOTE: We have to make sure every sample has the same keys, or /anaconda/envs/sca-v2/lib/python3.9/site-packages/datasets/arrow_dataset.py:2799L raises error due to missing value.
        # XXX: unused keys of SAM are in None. But the empty samples are also None. Thus we need to use "input_ids" and "attention_mask" to filter the empty samples.
        encoding_prompts = {
            "input_boxes": None,
            "input_points": None,
            "input_labels": None,
        }
        try:
            encoding_prompts.update(
                self.sam_processor.process_prompts(original_sizes, input_boxes=input_boxes, return_tensors="pt")
            )
        except Exception as e:
            logger.warning(
                f"Error in processing prompts: {e} maybe due to no visual prompts."
                "As they are truncate by the normalization and filtering of boxes."
                "e.g., large scale jittering."
            )

        # NOTE(xiaoke): remove the batch dimension (=1)
        for k, v in encoding_prompts.items():
            if v is not None:
                encoding_prompts[k] = v.squeeze(0)

        # NOTE(xiaoke): first truncate based on `model_max_length`.
        # the dynamic batch padding happens in the collator.
        # NOTE: We have to make sure every sample has the same keys, or /anaconda/envs/sca-v2/lib/python3.9/site-packages/datasets/arrow_dataset.py:2799L raises error due to missing value.
        # NOTE: empty sample has both "input_ids": None, "attention_mask": None
        encoding_tokenizer = {"input_ids": None, "attention_mask": None}
        try:
            encoding_tokenizer.update(self.process_tokens(original_caption_ls))
        except Exception as e:
            logger.warning(
                f"Error in processing tokens: {e} maybe due to no captions."
                "As they are truncate by the normalization and filtering of boxes."
                "e.g., large scale jittering."
            )

        # NOTE(xiaoke): get metadata by index
        metadata_image_id = torch.tensor([regions[index]["image_id"] for index in region_index_ls])
        metadata_region_id = torch.tensor([regions[index]["region_id"] for index in region_index_ls])
        if any(image_id_to_be_checked != metadata_image_id):
            logger.warning(
                f"There are image_id in region different from the ture image_id: {image_id_to_be_checked} != {metadata_image_id}"
            )

        return {
            **encoding_tokenizer,
            **encoding_prompts,
            "metadata_input_boxes": original_boxes,
            "metadata_captions": original_caption_ls,
            "metadata_image_id": metadata_image_id,
            "metadata_region_id": metadata_region_id,
        }

    def load_regions_by_index_ls(self, regions, region_index_ls):
        # mask_ls = []
        box_ls = []
        original_caption_ls = []
        for region_index in region_index_ls:
            region = regions[region_index]

            # TODO(xiaoke): add mask to support point prompts
            # mask = torch.tensor(pycocotools.mask.decode(region["mask"]))
            box = torch.tensor(
                [
                    region["x"],
                    region["y"],
                    region["x"] + region["width"],
                    region["y"] + region["height"],
                ]
            )
            # XXX(xiaoke): How to support multiple gt captions?
            if self.split == "train":
                # NOTE(xiaoke): Now we only randomly take one caption among multiple gt captions.
                gt_caption = random.choice(region["phrases"])
            else:
                gt_caption = region["phrases"][0]

            # mask_ls.append(mask)
            box_ls.append(box)
            original_caption_ls.append(gt_caption)

        boxes = torch.stack(box_ls)  # (num_regions, 4)
        # return boxes, original_caption_ls, mask_ls
        return boxes, original_caption_ls

    def process_tokens(self, texts):
        return self.tokenizer(texts, truncation=True)


def normalize_and_filter_transformed_boxes(
    original_sizes, transformed_boxes, region_index_ls, original_boxes, original_caption_ls, mode
):
    # NOTE: filter empty regions
    transformed_boxes = transformed_boxes.clamp(min=0)
    # NOTE: original sizes: (y, x), while transformed boxes: (x, y, x, y)
    # Flip as in `transform_instance_annotations` https://github.com/facebookresearch/detectron2/blob/2409af0bf0d4bdcc685feb6d2c7fd659828acac4/detectron2/data/detection_utils.py#L263
    transformed_boxes = torch.minimum(transformed_boxes, original_sizes.expand(2, 2).flatten().flip(0))

    if mode == "train":
        keep_indices = _box_nonempty(transformed_boxes)

        transformed_boxes = transformed_boxes[keep_indices]
        region_index_ls = region_index_ls[keep_indices]
        original_boxes = original_boxes[keep_indices]
        # NOTE: the implicit cast from tensor to numpy array may cause error when its size is 1. It cast [True] to [1], leading to index out of range.
        # Thus we need to explicit cast tensor to numpy array.
        original_caption_ls = np.array(original_caption_ls)[keep_indices.numpy()].tolist()

    return transformed_boxes, region_index_ls, original_boxes, original_caption_ls


def _box_nonempty(box, threshold=1e-5):
    # Copy from: https://github.com/facebookresearch/detectron2/blob/2bd05b42983468c50a1c80d5e7dc3952980e1cd4/detectron2/structures/boxes.py#L199
    # The threshold: https://github.com/facebookresearch/detectron2/blob/2409af0bf0d4bdcc685feb6d2c7fd659828acac4/detectron2/data/detection_utils.py#L498
    widths = box[..., 2] - box[..., 0]
    heights = box[..., 3] - box[..., 1]
    keep = (widths > threshold) & (heights > threshold)

    return keep


class SCADataTransform(SamCaptionerDataTransform):
    def process_tokens(self, texts):
        # XXX(xiaoke): We trim two tokens for the "virtual" <BOS> and true <EOS>.
        outputs = self.tokenizer(texts, truncation=True, max_length=self.tokenizer.model_max_length - 2)
        # NOTE(xiaoke): add eos token
        # XXX(xiaoke): since we do not add the <BOS> token in both training and inference stage.
        # Therefore, we need to shift the labels to the right by one. However, this leads to the situation where labels have one more token than the input_ids,
        # and the max_length of the labels is larger by 1 than tokeinizer.model_max_length, which cause error in trainer eval when compute eval loss due to mismatched last dim during cross-batch-region paddding.
        # Our solution is to trim the input_ids and attention_mask by 1.
        # Check `src/data/collator.py:SCADataCollator:prepare_labels` for details.
        for i in range(len(outputs["input_ids"])):
            outputs["input_ids"][i] += [self.tokenizer.eos_token_id]
            outputs["attention_mask"][i] += [1]
        return outputs
