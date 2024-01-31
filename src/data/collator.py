import logging
from ..models.sam.processing_sam import SamProcessor
from transformers import PreTrainedTokenizer
from typing import Optional, Union, List, Dict, Any
import torch
import pycocotools.mask
from collections import defaultdict
from .transforms import REGION_KEYS


logger = logging.getLogger(__name__)


class SamCaptionerDataCollator:
    label_pad_token_id: int = -100

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # NOTE: Filter the batch for any sample samples based on "input_ids"
        nonempty_sample_indices = []
        for i, sample in enumerate(batch):
            if sample["input_ids"] is not None:
                nonempty_sample_indices.append(i)

        if len(nonempty_sample_indices) == 0:
            logger.error(f"batch is empty, skip this batch of data.")
            return None
        elif len(nonempty_sample_indices) < len(batch):
            num_skip = len(batch) - len(nonempty_sample_indices)
            logger.warning(f"batch is not empty, but some samples are empty, skip {num_skip} samples.")

        batch = [batch[i] for i in nonempty_sample_indices]

        # NOTE(xiaoke) dynamic padding
        # inputs_ids List[List[int]]
        # attention_mask List[List[int]]
        num_regions_per_sample = [len(sample["input_ids"]) for sample in batch]
        num_minimum_regions_per_sample = min(num_regions_per_sample)

        is_batch_of_regions = all(x == num_minimum_regions_per_sample for x in num_regions_per_sample)
        # NOTE: if num_masks_per_sample is larger than all the numbers of regions in the batch, we then need to chunk with the minimum number of batches
        if not is_batch_of_regions:
            logger.warning(
                f"is_batch_of_regions is False due to num_minimum_regions_per_sample {num_minimum_regions_per_sample} < num_regions_per_sample {num_regions_per_sample}. "
                "Thus we chunk the regions with the minimum number of regions in the batch."
            )

        flat_input_ids = []
        flat_attention_mask = []
        for sample in batch:
            # NOTE(xiaoke): pop out the input_ids and attention_mask
            flat_input_ids.extend(sample.pop("input_ids"))
            flat_attention_mask.extend(sample.pop("attention_mask"))

        # NOTE(xiaoke): pad the input_ids and attention_mask
        # which are already truncated to `model_max_length`
        encoding_tokenizer = self.tokenizer.pad(
            dict(input_ids=flat_input_ids, attention_mask=flat_attention_mask),
            padding=True,
            return_tensors="pt",
        )
        # add labels, pad with -100 to ignore in loss computation
        encoding_tokenizer["labels"] = self.prepare_labels(encoding_tokenizer)

        for k, v in encoding_tokenizer.items():
            encoding_tokenizer_ = v.split(num_regions_per_sample)
            encoding_tokenizer_ = [x[:num_minimum_regions_per_sample] for x in encoding_tokenizer_]
            encoding_tokenizer[k] = torch.stack(encoding_tokenizer_)

        # process other fields, e.g., `input_boxes`, `metadata_*`, etc.
        encoding_else = {}
        for k, v in batch[0].items():
            if v is None:
                # NOTE: if the value is None, we set it to None and not batchfity it.
                encoding_else[k] = None
            elif isinstance(v, torch.Tensor):
                if k in REGION_KEYS:
                    # NOTE(xiaoke): it is possible for the number of regions to be different
                    # i.e., less than num_masks_per_sample during training.
                    # NOTE(xiaoke): we make sure that eval_batch_size=1
                    encoding_else[k] = torch.stack([sample[k][:num_minimum_regions_per_sample] for sample in batch])
                else:
                    encoding_else[k] = torch.stack([sample[k] for sample in batch])
            else:
                encoding_else[k] = [sample[k] for sample in batch]

        return {
            **encoding_tokenizer,
            **encoding_else,
        }

    def prepare_labels(self, encoding_tokenizer):
        label_mask = encoding_tokenizer["attention_mask"].bool()
        labels = encoding_tokenizer["input_ids"].clone()
        labels.masked_fill_(~label_mask, self.label_pad_token_id)
        return labels


class SCADataCollator(SamCaptionerDataCollator):
    def prepare_labels(self, encoding_tokenizer):
        labels = super().prepare_labels(encoding_tokenizer)
        # XXX(xiaoke): since we do not add the <BOS> token in both training and inference stage.
        # Therefore, we need to shift the labels to the right by one. However, this leads to the situation where labels have one more token than the input_ids,
        # and the max_length of the labels is larger by 1 than tokeinizer.model_max_length, which cause error in trainer eval when compute eval loss due to mismatched last dim during cross-batch-region paddding.
        # Our solution is to trim the input_ids and attention_mask by 1.
        # Check `src/data/transforms/base_transforms.py:SCADataTransform:process_tokens` for more details.
        return torch.nn.functional.pad(labels, (1, 0), value=self.label_pad_token_id)
