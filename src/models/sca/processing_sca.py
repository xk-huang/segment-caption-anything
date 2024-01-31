"""
Copy from src.models.sam_captioner.processing_sam_captioner.py
"""
from ..sam.processing_sam import SamProcessor
from transformers.models.blip import BlipProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.image_utils import make_list_of_images
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from typing import List, Optional, Union
from ..sam_captioner.processing_sam_captioner import SAMCaptionerProcessor
import logging

logger = logging.getLogger(__name__)


class ScaProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, sam_processor, tokenizer):
        super().__init__(tokenizer)
        self.sam_processor: SamProcessor = sam_processor

    def __call__(
        self,
        # from ../sam/processing_sam.py
        images=None,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        original_sizes=None,
        # from transformers.models.blip.processing_blip.py
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors=None,
        **kwargs,
    ):
        if images is None and original_sizes is None:
            raise ValueError(f"images and original_sizes cannot both be None.")

        if images is not None:
            input_encoding = self.sam_processor(
                images=images,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                return_tensors=return_tensors,
                **kwargs,
            )
            images = make_list_of_images(images)
            input_encoding["images"] = make_list_of_images(images)
        else:
            input_encoding = self.sam_processor.process_prompts(
                original_sizes=original_sizes,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                return_tensors=return_tensors,
            )

        if text is not None:
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            text_encoding = {}
        input_encoding.update(text_encoding)

        return input_encoding

    def post_process_masks(self, *args, **kwargs):
        return self.sam_processor.post_process_masks(*args, **kwargs)

    @classmethod
    def from_sam_text_pretrained(cls, sam_pretrained_model_name_or_path, text_pretrained_model_name_or_path, **kwargs):
        sam_processor = SamProcessor.from_pretrained(sam_pretrained_model_name_or_path, **kwargs)
        # NOTE: To be compatible with OpenLLAMA which uses the slow tokenizer to avoid a bug.
        # Ref: https://github.com/openlm-research/open_llama#loading-the-weights-with-hugging-face-transformers
        if "open_llama" in text_pretrained_model_name_or_path:
            logger.warning(f"Using slow tokenizer for {text_pretrained_model_name_or_path}.")
            use_fast = False
        else:
            use_fast = True
        captioner_processor = AutoTokenizer.from_pretrained(
            text_pretrained_model_name_or_path, use_fast=use_fast, **kwargs
        )
        return cls(sam_processor, captioner_processor)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        sam_processor_input_names = self.sam_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + sam_processor_input_names))
