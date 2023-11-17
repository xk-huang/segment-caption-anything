import gc
import os
import tempfile
from typing import Optional, Tuple, Union, List, Dict, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel, AutoModelForCausalLM
from .configuration_sam_captioner import SAMCaptionerConfig
from ..sam.modeling_sam import SamVisionEncoderOutput, SamImageSegmentationOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.image_utils import ImageInput
from transformers.models.auto.processing_auto import AutoProcessor
import transformers
from torchvision.ops import masks_to_boxes
from PIL import Image
from ...data.transforms import UNUSED_KEYS_IN_GENERATE


logger = logging.get_logger(__name__)


@dataclass
class SAMCaptionerOutput(SamImageSegmentationOutput, CausalLMOutputWithPast):
    """_summary_

    Args:
        SamImageSegmentationOutput (_type_): _description_
        CausalLMOutputWithPast (_type_): _description_
    """


@dataclass
class SAMCaptionerGenerationOutput(SAMCaptionerOutput):
    """_summary_

    Args:
        sequences (torch.Tensor): (batch_size, num_masks, num_heads, num_tokens)
        patches: (List[List[List[Image.Image]]]): (batch_size, num_masks, num_heads)
        loss: (torch.Tensor): (1,)
        logits: (torch.Tensor): (batch_size, num_masks, num_heads, num_tokens, vocab_size)
    """

    sequences: torch.LongTensor = None
    patches: List[List[List[Image.Image]]] = None
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class SAMCaptionerModel(PreTrainedModel):
    config_class = SAMCaptionerConfig

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        sam: Optional[PreTrainedModel] = None,
        sam_processor: Optional[AutoProcessor] = None,
        captioner: Optional[PreTrainedModel] = None,
        captioner_processor: Optional[AutoProcessor] = None,
    ):
        if config is None and (sam is None or captioner is None):
            raise ValueError("Either a configuration or a model has to be provided")

        if config is None:
            config = SAMCaptionerConfig.from_sam_captioner_configs(sam.config, captioner.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")
        if sam is None:
            sam = AutoModel.from_config(config.sam)
        if captioner is None:
            captioner = AutoModelForCausalLM.from_config(config.captioner)

        # FIXME(xiaoke): initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        self.sam = sam
        self.captioner = captioner

        if self.sam.config.to_dict() != self.config.sam.to_dict():
            logger.warning(
                f"Config of the sam: {self.sam.__class__} is overwritten by shared sam config:" f" {self.config.sam}"
            )
        if self.captioner.config.to_dict() != self.config.captioner.to_dict():
            logger.warning(
                f"Config of the captioner: {self.captioner.__class__} is overwritten by shared captioner config:"
                f" {self.config.captioner}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.sam.config = self.config.sam
        self.captioner.config = self.config.captioner

        if sam_processor is None:
            if getattr(config.sam, "_name_or_path", None) is None:
                raise ValueError(f"sam_processor has to be provided if sam is not pretrained")
            sam_processor = AutoProcessor.from_pretrained(getattr(config.sam, "_name_or_path"))
        self.sam_processor = sam_processor

        if captioner_processor is None:
            if getattr(config.captioner, "_name_or_path", None) is None:
                raise ValueError(f"captioner_processor has to be provided if captioner is not pretrained")
            captioner_processor = AutoProcessor.from_pretrained(getattr(config.captioner, "_name_or_path"))
        self.captioner_processor = captioner_processor

        # Find generation config in language model
        def search_generation_config(obj, parent_key="base"):
            generation_configs = []
            for attr in dir(obj):
                if attr.startswith("_"):
                    continue
                elif attr == "generation_config" and getattr(obj, attr) is not None:
                    generation_configs.append((f"{parent_key}-{attr}", getattr(obj, attr)))
                elif isinstance(getattr(obj, attr), (nn.Module, PreTrainedModel)):
                    # skip self reference to avoid infinite recursion
                    if obj == getattr(obj, attr):
                        continue
                    generation_configs.extend(
                        search_generation_config(getattr(obj, attr), parent_key=f"{parent_key}-{attr}")
                    )
            return generation_configs

        generation_configs = search_generation_config(self.captioner, parent_key="captioner")
        if len(generation_configs) != 1:
            logger.warning(f"generation_configs: {generation_configs} has to be of length 1, we use the first one")
        generation_config = generation_configs[0][1]
        if generation_config is not None:
            self.generation_config = generation_config
            logger.info(f"generation_config: {generation_config} is used for `generate`")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_sam_captioner_pretrained(
        cls,
        sam_pretrained_model_name_or_path: str = None,
        captioner_pretrained_model_name_or_path: str = None,
        **kwargs,
    ) -> PreTrainedModel:
        sam_config = AutoConfig.from_pretrained(sam_pretrained_model_name_or_path, **kwargs)
        sam_architectures = sam_config.architectures
        if len(sam_architectures) != 1:
            logger.warning(f"sam_architectures: {sam_architectures} has to be of length 1")
        sam_architecture = sam_architectures[0]
        sam_module = getattr(transformers, sam_architecture)
        sam = sam_module.from_pretrained(sam_pretrained_model_name_or_path, **kwargs)

        captioner_config = AutoConfig.from_pretrained(captioner_pretrained_model_name_or_path, **kwargs)
        # NOTE(xiaoke): load architecture from config, or the model architecture will not be initialized correctly
        caption_architectures = captioner_config.architectures
        if len(caption_architectures) != 1:
            logger.warning(f"captioner_architectures: {caption_architectures} has to be of length 1")
        captioner_architecture = caption_architectures[0]
        captioner_module = getattr(transformers, captioner_architecture)
        captioner = captioner_module.from_pretrained(captioner_pretrained_model_name_or_path, **kwargs)

        # instantiate config with corresponding kwargs
        config = SAMCaptionerConfig.from_sam_captioner_configs(sam.config, captioner.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False

        sam_processor = AutoProcessor.from_pretrained(sam_pretrained_model_name_or_path, **kwargs)
        captioner_processor = AutoProcessor.from_pretrained(captioner_pretrained_model_name_or_path, **kwargs)

        return cls(
            sam=sam,
            captioner=captioner,
            sam_processor=sam_processor,
            captioner_processor=captioner_processor,
            config=config,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.forward("inference", *args, **kwargs)

    def forward(
        self,
        mode="train",
        images: Optional[List[ImageInput]] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.FloatTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=None,
        return_patches: Optional[bool] = None,
        # blip arguments
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        chunkified_forward_size: Optional[int] = None,
        # other arguments
        original_sizes: Optional[torch.LongTensor] = None,
        reshaped_input_sizes: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """_summary_

        Args:
            images (Optional[List[ImageInput]], optional): (batch_size, ). Defaults to None.
            pixel_values (Optional[torch.FloatTensor], optional): (batch_size, num_channel, h, w). Defaults to None.
            input_points (Optional[torch.FloatTensor], optional): (batch_size, num_patches, num_points_per_patch, 2). Defaults to None.
            input_labels (Optional[torch.LongTensor], optional): (batch_size, num_patches, num_points_per_patch). Defaults to None.
            input_boxes (Optional[torch.FloatTensor], optional): (batch_size, num_patches, 4). Defaults to None.
            input_masks (Optional[torch.LongTensor], optional): (batch_size, window_size, window_size). Defaults to None.
            image_embeddings (Optional[torch.FloatTensor], optional): (batch_size, output_channels, window_size, window_size). Defaults to None.
            multimask_output (bool, optional): _description_. Defaults to True.
            attention_similarity (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            target_embedding (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            output_attentions (Optional[bool], optional): _description_. Defaults to None.
            output_hidden_states (Optional[bool], optional): _description_. Defaults to None.
            return_dict (_type_, optional): _description_. Defaults to None.
            return_patches (Optional[bool], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            List[Dict[str, torch.Tensor]]: _description_
        """
        if images is None:
            raise ValueError("images has to be provided to crop to patches")
        if not isinstance(images[0], Image.Image):
            raise ValueError(f"images has to be of type List[Image.Image], got {type(images[0])}")

        sam_outputs: SamImageSegmentationOutput = self.sam(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
            image_embeddings=image_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        # iou_scores, (batch_size, num_masks, num_output_heads)
        # pred_masks, (batch_size, num_masks, num_output_heads, logits_height, logits_width)

        pred_masks = sam_outputs.pred_masks
        # (batch_size, num_masks, num_heads, logits_height, logits_width)

        if original_sizes is None:
            raise ValueError("original_sizes has to be provided")
        if reshaped_input_sizes is None:
            raise ValueError("reshaped_input_sizes has to be provided")
        batch_masks = self.sam_processor.post_process_masks(
            pred_masks, original_sizes, reshaped_input_sizes
        )  # List[Tensor(num_masks, 1 OR num_heads, logits_height, logits_width)], as image may have different sizes

        # FIXME(xiaoke): for loop is slow
        patches = []
        batch_size, num_masks, num_heads, *_ = pred_masks.shape
        for batch_idx, (image, masks) in enumerate(zip(images, batch_masks)):
            masks = masks.flatten(0, 1)
            for mask_idx, mask in enumerate(masks):
                if (~mask).all():
                    # NOTE(xiaoke): corner case, handle the case where the mask is all False.
                    logger.warning(
                        f"mask at ({batch_idx}, {mask_idx // num_heads}, {mask_idx % num_heads}) "
                        "is all False, we set (0, 0) = True to avoid error in `masks_to_boxes`"
                    )
                    # NOTE(xiaoke): corner case, mask size is equal
                    # to the size of input image, they should not be too small.
                    dummy_mask_max_edge = 2
                    # dummy_mask_max_edge = min(2, min(mask.shape))
                    # if dummy_mask_max_edge < 2:
                    #     raise ValueError("the patch is only 1 pixel, which may not be used by captioner like BLIP")
                    mask[[0, dummy_mask_max_edge], [0, dummy_mask_max_edge]] = True

            boxes = masks_to_boxes(masks)
            # NOTE(xiaoke): corner case, if the width of any edge of the patch is 1100,5,205,50
            # that patch may not be used by captioner like BLIP.
            # NOTE(xiaoke): PIL.Image.crop will pad the patch if the box is larger than the image.
            boxes[:, 2] = torch.maximum(boxes[:, 0] + 2, boxes[:, 2])
            boxes[:, 3] = torch.maximum(boxes[:, 1] + 2, boxes[:, 3])
            boxes = boxes.cpu().numpy()
            for box in boxes:
                patches.append(image.crop(box))
        del pred_masks
        del batch_masks

        # NOTE(xiaoke): The ModelOuput object cannot be modified in-place
        sam_captioner_output = dict(**sam_outputs)

        if chunkified_forward_size is None:
            logger.debug(f"chunkified_forward_size is not provided, we use 16 as default.")
            chunkified_forward_size = 16
        # NOTE(xiaoke): the captioner's processor can have either images & text arguement signature or text & images arguement signature
        captioner_inputs = self.captioner_processor(images=patches, return_tensors="pt").to(self.device)
        if mode == "train":
            # num_heads is either 1 or num_heads in SAM
            input_ids = input_ids.unsqueeze(-2).repeat_interleave(num_heads, dim=-2).flatten(0, 2)
            attention_mask = attention_mask.unsqueeze(-2).repeat_interleave(num_heads, dim=-2).flatten(0, 2)
            labels = labels.unsqueeze(-2).repeat_interleave(num_heads, dim=-2).flatten(0, 2)
            # compute loss for all three heads
            # NOTE(xiaoke): inputs are:
            #   - captioner_inputs:
            #       - pixel_values torch.Size([batch_size*num_masks*num_heads, 3, h, w])
            #   - input_ids: torch.Size([batch_size*num_masks*num_heads, PADDED_length])
            #   - attention_mask: torch.Size([batch_size*num_masks*num_heads, PADDED_length])
            #   - labels: torch.Size([batch_size*num_masks*num_heads, PADDED_length])
            captioner_outputs = self._chunkify_forward(
                self.captioner,
                chunkified_forward_size,
                **captioner_inputs,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            sam_captioner_output["loss"] = captioner_outputs.loss
            # FIXME(xiaoke): BlipForConditionalGenerationModelOutput has `decoder_logits`
            # while Blip2ForConditionalGenerationModelOutput has `logits`
            # So we do not return logits so far.
        else:
            for key in list(kwargs.keys()):
                # remove the keys that are not used by captioner.generate.
                # Or it will raise error in `transformers/generation/utils.py:_validate_model_kwargs`
                # they are used for post-processing
                if key in UNUSED_KEYS_IN_GENERATE:
                    kwargs.pop(key)
            # NOTE(xiaoke): inputs are:
            #   - captioner_inputs:
            #       - pixel_values torch.Size([batch_size*num_masks*num_heads, 3, h, w])
            #   - kwargs: {'max_length': 20, 'num_beams': 1, 'synced_gpus': False}
            captioner_generate_ids: torch.LongTensor = self._chunkify_forward(
                self.captioner.generate, chunkified_forward_size, **captioner_inputs, **kwargs
            )
            # TODO: fix this hack and figure out what is the output format for each captioner.
            try:
                num_tokens = captioner_generate_ids.shape[-1]
            except Exception as e:
                captioner_generate_ids = captioner_generate_ids["sequences"]
                num_tokens = captioner_generate_ids.shape[-1]
            captioner_generate_ids = captioner_generate_ids.view(batch_size, num_masks, num_heads, num_tokens)
            sam_captioner_output["sequences"] = captioner_generate_ids

        # return patch for visulization
        if return_patches is None:
            return_patches = False
        if return_patches is True:
            # FIXME(xiaoke): a dirty hack to not convert PIL.Image into np array if all the dims are the same
            returned_patches = (
                np.array(patches + [""], dtype=object)[:-1].reshape(batch_size, num_masks, num_heads).tolist()
            )
            sam_captioner_output["patches"] = returned_patches

        return SAMCaptionerGenerationOutput(**sam_captioner_output)

    def _chunkify_forward(self, func, regional_chunk_size, **kwargs):
        # NOTE(xiaoke): process (sample_size, ...) inputs and ouputs by chunks
        # Only chunkify the first dimension of the inputs and outputs
        chunked_output_list = []
        for chunked_kwargs in self._chunkify_inputs_generator(regional_chunk_size, **kwargs):
            chunked_outputs = func(**chunked_kwargs)
            chunked_output_list.append(chunked_outputs)
        return self._concat_chunked_outputs(chunked_output_list)

    def _chunkify_inputs_generator(self, regional_chunk_size, **kwargs):
        chunkified_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        unchunkified_kwargs = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
        chunkified_keys = list(chunkified_kwargs.keys())
        chunked_kwarg_shapes = [len(chunkified_kwargs[k]) for k in chunkified_keys]
        if not all([s == chunked_kwarg_shapes[0] for s in chunked_kwarg_shapes]):
            raise ValueError(
                f"all the first dimension of the inputs have to be the same, but they are {chunked_kwarg_shapes} for {chunkified_keys}"
            )
        num_samples = len(chunkified_kwargs[chunkified_keys[0]])
        logger.debug(f"num_samples: {num_samples}")
        for start_idx in range(0, num_samples, regional_chunk_size):
            end_idx = start_idx + regional_chunk_size
            chunked_chunkified_kwargs = {k: v[start_idx:end_idx] for k, v in chunkified_kwargs.items()}
            yield {**chunked_chunkified_kwargs, **unchunkified_kwargs}

    def _concat_chunked_outputs(self, chunked_outputs: Union[Mapping, Sequence, torch.Tensor]):
        output_type = type(chunked_outputs[0])
        if isinstance(chunked_outputs[0], Sequence):
            return output_type([self._concat_chunked_outputs(col_outputs) for col_outputs in zip(*chunked_outputs)])
        elif isinstance(chunked_outputs[0], Mapping) or hasattr(output_type, "items"):
            return output_type(
                {
                    k: self._concat_chunked_outputs([col_outputs[k] for col_outputs in chunked_outputs])
                    for k in chunked_outputs[0].keys()
                }
            )
        elif not isinstance(chunked_outputs[0], torch.Tensor):
            raise ValueError(f"output_type: {output_type} has to be of type Sequence, Mapping")

        output_shapes = [output.shape for output in chunked_outputs]
        if len(output_shapes[0]) > 1 and not all([s[1] == output_shapes[0][1] for s in output_shapes]):
            logger.debug(
                f"output_shapes: {output_shapes} are not all the same, "
                "The only situation is that it is the output of a language model"
                "We pad the second dim assuming a shape of (batch_size, num_tokens, ...)"
            )
            max_num_tokens = max([s[1] for s in output_shapes])
            num_dims = len(output_shapes[0])

            # NOTE(xiaoke): copy from `transformers/generation/utils.py:GenerationMixin:generate`
            generation_config = self.generation_config
            if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
                eos_token_id = generation_config.eos_token_id
                if isinstance(eos_token_id, list):
                    eos_token_id = eos_token_id[0]
                logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
                generation_config.pad_token_id = eos_token_id
            chunked_outputs = [
                torch.nn.functional.pad(
                    output,
                    [0] * (num_dims - 2) * 2 + [0, max_num_tokens - output.shape[1]],
                    mode="constant",
                    value=generation_config.pad_token_id,
                )
                for output in chunked_outputs
            ]
        if len(output_shapes[0]) == 0:
            # NOTE(xiaoke): corner case, when the output is a scalar, e.g. loss.
            # NOTE(xiaoke): if directly use `output_type(chunked_outputs)`, it will raise error
            # as it converts a list of cuda tensor into a cpu tensor.
            return output_type(torch.stack(chunked_outputs)).mean()
        return output_type(torch.cat(chunked_outputs, dim=0))
