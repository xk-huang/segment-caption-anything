"""
Copied from `.modeling_sca` commit dec0744 and modified by Xiaoke.
"""
import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..sam.configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
from ..sam.modeling_sam import (
    SAM_PRETRAINED_MODEL_ARCHIVE_LIST,
    SamVisionEncoderOutput,
    SamImageSegmentationOutput,
    SamPreTrainedModel,
    SamPositionalEmbedding,
    SamPromptEncoder,
    SamVisionEncoder,
    SamTwoWayTransformer,
    SamLayerNorm,
    SamFeedForward,
)
from .configuration_sca import ScaConfig, ScaMaskCaptionDecoderConfig
from transformers.models.auto import AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import copy
import transformers
from ...data.transforms import UNUSED_KEYS_IN_GENERATE

logger = logging.get_logger(__name__)


@dataclass
class ScaForConditionalGnerationModelOutput(ModelOutput):
    """_summary_

    Args:
        ModelOutput (_type_): _description_

    Returns:
        _type_: _description_
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    segmentation_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None
    # For generate
    sequences: Optional[Tuple[torch.LongTensor]] = None
    iou_scores: Optional[torch.FloatTensor] = None
    pred_masks: Optional[torch.FloatTensor] = None
    # For debuging
    query_logits: Optional[torch.FloatTensor] = None
    projected_query_logits: Optional[torch.FloatTensor] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "segmentation_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# Copied from ..sam.modeling_sam.SamMaskDecoder
class ScaMaskCaptionDirectDecodingDecoder(nn.Module):
    def __init__(self, config: ScaMaskCaptionDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

        # NOTE(xiaoke): add additional fusion transformer layers
        addtional_transformer_config = copy.deepcopy(config)
        addtional_transformer_config.num_hidden_layers = addtional_transformer_config.additional_num_hidden_layers
        del addtional_transformer_config.additional_num_hidden_layers
        self.additional_transformer = SamTwoWayTransformer(addtional_transformer_config)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        # NOTE(xiaoke): Modified. We need to outputs one more tensor: `query_outputs` for captioning
        caption_tokens = mask_tokens_out[:, :, mask_slice]
        num_caption_tokens = len(caption_tokens)
        point_embeddings = torch.cat([caption_tokens, point_embeddings], dim=-2)

        point_embedding, image_embeddings, attentions = self.additional_transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        caption_tokens_out = point_embedding[:, :, :num_caption_tokens, :]

        outputs = (masks, iou_pred, caption_tokens_out)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs
        # low_res_masks, iou_predictions, query_outputs, mask_decoder_attentions
        # low_res_masks: (batch_size, num_masks, num_output_heads, logits_height, logits_width)
        # iou_predictions: (batch_size, num_masks, num_output_heads)
        # query_outputs: (batch_size, num_masks, num_output_heads, hidden_size)


class ScaDirectDecodingPretrainedModel(SamPreTrainedModel):
    config_class = ScaConfig
    base_model_prefix = "sca_direct_decoding"
    main_input_name = "pixel_values"


class ScaDirectDecodingModel(ScaDirectDecodingPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config: ScaConfig, language_model: nn.Module = None):
        super().__init__(config)
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        # NOTE(xiaoke): Modified. We need to outputs one more tensor: `query_outputs` for captioning
        # Thus its real name is `mask_caption_decoder`, but we keep the name `mask_decoder` for loading SAM weights.
        self.mask_decoder = ScaMaskCaptionDirectDecodingDecoder(config.mask_caption_decoder_config)

        self.language_project = nn.Linear(
            config.mask_caption_decoder_config.hidden_size, config.text_config.hidden_size
        )
        if language_model is None:
            if config.use_decoder_only_language_model:
                language_model = AutoModelForCausalLM.from_config(config.text_config)
            else:
                raise ValueError("Only decoder only language model is supported.")
        self.language_model = language_model

        if config.text_config != self.language_model.config:
            text_config_dict = config.text_config.to_dict()
            language_model_config_dict = self.language_model.config.to_dict()
            all_keys = set(text_config_dict.keys()) | set(language_model_config_dict.keys())
            diff_kv = {}
            for k in all_keys:
                if k not in text_config_dict and k in language_model_config_dict:
                    diff_kv[k] = (None, language_model_config_dict[k])
                elif k in text_config_dict and k not in language_model_config_dict:
                    diff_kv[k] = (text_config_dict[k], None)
                else:
                    if text_config_dict[k] != language_model_config_dict[k]:
                        diff_kv[k] = (text_config_dict[k], language_model_config_dict[k])
            logger.warning(
                "The text config is different from the original config and the language model config. The following keys have different "
                "values: {}".format(diff_kv)
            )
        # NOTE: To support gradient checkpoint for LM: https://github.com/huggingface/transformers/pull/19990/files
        self.supports_gradient_checkpointing = True

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

        generation_configs = search_generation_config(self.language_model, parent_key="captioner")
        if len(generation_configs) != 1:
            logger.warning(f"generation_configs: {generation_configs} has to be of length 1, we use the first one")
        generation_config = generation_configs[0][1]
        if generation_config is not None:
            self.generation_config = generation_config
            logger.info(f"generation_config: {generation_config} is used for `generate`")

        self.config_parameters()
        self.post_init()

    # Copied from ..sam.modeling_sam.SamModel
    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    # NOTE(xiaoke). Modified from ..sam.modeling_sam.SamModel
    def forward(
        self,
        mode="train",
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict=None,
        # segmentation arguments
        mask_labels: Optional[torch.LongTensor] = None,
        # language model arguments
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # legacy arguments for catching the inputs for sam captioner
        images=None,
        original_sizes=None,
        reshaped_input_sizes=None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                vision_attentions = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        # NOTE(xiaoke): Modified. We need to outputs one more tensor: `query_outputs`
        low_res_masks, iou_predictions, query_outputs, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        # low_res_masks: (batch_size, num_masks, num_output_heads, logits_height, logits_width)
        # iou_predictions: (batch_size, num_masks, num_output_heads)
        # query_outputs: (batch_size, num_masks, num_output_heads, hidden_size)
        batch_size, num_masks, num_output_heads, hidden_size = query_outputs.shape
        # NOTE(xiaoke): We use `expand` instead of `repeat` to avoid copying the tensor.
        # So now we need to `reshape` the tensor to the original shape due to the mismatched stride.
        query_outputs = query_outputs.reshape(
            -1, 1, hidden_size
        )  # (batch_size * num_masks * num_output_heads, 1, hidden_size)

        language_model_inputs = self.language_project(
            query_outputs
        )  # (batch_size * num_masks * num_output_heads, 1, hidden_size)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )  # (batch_size * num_masks * num_output_heads, 1)

        # NOTE(xiaoke): Handle the edge case. If in train mode, and one of the input_ids and attention_mask is None, we should set the labels to None explicitly.
        if mode == "train" and (input_ids is None or attention_mask is None):
            logger.info(
                "In train mode, and one of the input_ids and attention_mask is None. Set them and labels to None."
            )
            input_ids = None
            attention_mask = None
            labels = None

        if mode == "train" and (input_ids is not None and attention_mask is not None):
            # input_ids: (batch_size, num_masks, PADDED_length)
            # attention_mask: (batch_size, num_masks, PADDED_length)
            # NOTE(xiaoke): Copy from ..sam_captioner.modeling_sam_captioner.SamCaptionerModel
            input_ids = input_ids.unsqueeze(-2).repeat_interleave(num_output_heads, dim=-2).flatten(0, 2)
            attention_mask = (
                attention_mask.unsqueeze(-2).repeat_interleave(num_output_heads, dim=-2).flatten(0, 2)
            )  # (batch_size * num_masks * num_output_heads, PADDED_length)

            # TODO(xiaoke): Now we repeat the labels num_output_heads times. Is this correct?
            # Shall we follow SAM to backpropagate the loss for the head with the lowest IoU?
            if labels is not None:
                labels = labels.unsqueeze(-2).repeat_interleave(num_output_heads, dim=-2).flatten(0, 2)

            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            expected_device = language_model_attention_mask.device
            attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        else:
            inputs_embeds = language_model_inputs
            attention_mask = language_model_attention_mask

        if self.config.use_decoder_only_language_model:
            if mode == "train":
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                logits = outputs.logits if return_dict else outputs[0]
                loss = None
                # we compute the loss here since we need to take into account the sequence length of the query embeds
                if labels is not None:
                    # TODO(xiaoke): Now we repeat the labels num_output_heads times. Is this correct?
                    # Shall we follow SAM to backpropagate the loss for the head with the lowest IoU?
                    labels = labels.to(logits.device)
                    logits = logits[:, -labels.size(1) :, :]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(logits.device)

                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss(reduction="mean")

                    loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
            else:
                for key in list(kwargs.keys()):
                    # remove the keys that are not used by captioner.generate.
                    # Or it will raise error in `transformers/generation/utils.py:_validate_model_kwargs`
                    # they are used for post-processing
                    if key in UNUSED_KEYS_IN_GENERATE:
                        kwargs.pop(key)
                language_model_generate_ids = self.language_model.generate(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
                )
                sam_output = SamImageSegmentationOutput(iou_scores=iou_predictions, pred_masks=low_res_masks)
                language_model_generate_ids = language_model_generate_ids.view(
                    batch_size, num_masks, num_output_heads, -1
                )
                query_outputs = query_outputs.view(batch_size, num_masks, num_output_heads, 1, -1)
                language_model_inputs = language_model_inputs.view(batch_size, num_masks, num_output_heads, 1, -1)
                return language_model_generate_ids, sam_output, query_outputs, language_model_inputs
        else:
            raise ValueError("Only decoder only language model is supported.")

        if not return_dict:
            sam_output = (iou_predictions, low_res_masks)
            if output_hidden_states:
                sam_output = sam_output + (vision_hidden_states,)

            if output_attentions:
                sam_output = sam_output + (vision_attentions, mask_decoder_attentions)
            output = (loss, logits) + sam_output + outputs + (query_outputs, language_model_inputs)
            return output

        sam_output = SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
        )
        return ScaForConditionalGnerationModelOutput(
            loss=loss,
            logits=logits,
            segmentation_outputs=sam_output,
            language_model_outputs=outputs,
            query_logits=query_outputs,
            projected_query_logits=language_model_inputs,
        )

    @classmethod
    def from_sam_text_pretrained(
        cls,
        sam_pretrained_model_name_or_path: str = None,
        text_pretrained_model_name_or_path: str = None,
        additional_num_hidden_layers: int = 2,
        **kwargs,
    ):
        sam_config = transformers.AutoConfig.from_pretrained(sam_pretrained_model_name_or_path, **kwargs)
        sam_architectures = sam_config.architectures
        if len(sam_architectures) != 1:
            logger.warning(f"sam_architectures: {sam_architectures} has to be of length 1")
        text_config = transformers.AutoConfig.from_pretrained(text_pretrained_model_name_or_path, **kwargs)
        config = ScaConfig.from_sam_text_configs(
            sam_config=sam_config,
            text_config=text_config,
            additional_num_hidden_layers=additional_num_hidden_layers,
            **kwargs,
        )
        language_model = AutoModelForCausalLM.from_pretrained(text_pretrained_model_name_or_path, **kwargs)
        sca_model = cls.from_pretrained(
            sam_pretrained_model_name_or_path, config=config, language_model=language_model, **kwargs
        )
        # NOTE(xiaoke): Validate the unloaded weights in the model by calling
        # `set([".".join(i.split(".")[0:2]) for i in unloaded_weights])`
        # There should be no weights left in the pretrained weights that are unloaded.
        return sca_model

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        language_model_generate_ids, sam_output, query_outputs, language_model_inputs = self.forward(
            "inference", *args, **kwargs
        )
        return ScaForConditionalGnerationModelOutput(
            sequences=language_model_generate_ids,
            segmentation_outputs=sam_output,
            query_logits=query_outputs,
            projected_query_logits=language_model_inputs,
            iou_scores=sam_output.iou_scores,
            pred_masks=sam_output.pred_masks,
        )

    def config_parameters(self):
        # NOTE(xiaoke): By default we freeze all the parameters in the config.
        # HF transformers trainer use requires_grad=True to filter out the parameters that need to be optimized.
        for param in self.parameters():
            param.requires_grad = False

        # Turn on the parameters that need to be optimized.
        TO_BE_OPTIMIZED = [
            self.mask_decoder.additional_transformer,
            self.language_project,
        ]
        for module in TO_BE_OPTIMIZED:
            for param in module.parameters():
                param.requires_grad = True

    # NOTE: To support gradient checkpoint for LM: https://github.com/huggingface/transformers/pull/19990/files
    def _set_gradient_checkpointing(self, module, value=False):
        # NOTE: Most language models in HF supprots gradient checkpointing
        # e.g., OpenLLAMA: https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/models/deprecated/open_llama/modeling_open_llama.py#L464C9-L464C36
        # gpt2: https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/models/gpt2/modeling_gpt2.py#L483C9-L483C36
        self.language_model._set_gradient_checkpointing(module, value=value)

        # NOTE: SAM vision encoder supports gradient checkponit
        # https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/models/sam/modeling_sam.py#L1012C14-L1012C37
        self.vision_encoder.gradient_checkpointing = value
