from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
from ..sam.configuration_sam import (
    SamPromptEncoderConfig,
    SamVisionConfig,
    SAM_PRETRAINED_CONFIG_ARCHIVE_MAP,
    SamConfig,
)
from transformers.models.auto import CONFIG_MAPPING
import copy
from typing import Optional


logger = logging.get_logger(__name__)


class ScaMaskCaptionDecoderConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=256,
        hidden_act="relu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        layer_norm_eps=1e-6,
        # NOTE(xiaoke): for captioning
        # NOTE: Remember to change `from_sam_text_configs` as well!
        additional_num_hidden_layers: int = 2,
        num_caption_tokens: int = 1,
        num_caption_heads: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_dim = mlp_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.layer_norm_eps = layer_norm_eps
        # NOTE(xiaoke): additional_num_hidden_layers used in transformers layers to further fuse features
        self.additional_num_hidden_layers = additional_num_hidden_layers
        self.num_caption_tokens = num_caption_tokens
        self.num_caption_heads = num_caption_heads


class ScaConfig(PretrainedConfig):
    model_type = "sca"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        prompt_encoder_config=None,
        mask_caption_decoder_config=None,
        text_config=None,
        initializer_range=0.02,
        # NOTE: for recoginition pretrain
        num_task_tokens: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vision_config = vision_config if vision_config is not None else {}
        prompt_encoder_config = prompt_encoder_config if prompt_encoder_config is not None else {}
        mask_caption_decoder_config = mask_caption_decoder_config if mask_caption_decoder_config is not None else {}
        text_config = text_config if text_config is not None else {}

        if isinstance(vision_config, SamVisionConfig):
            self.vision = vision_config.to_dict()
        if isinstance(prompt_encoder_config, SamPromptEncoderConfig):
            self.prompt_encoder = prompt_encoder_config.to_dict()
        if isinstance(mask_caption_decoder_config, ScaMaskCaptionDecoderConfig):
            self.mask_caption_decoder_config = mask_caption_decoder_config.to_dict()

        text_model_type = text_config["model_type"] if "model_type" in text_config else "gpt2"
        # NOTE(xiaoke): use_decoder_only_language_model only return the model class like GPT2, rather the task model class
        # like GPT2forCausalLM. We need the task model class to load the pretrained weights for the task.
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

        self.vision_config = SamVisionConfig(**vision_config)
        self.prompt_encoder_config = SamPromptEncoderConfig(**prompt_encoder_config)
        self.mask_caption_decoder_config = ScaMaskCaptionDecoderConfig(**mask_caption_decoder_config)
        self.initializer_range = initializer_range

        self.num_task_tokens = num_task_tokens

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["prompt_encoder_config"] = self.prompt_encoder_config.to_dict()
        output["mask_caption_decoder_config"] = self.mask_caption_decoder_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output

    @classmethod
    def from_sam_text_configs(
        cls,
        sam_config: SamConfig,
        text_config: Optional[PretrainedConfig] = None,
        additional_num_hidden_layers: Optional[int] = None,
        num_caption_tokens: Optional[int] = None,
        num_task_tokens: Optional[int] = None,
        num_caption_heads: Optional[int] = None,
        vl_projector_type: Optional[str] = None,
        vl_projector_norm_type: Optional[str] = None,
        **kwargs,
    ):
        if additional_num_hidden_layers is None:
            logger.warning("additional_num_hidden_layers is not set, using default value: 2. Make sure it is correct!")
            additional_num_hidden_layers = 2
        if num_caption_tokens is None:
            logger.warning("num_caption_tokens is not set, using default value: 1. Make sure it is correct!")
            num_caption_tokens: int = 1
        if num_task_tokens is None:
            logger.warning("num_task_tokens is not set, using default value: 6. Make sure it is correct!")
            num_task_tokens = 6
        if num_caption_heads is None:
            logger.warning("num_caption_heads is not set, using default value: 1. Make sure it is correct!")
            num_caption_heads = 1
        if vl_projector_type is None:
            logger.warning("vl_projector_type is not set, using default value: linear. Make sure it is correct!")
            vl_projector_type = "linear"
        if vl_projector_norm_type is None:
            logger.warning("vl_projector_norm_type is not set, using default value: none. Make sure it is correct!")
            vl_projector_norm_type = "none"

        return cls(
            vision_config=sam_config.vision_config.to_dict(),
            prompt_encoder_config=sam_config.prompt_encoder_config.to_dict(),
            mask_caption_decoder_config={
                **sam_config.mask_decoder_config.to_dict(),
                "additional_num_hidden_layers": additional_num_hidden_layers,
                "num_caption_tokens": num_caption_tokens,
                "num_caption_heads": num_caption_heads,
            },
            text_config=text_config.to_dict() if text_config is not None else None,
            num_task_tokens=num_task_tokens,
            vl_projector_type=vl_projector_type,
            vl_projector_norm_type=vl_projector_norm_type,
            **kwargs,
        )


class ScaTimmConfig(ScaConfig):
    model_type = "sca_timm"
    is_composition = True

    def __init__(
        self,
        timm_vision_name=None,
        vision_config=None,
        prompt_encoder_config=None,
        mask_caption_decoder_config=None,
        text_config=None,
        initializer_range=0.02,
        # NOTE: for recoginition pretrain
        num_task_tokens: int = 6,
        **kwargs,
    ):
        super().__init__(
            vision_config=vision_config,
            prompt_encoder_config=prompt_encoder_config,
            mask_caption_decoder_config=mask_caption_decoder_config,
            text_config=text_config,
            initializer_range=initializer_range,
            num_task_tokens=num_task_tokens,
            **kwargs,
        )
        timm_vision_name = timm_vision_name if timm_vision_name is not None else "vit_base_patch16_clip_224.openai"
        if isinstance(timm_vision_name, str):
            self.timm_vision_name = timm_vision_name

    def to_dict(self):
        output = super().to_dict()
        output["timm_vision_name"] = self.timm_vision_name
        return output

    @classmethod
    def from_sam_timm_text_configs(
        cls,
        timm_vision_name: str,
        sam_config: SamConfig,
        text_config: Optional[PretrainedConfig] = None,
        additional_num_hidden_layers: Optional[int] = None,
        num_caption_tokens: Optional[int] = None,
        num_task_tokens: Optional[int] = None,
        num_caption_heads: Optional[int] = None,
        vl_projector_type: Optional[str] = None,
        vl_projector_norm_type: Optional[str] = None,
        **kwargs,
    ):
        if additional_num_hidden_layers is None:
            logger.warning("additional_num_hidden_layers is not set, using default value: 2. Make sure it is correct!")
            additional_num_hidden_layers = 2
        if num_caption_tokens is None:
            logger.warning("num_caption_tokens is not set, using default value: 1. Make sure it is correct!")
            num_caption_tokens: int = 1
        if num_task_tokens is None:
            logger.warning("num_task_tokens is not set, using default value: 6. Make sure it is correct!")
            num_task_tokens = 6
        if num_caption_heads is None:
            logger.warning("num_caption_heads is not set, using default value: 1. Make sure it is correct!")
            num_caption_heads = 1
        if vl_projector_type is None:
            logger.warning("vl_projector_type is not set, using default value: linear. Make sure it is correct!")
            vl_projector_type = "linear"
        if vl_projector_norm_type is None:
            logger.warning("vl_projector_norm_type is not set, using default value: none. Make sure it is correct!")
            vl_projector_norm_type = "none"

        return cls(
            timm_vision_name=timm_vision_name,
            vision_config=sam_config.vision_config.to_dict(),
            prompt_encoder_config=sam_config.prompt_encoder_config.to_dict(),
            mask_caption_decoder_config={
                **sam_config.mask_decoder_config.to_dict(),
                "additional_num_hidden_layers": additional_num_hidden_layers,
                "num_caption_tokens": num_caption_tokens,
                "num_caption_heads": num_caption_heads,
            },
            text_config=text_config.to_dict() if text_config is not None else None,
            num_task_tokens=num_task_tokens,
            vl_projector_type=vl_projector_type,
            vl_projector_norm_type=vl_projector_norm_type,
            **kwargs,
        )
