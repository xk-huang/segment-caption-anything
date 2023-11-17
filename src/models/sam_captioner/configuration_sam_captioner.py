from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto.configuration_auto import AutoConfig
import copy

logger = logging.get_logger(__name__)


class SAMCaptionerConfig(PretrainedConfig):
    model_type = "sam-captioner"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        sam_config = kwargs.pop("sam", None)
        sam_model_type = sam_config.pop("model_type", None)
        captioner_config = kwargs.pop("captioner", None)
        captioner_model_type = captioner_config.pop("model_type", None)

        self.sam = AutoConfig.for_model(sam_model_type, **sam_config)
        self.captioner = AutoConfig.for_model(captioner_model_type, **captioner_config)

    @classmethod
    def from_sam_captioner_configs(
        cls, sam_config: PretrainedConfig, captioner_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        return cls(sam=sam_config.to_dict(), captioner=captioner_config.to_dict(), **kwargs)
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["sam"] = self.sam.to_dict()
        output["captioner"] = self.captioner.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
