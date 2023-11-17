import logging
import os
import os.path as osp
import socket
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Any, Tuple, Dict

import datasets
import torch  # noqa
import transformers
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from transformers import Seq2SeqTrainingArguments, TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class SCATrainingArguments(TrainingArguments):
    report_to: Any = field(
        default="none"
    )  # THIS MUST BE NONE. Use wandb args to control logging. Otherwise, the logs are not controllable.
    remove_unused_columns: bool = field(default=False)

    # the eval batch size must be 1, since we cannot batchify
    # different number of masks per sample during eval
    per_device_eval_batch_size: int = field(default=1)

    # use manually constructed `labels`; without using `label` or `label_ids`
    label_names: List[str] = field(default_factory=lambda: ["labels"])

    # to freely generete captions without conditioning on the gt captions
    predict_with_generate: bool = field(default=True)

    # Set log_level to `info`. By default, it is `warning`.
    # debug - 10; info - 20; warning - 30; error - 40; critical - 50;
    # by default, it is `passive` which is 30.
    log_level: str = field(default="info")

    # NOTE(xiaoke): here list the custom arguments
    num_masks_per_sample: Optional[int] = field(default=None)
    # https://huggingface.co/docs/transformers/run_scripts#test-a-script
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)

    # external log dir, used in amulet
    output_log_dir: Optional[str] = field(default=None)

    # inference and save the generated captions
    do_inference: bool = field(default=False)

    # Fist evalute before training, from Keras
    evaluate_before_train: bool = field(default=False)

    # Config the trainable parameters
    trainable_params: Optional[List[str]] = field(default=None)
    custom_param_lrs: Dict[str, float] = field(
        default_factory=lambda: dict(),
        metadata={
            "help": "custom param lrs, prefix: lr, e.g., language_model, prefix: lr, e.g., +training.custom_param_lrs='{language_model:0.1}'"
        },
    )

    # Evaluate with metric computation beyond only loss
    compute_metrics: Optional[bool] = field(default=None)

    # Apply large-scale jittering and random flip augmentations for training
    data_transforms: Optional[Any] = field(default=None)

    # Save strategies
    # NOTE: by default, we save two checkpoint, one for best, the other for last
    # ref: https://github.com/huggingface/transformers/issues/19041#issuecomment-1248056494
    load_best_model_at_end: bool = field(default=True)
    # NOTE: you may also need to change: metric_for_best_model
    save_total_limit: int = field(default=2)
    save_save_strategy: str = field(default="steps")
    evaluation_strategy: str = field(default="steps")

    # NOTE: chunk inference to reduce memory usage
    generate_chunk_size: Optional[int] = field(default=None)

    _run_post_init: bool = field(default=False)

    def __post_init__(self):
        # Don't run post-init until ready to convert to TrainingArgs
        # to avoid `_n_gpu` which is not exists in `Trainer` arguments
        # and type check by OmegaConf
        if self.report_to != "none":
            raise ValueError(f"report_to must be None, got {self.report_to}")
        if self.label_smoothing_factor != 0:
            raise ValueError(
                f"label_smoothing_factor must be 0 as the first output of the model is not language model logits, got {self.label_smoothing_factor}"
            )
        if self._run_post_init:
            if self.per_device_eval_batch_size != 1:
                raise ValueError(
                    "per_device_eval_batch_size must be 1, "
                    "since we cannot batchify different "
                    "number of masks per sample during eval."
                )
            super().__post_init__()


@dataclass
class _Seq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    # OmegaConf doesn't support Union, so we need to use Any
    # version 4.30.2
    generation_config: Any
    # version 4.32.0
    debug: Any
    sharded_ddp: Any
    fsdp: Any


@dataclass
class SCASeq2SeqTrainingArguments(SCATrainingArguments, _Seq2SeqTrainingArguments):
    pass


@dataclass
class ModelArguments:
    model_max_length: int = field(default=20)
    cache_dir: str = field(default=".model.cache")


@dataclass
class SAMCaptionerModelArguments(ModelArguments):
    sam_model_name_or_path: str = field(default="facebook/sam-vit-huge")
    captioner_model_name_or_path: str = field(default="Salesforce/blip-image-captioning-base")


@dataclass
class SCAModelBaseArguments(ModelArguments):
    model_name_or_path: Optional[str] = field(default=None)
    sam_model_name_or_path: str = field(default="facebook/sam-vit-huge")
    lm_head_model_name_or_path: str = field(default="gpt2")
    additional_num_hidden_layers: int = field(default=2)


@dataclass
class SCAModelArguments(SCAModelBaseArguments):
    num_caption_tokens: int = field(default=1)


@dataclass
class SCADirectDecodingModelArguments(SCAModelBaseArguments):
    pass


@dataclass
class SCAMultitaskModelArguments(SCAModelBaseArguments):
    num_caption_tokens: int = field(default=1)
    num_task_tokens: int = field(default=6)


@dataclass
class ScaMultitaskV2ModelArguments(SCAModelBaseArguments):
    num_caption_tokens: int = field(default=1)
    num_task_tokens: int = field(default=6)
    num_caption_heads: int = field(default=1)


@dataclass
class SCAMultitaskSplitMixerModelArguments(SCAModelBaseArguments):
    num_caption_tokens: int = field(default=1)
    num_task_tokens: int = field(default=6)
    num_caption_heads: int = field(default=1)


@dataclass
class SCADirectDecodingV2ModelArguments(SCAModelBaseArguments):
    num_task_tokens: int = field(default=6)


@dataclass
class SCAMultitaskROIPoolModelArguments(SCAModelBaseArguments):
    num_task_tokens: int = field(default=6)
    vl_projector_type: str = field(default="linear")
    vl_projector_norm_type: str = field(default="none")


@dataclass
class ScaTimmMultitaskV2ModelArguments(SCAModelBaseArguments):
    timm_vision_name: str = field(default="vit_base_patch16_clip_224.openai")
    num_caption_tokens: int = field(default=1)
    num_task_tokens: int = field(default=6)
    num_caption_heads: int = field(default=1)


@dataclass
class DataArguments:
    _target_: str = "datasets.load_dataset"
    path: Optional[str] = field(default=None)
    name: Optional[str] = field(default=None)
    split: Optional[str] = field(default=None)
    cache_dir: str = field(default=".data.cache")
    streaming: bool = field(default=False)


@dataclass
class VGDenseCapDataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "visual_genome.py"))
    name: str = "region_descriptions_v1.2.0"

    base_image_url: Optional[str] = field(default=None)
    base_annotation_url: Optional[str] = field(default=None)
    sas_key: Optional[str] = field(default=None)

    use_densecap_splits: bool = field(default=True)

    with_image: bool = field(default=True)

    def __post_init__(self):
        if self.base_image_url is None:
            raise ValueError(
                "base_image_url must be specified in VGDenseCapDataArgument, since VisualGenome is not public available."
            )
        if self.base_annotation_url is None:
            raise ValueError(
                "base_annotation_url must be specified in VGDenseCapDataArgument, since VisualGenome is not public available."
            )
        if self.sas_key is None:
            logger.warning("sas_key maybe be specified in VGDenseCapDataArgument, since we fetch data from Azure.")


@dataclass
class VGDenseCapLocalDataArgument(DataArguments):
    path: str = field(
        default=osp.join(osp.dirname(__file__), "data", "data_scripts", "visual_genome-densecap-local.py")
    )
    name: str = "densecap"

    with_image: bool = field(default=True)

    base_dir: Optional[str] = field(default=None)
    base_annotation_dir: Optional[str] = field(default=None)


@dataclass
class VGGRiTLocalDataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "visual_genome-grit-local.py"))
    name: str = "grit"

    with_image: bool = field(default=True)

    base_dir: Optional[str] = field(default=None)
    base_annotation_dir: Optional[str] = field(default=None)


@dataclass
class RefCOCODataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "refcoco.py"))
    name: str = "refcoco-unc"

    base_url: Optional[str] = field(default=None)
    sas_key: Optional[str] = field(default=None)

    with_image: bool = field(default=True)
    with_mask: bool = field(
        default=False
    )  # To align with default vg-densecap-region_descriptions, which has no mask. Therefore we can concatenate them smoothly.


@dataclass
class SA1BCapDataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "sa1b_cap.py"))
    name: str = "mask_region_descriptions_v0.0.1"

    sa1b_tar_url: Optional[str] = field(default=None)
    sa1b_tar_template: Optional[str] = field(default=None)

    sa1b_annot_tsv_url: Optional[str] = field(default=None)
    sa1b_annot_template: Optional[str] = field(default=None)

    sa1b_cap_tsv_url: Optional[str] = field(default=None)
    sa1b_cap_template: Optional[str] = field(default=None)

    sa1b_filter_tsv_url: Optional[str] = field(default=None)
    sa1b_filter_template: Optional[str] = field(default=None)

    sa1b_file_range: Optional[str] = field(
        default=None,
        metadata={
            "help": "We use `ast.literal_eval` to parse the Python object. We assume it is a list of int or a `range` object."
        },
    )

    with_image: bool = field(default=True)
    with_mask: bool = field(
        default=False
    )  # To align with default vg-densecap-region_descriptions, which has no mask. Therefore we can concatenate them smoothly.


@dataclass
class COCOInstanceDataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "coco_instance.py"))
    name: str = "2017"

    coco_zip_url: Optional[str] = field(default=None)
    coco_annotations_zip_url: Optional[str] = field(default=None)

    with_image: bool = field(default=True)
    with_mask: bool = field(
        default=False
    )  # To align with default vg-densecap-region_descriptions, which has no mask. Therefore we can concatenate them smoothly.

    task_type: str = field(default="recognition")


@dataclass
class COCOInstanceLocalDataArgument(COCOInstanceDataArgument):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "coco_instance-local.py"))


@dataclass
class Objects365LocalDataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "objects365-local.py"))
    name: str = "v2"

    objects365_base_dir: Optional[str] = field(default=None)
    objects365_base_annotations_dir: Optional[str] = field(default=None)

    with_image: bool = field(default=True)
    with_mask: bool = field(
        default=False
    )  # To align with default vg-densecap-region_descriptions, which has no mask. Therefore we can concatenate them smoothly.

    task_type: str = field(default="recognition")


@dataclass
class V3DetLocalDataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "v3det-local.py"))
    name: str = "v1"

    v3det_base_dir: Optional[str] = field(default=None)
    v3det_base_annotations_dir: Optional[str] = field(default=None)

    with_image: bool = field(default=True)
    with_mask: bool = field(
        default=False
    )  # To align with default vg-densecap-region_descriptions, which has no mask. Therefore we can concatenate them smoothly.

    task_type: str = field(default="recognition")


@dataclass
class SBUPseudoRegionDataArgument(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "sbu-pseudo_region.py"))
    name: str = "pseudo_region"

    base_dir: Optional[str] = field(default=None)
    base_annotations_dir: Optional[str] = field(default=None)

    with_image: bool = field(default=True)
    with_mask: bool = field(default=False)  # NOTE: we don't have mask for sbu


@dataclass
class SBUPseudoRegionLocalDataArgument(SBUPseudoRegionDataArgument):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "sbu-pseudo_region-local.py"))


@dataclass
class COCOCaptionPseudoRegion(DataArguments):
    path: str = field(default=osp.join(osp.dirname(__file__), "data", "data_scripts", "coco_caption-pseudo_region.py"))
    name: str = "2017"

    coco_zip_url: Optional[str] = field(default=None)
    coco_annotations_zip_url: Optional[str] = field(default=None)

    with_image: bool = field(default=True)
    with_mask: bool = field(default=False)  # NOTE: we don't have mask for sbu


@dataclass
class WandbArguments:
    log: bool = field(default=True)
    project: Optional[str] = field(default="sca", metadata={"help": "wandb project"})
    group: Optional[str] = field(default="debug", metadata={"help": "wandb group"})
    name: Optional[str] = field(default="run", metadata={"help": "wandb run name"})
    tags: Optional[List[str]] = field(default=None, metadata={"help": "wandb tags"})
    resume: str = field(default="allow", metadata={"help": "wandb resume strategy"})
    id: Optional[str] = field(default=None, metadata={"help": "wandb run id"})


@dataclass
class DataTransformsArguments:
    min_scale: float = 0.1
    max_scale: float = 2.0
    image_size: int = 1024


defaults = [{"wandb": "base_wandb"}]


@dataclass
class Arguments:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    training: SCASeq2SeqTrainingArguments = field(default_factory=lambda: SCASeq2SeqTrainingArguments(output_dir="?"))

    # NOTE(xiaoke): to only maintain one sort of data config, we use soft links to link the data config to the train/eval config separately.
    # NOTE(xiaoke): Should be Union[List[DataArguments], DataArguments], while OmegaConf doesn't support Union. So use str to compose the configs dynamically.
    # NOTE(xiaoke): So we cannot override the args in the config file, since it will be converted to str.
    train_data: List[str] = field(default_factory=list)
    train_data_interleave_probabilities: Optional[List[float]] = field(default=None)
    train_data_overrides: List[str] = field(
        default_factory=list,
        metadata={"help": "overrides for train data. \"train_data_overrides='[data.with_image\=False]'\""},
    )

    eval_data: List[str] = field(default_factory=list)
    eval_data_overrides: List[str] = field(
        default_factory=list,
        metadata={"help": "overrides for eval data. \"eval_data_overrides='[data.with_image\=False]'\""},
    )

    model: ModelArguments = field(default_factory=ModelArguments)
    wandb: WandbArguments = field(default_factory=WandbArguments)

    data_transforms: Optional[DataTransformsArguments] = field(default=None)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Arguments)

cs.store(group="data", name="base_vg_densecap", node=VGDenseCapDataArgument)
cs.store(group="data", name="base_vg_densecap_local", node=VGDenseCapLocalDataArgument)
cs.store(group="data", name="base_vg_grit_local", node=VGGRiTLocalDataArgument)
cs.store(group="data", name="base_refcoco", node=RefCOCODataArgument)
cs.store(group="data", name="base_sa1b_cap", node=SA1BCapDataArgument)
cs.store(group="data", name="base_coco_instance", node=COCOInstanceDataArgument)
cs.store(group="data", name="base_coco_instance_local", node=COCOInstanceLocalDataArgument)
cs.store(group="data", name="base_objects365_local", node=Objects365LocalDataArgument)
cs.store(group="data", name="base_v3det_local", node=V3DetLocalDataArgument)
cs.store(group="data", name="base_sbu_pseudo_region", node=SBUPseudoRegionDataArgument)
cs.store(group="data", name="base_sbu_pseudo_region_local", node=SBUPseudoRegionLocalDataArgument)
cs.store(group="data", name="base_coco_caption_pseudo_region", node=COCOCaptionPseudoRegion)

cs.store(group="model", name="base_sam_captioner", node=SAMCaptionerModelArguments)
cs.store(group="model", name="base_sca", node=SCAModelArguments)
cs.store(group="model", name="base_sca_direct_decoding", node=SCADirectDecodingModelArguments)
cs.store(group="model", name="base_sca_multitask", node=SCAMultitaskModelArguments)
cs.store(group="model", name="base_sca_multitask_v2", node=ScaMultitaskV2ModelArguments)
cs.store(group="model", name="base_sca_multitask_split_mixer", node=SCAMultitaskSplitMixerModelArguments)
cs.store(group="model", name="base_sca_direct_decoding_v2", node=SCADirectDecodingV2ModelArguments)
cs.store(group="model", name="base_sca_multitask_roi_pool", node=SCAMultitaskROIPoolModelArguments)
cs.store(group="model", name="base_sca_timm_multitask_v2", node=ScaTimmMultitaskV2ModelArguments)


cs.store(group="wandb", name="base_wandb", node=WandbArguments)

cs.store(group="data_transforms", name="base_data_transforms", node=DataTransformsArguments)


def global_setup(
    args: DictConfig,
) -> Tuple[Arguments, SCASeq2SeqTrainingArguments, ModelArguments]:
    """Global setup of arguments."""
    if args.training.output_log_dir is not None:
        output_log_dir = args.training.output_log_dir
        if not osp.exists(output_log_dir):
            os.makedirs(output_log_dir)
        # NOTE: this is a dirty hack to enable logging to a different directory
        # by default in Hydra, logging.root.handlers contains two handler: stream & file
        # NOTE: mainly used in amulet
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                file_path = handler.baseFilename
                file_name = osp.basename(file_path)
                external_file_path = osp.join(output_log_dir, file_name)
                logging.root.addHandler(logging.FileHandler(external_file_path))
                logger.info(f"Add external file handler to {external_file_path}")
                break

    hostname = socket.gethostname()
    logger.info(f"Running on {hostname}")

    # Convert args to the actual dataclass object, to enable methods.  Need to
    # delete _n_gpu, a property that TrainingArgs init doesn't expect.
    del args.training._n_gpu
    # Dirty hack: only run post init when we're ready to convert to TrainingArgs
    args.training._run_post_init = True
    # NOTE: otherwise, do_eval will be set to True in TrainingArguments.__post_init__
    if args.training.do_eval == False and args.training.do_train == False:
        args.training.evaluation_strategy = "no"
        args.training.load_best_model_at_end = False

    training_args = OmegaConf.to_object(args.training)
    model_args = OmegaConf.to_object(args.model)

    if (
        isinstance(model_args, (SCAModelArguments, SCADirectDecodingModelArguments))
        and args.model.model_name_or_path is None
    ):
        # NOTE: we need to set the default value of `model_name_or_path` to None
        # otherwise, it will be set to `base_sca` by default
        raise ValueError(f"{type(model_args)} is not supported in model cfg name.")

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device},"
        f" log_level: {log_level} n_gpu: {training_args.n_gpu}"
        f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits"
        f" training: {training_args.fp16}, bf16 training: {training_args.bf16}"
    )
    logger.debug(f"Training/evaluation parameters {training_args}")

    return args, training_args, model_args
