# Usage

```shell
# accelerate launch --config_file amlt_configs/accelerate_deepspeed_config.local.yaml \
python \
    -m src.train \
    train_data='[vg-densecap-local]' eval_data='[vg-densecap-local]' \
    +model=base_sca_multitask_v2 \
    model.cache_dir=.model.cache/ \
    training.do_train=True \
    training.do_eval=True \
    training.fp16=True \
    training.num_masks_per_sample=16 \
    training.per_device_train_batch_size=1 \
    training.dataloader_num_workers=4 \
    training.max_steps=99 \
    training.logging_first_step=True \
    training.logging_steps=5 \
    training.evaluate_before_train=True \
    training.max_eval_samples=3 \
    training.eval_steps=50 \
    training.save_steps=50 \
    wandb.log=False \
    training.lr_scheduler_type=cosine \
    +data_transforms=lsj-0_1-2_0 \
    model.lm_head_model_name_or_path=gpt2 \
    model.sam_model_name_or_path=facebook/sam-vit-base

    # model.lm_head_model_name_or_path=openlm-research/open_llama_3b_v2
    # To use llama, you need to install sentencepiece
    # training.gradient_checkpointing=true

    # Use extra args in data module
    # train_data_overrides='[data.streaming\=True]'
```

SCA

Training.

```shell
python \
    -m src.train \
    train_data='[vg-densecap-local]' eval_data='[vg-densecap-local]' \
    +model=base_sca \
    training.do_train=True \
    training.do_eval=True \
    training.num_masks_per_sample=32 \
    # training.num_masks_per_sample=10 \
    # training.num_masks_per_sample=4 \
    +data.streaming=False \
    training.per_device_train_batch_size=1 \
    training.fp16=True \
    # model.lm_head_model_name_or_path=gpt2-large \
    # model.lm_head_model_name_or_path=gpt2-xl \
    training.dataloader_num_workers=4 \
    training.logging_first_step=True \
    training.trainable_params='[mask_decoder.additional_transformer,mask_decoder.caption_tokens,task_tokens,language_project,language_model]'
    +training.custom_param_lrs='{language_model:1e-5}'
    training.compute_metrics=null # Computer METEOR during training. If ture, use generate, about 0.4 it/s on A100; false or null, only compute loss, 1.5 it/s
```

Inference.

```shell
python \
    -m src.train \
    train_data='[vg-densecap-local]' eval_data='[vg-densecap-local]' \
    +model=base_sca \
    training.do_train=False \
    training.do_eval=False \
    training.do_inference=True \
    training.output_dir=amlt/train-sca-vg_densecap-081023/gpt2-large/ \
    wandb.log=False \    # training.fp16_full_eval=True
    model.model_name_or_path=amlt/train-sca-vg_densecap-081023/gpt2-large/checkpoint-9000 \
    # FIXME: when load weights from existing sca model, we should use the same tokenizer as the existing sca model
    # model.lm_head_model_name_or_path=$(grep lm_head_model_name_or_path $AMLT_MAP_INPUT_DIR/.hydra/config.yaml | tail -n1 | sed 's/ *//g' | cut -d ':' -f2)
    # model.sam_model_name_or_path=$(grep sam_model_name_or_path $AMLT_MAP_INPUT_DIR/.hydra/config.yaml | tail -n1 | sed 's/ *//g' | cut -d ':' -f2)
```

## Data Configs

```shell
src/conf/data
├── coco_caption-pseudo_region.yaml
├── coco-instance-local.yaml
├── coco-instance-task_type_caption-local.yaml
├── coco-instance-task_type_caption.yaml
├── coco-instance.yaml
├── objects365-local.yaml
├── objects365-task_type_caption-local.yaml
├── refclef-berkeley.yaml
├── refclef-unc.yaml
├── refcocog-google.yaml
├── refcoco-google.yaml
├── refcocog-umd.yaml
├── refcoco+-unc-split_testA.yaml
├── refcoco-unc-split_testA.yaml
├── refcoco+-unc-split_testB.yaml
├── refcoco-unc-split_testB.yaml
├── refcoco+-unc.yaml
├── refcoco-unc.yaml
├── sa1b-cap-streaming-hard_code_filter-num_tars_11.yaml
├── sa1b-cap-streaming-hard_code_filter-num_tars_2.yaml
├── sa1b-cap-streaming-hard_code_filter-num_tars_6.yaml
├── sa1b-cap-streaming-num_tars_11.yaml
├── sa1b-cap-streaming-num_tars_2.yaml
├── sa1b-cap-streaming-num_tars_6.yaml
├── sa1b-cap-streaming.yaml
├── sbu-pseudo_region-local.yaml
├── sbu-pseudo_region.yaml
├── v3det-local.yaml
├── v3det-task_type_caption-local.yaml
├── vg-densecap-local.yaml
├── vg-densecap-mask_region_descriptions.yaml
├── vg-densecap-region_descriptions.yaml
├── vg_densecap.yaml
├── vg-full-vg-densecap-mask_region_descriptions.yaml
├── vg-full-vg-densecap-region_descriptions.yaml
└── vg-grit-local.yaml
```

## Debug

Use vscode debugger, the config is in `.vscode/launch.json`.

```shell
python -m debugpy --wait-for-client --listen 0.0.0.0:5678 \
    -m src.train \
    train_data='[vg-densecap-region_descriptions]' eval_data='[vg-densecap-region_descriptions]' \
    +model=base_sam_captioner \
    training.do_train=True \
    training.do_eval=True \
    training.num_masks_per_sample=6 \
    +data.streaming=False \
    # sample
    training.max_eval_samples=1 \
    training.max_train_samples=1 \
    # logging training step
    training.logging_steps=5 \
    # eval
    training.evaluation_strategy=steps \
    training.eval_steps=5 \
    # num_stape
    training.max_steps=1000 \
    # save model
    training.save_strategy=steps \
    training.save_steps=10 \
    training.save_total_limit=2 \
    # optimizer
    training.optim=adamw_torch
    training.learning_rate=5e-5
    # wandb
    wandb.log=False
    wandb.project=sca
    wandb.group=debug
    wandb.name=sca-debug
    # test
    training.evaluate_before_train=False \
    # Set log_level in `transformer` to `info`. By default, it is `warning`.
    # debug - 10; info - 20; warning - 30; error - 40; critical - 50;
    # by default, it is `passive` which is 30.
    training.log_level="info"
    # Set log_level=DEBUG in my loggers controlled by hydra.
    hydra.verbose=true
```




## About Wandb Resume

We save the run id inside `training.output_dir/wandb_id`. Therefore, if the output_dir is different, then the wandb run_id should be different.

- Reference: https://github.com/wandb/wandb/issues/335#issuecomment-493284910
