# Experiments

## Types of visual prompts:

Our method supports both point and box prompts as SAM does.

The dataset only contains ground-truth boxes instead of points, so we propose to generate pseudo-point prompts with

1) the center point of the box,
2) the random point in the box, and
3) the random point in the mask with highest confidence score predicted by SAM.


### Debug code:

```shell
DATASET=vg-densecap-local
CKPT_PATH=
conda run -n sca-v2 --no-capture-output python \
    -m src.train \
    wandb.log=False \
    train_data='['$DATASET']' eval_data='['$DATASET']' \
    +model=base_sca_multitask_v2 \
    training.do_train=False \
    training.do_eval=False \
    training.do_inference=True \
    model.model_name_or_path=$CKPT_PATH \
    model.lm_head_model_name_or_path=$(python scripts/tools/get_sub_model_name_from_ckpt.py $CKPT_PATH "lm") \
    training.output_dir=exp/debug/$DATASET \
    training.generate_chunk_size=64 \
    training.max_eval_samples=10  \
    # training.generation_num_beams=3 \  # reduce inference speed. maybe about 30%
    # training.fp16_full_eval=True \  # faster inference on A100, for 1X speed up
    # training.prompt_types_to_ablate_on_vg=null  # Time: 1:20 ; GPU: 11G
    # training.prompt_types_to_ablate_on_vg=center_point_in_box  # Time: 1:20 ; GPU: 11G
    # training.prompt_types_to_ablate_on_vg=random_point_in_box  # Time: 1:20 ; GPU: 11G
    # training.prompt_types_to_ablate_on_vg=random_point_in_mask  # Time: 2:30 ; GPU: 12G
```

```shell
DATASET=vg-densecap-local
CKPT_PATH=
for generation_num_beams in 1 3; do
for prompt_types_to_ablate_on_vg in null center_point_in_box random_point_in_box random_point_in_mask; do
    conda run -n sca-v2 --no-capture-output python \
        -m src.train \
        wandb.log=False \
        train_data='['$DATASET']' eval_data='['$DATASET']' \
        +model=base_sca_multitask_v2 \
        training.do_train=False \
        training.do_eval=False \
        training.do_inference=True \
        model.model_name_or_path=$CKPT_PATH \
        model.lm_head_model_name_or_path=$(python scripts/tools/get_sub_model_name_from_ckpt.py $CKPT_PATH "lm") \
        training.output_dir=exp/ablat-prompt_type/$DATASET/beam_num-$generation_num_beams/$prompt_types_to_ablate_on_vg \
        training.generate_chunk_size=64 \
        training.generation_num_beams=$generation_num_beams \
        training.fp16_full_eval=True \
        training.prompt_types_to_ablate_on_vg=$prompt_types_to_ablate_on_vg
done
done
```

### Dev

Replace `input_boxes` to `input_points`

```
input_boxes
(batch_size, num_boxes_per_image, 4)
torch.Size([1, 35, 4])

input_points 
(batch_size, point_batch_size, num_points_per_image, 2)
torch.Size([1, 35, 1, 2])
```

### Eval

```shell
# One by one
conda run -n sca --no-capture-output vdtk score ciderd ???.json --split inference --save-dist-plot --save-scores

# all-in-one script
NO_POST_PROCESS=1 SKIP_CLIP_RECALL=1 conda run -n sca --no-capture-output bash scripts/tools/eval_suite.sh ???  xxx inference
```

## BLIP2 + V-CoT

```shell
# Salesforce/blip2-opt-2.7b
# Salesforce/blip2-opt-2.7b-coco
# Salesforce/blip2-flan-t5-xl  # no outputs
# Salesforce/instructblip-flan-t5-xl  # no outputs
model=Salesforce/blip2-opt-2.7b
conda run -n sca-v2 --no-capture-output python \
    -m src.train \
    train_data='[vg-densecap-local]' eval_data='[vg-densecap-local]' \
    model.cache_dir=.model.cache/ \
    +model=base_sam_captioner \
    training.do_train=False \
    training.do_eval=False \
    training.do_inference=True \
    +data.streaming=False \
    training.fp16=True \
    training.output_dir=tmp/sam_captioner/$model \
    training.dataloader_num_workers=4 \
    model.captioner_model_name_or_path=$model


# change `chunkified_forward_size = 64` for A100
# fp32: ? hours
# fp16: 4 hours
#     by default, model.dtype=float16
#     model.use_vcot=False
```

### Dev

```python
captioner_inputs = self.captioner_processor(images=patches[:2], text=["what is this?"]*2, return_tensors="pt").to(
    self.device, dtype=self.torch_dtype
)
self.captioner_processor.batch_decode(self.captioner.generate(**captioner_inputs))


captioner_inputs = self.captioner_processor(images=patches[:2], return_tensors="pt").to(
    self.device, dtype=self.torch_dtype
)
self.captioner_processor.batch_decode(self.captioner.generate(**captioner_inputs))
```


### Run

```shell
model=Salesforce/blip2-opt-2.7b
for max_eval_samples in 250 500 1000; do
conda run -n sca-v2 --no-capture-output python \
    -m src.train \
    train_data='[vg-densecap-local]' eval_data='[vg-densecap-local]' \
    model.cache_dir=.model.cache/ \
    +model=base_sam_captioner \
    training.do_train=False \
    training.do_eval=False \
    training.do_inference=True \
    +data.streaming=False \
    training.fp16=True \
    training.output_dir=tmp/sam_captioner/$model/$max_eval_samples/w_vcot \
    training.dataloader_num_workers=4 \
    model.captioner_model_name_or_path=$model \
    model.use_vcot=True \
    training.max_eval_samples=$max_eval_samples
conda run -n sca-v2 --no-capture-output python \
    -m src.train \
    train_data='[vg-densecap-local]' eval_data='[vg-densecap-local]' \
    model.cache_dir=.model.cache/ \
    +model=base_sam_captioner \
    training.do_train=False \
    training.do_eval=False \
    training.do_inference=True \
    +data.streaming=False \
    training.fp16=True \
    training.output_dir=tmp/sam_captioner/$model/$max_eval_samples/wo_vcot \
    training.dataloader_num_workers=4 \
    model.captioner_model_name_or_path=$model \
    model.use_vcot=False \
    training.max_eval_samples=$max_eval_samples
done
# w/ vcot: 8h
# w/o vcot: 5h
```

### Eval

```shell
# One by one
conda run -n sca --no-capture-output vdtk score ciderd ???.json --split inference --save-dist-plot --save-scores

# all-in-one script
SKIP_CLIP_RECALL=1 conda run -n sca --no-capture-output bash scripts/tools/eval_suite.sh ??? xxx inference
```

## Intuition of Verb metric

1. Download the json from `blip2`'s prediction.
2. Download the json from `sca`'s prediction.
3. Modify `vdtk` to save similarity scores.
4. Merge sentence pairs, score pairs.
5. Analyse results.

Cider:

```shell
json=???.json
conda run -n sca --no-capture-output vdtk score ciderd $json --split inference | tee $json.score.txt


for json in `find ??? -name '*.post.json'`; do
    conda run -n sca --no-capture-output vdtk score ciderd $json --split inference | tee $json.score.txt
done
```

Verb:

```shell
json=???.json
conda run -n sca --no-capture-output vdtk content-recall $json --split inference --num-workers 16 --save-dist-plot --save-scores | tee $json.content-recall.txt

for json in `find ??? -name '*.post.json'`; do
    conda run -n sca --no-capture-output vdtk content-recall $json --split inference --num-workers 16 --save-dist-plot --save-scores  | tee $json.content-recall.txt
done
```