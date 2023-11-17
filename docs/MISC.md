## Regex to remove iou scores in `infer.json`

```json
        },
        "logits": {
            "iou_scores": [
                0.95166015625,
                0.94873046875,
                0.82177734375
            ]
        }
```

```re
,\n\s*"logits": \{\n\s*"iou_scores":\s*\[\n\s*([\d.]+)\s*,\n\s*([\d.]+)\s*,\n\s*([\d.]+)\n\s*\]\n\s*\}
```

## List of captioner models

Salesforce/blip-image-captioning-large
Salesforce/blip-image-captioning-base

Salesforce/blip2-opt-2.7b
Salesforce/blip2-opt-6.7b-coco
Salesforce/blip2-opt-6.7b
Salesforce/blip2-opt-2.7b-coco

<!-- Need prompts -->
<!-- Salesforce/instructblip-vicuna-7b -->
<!-- Salesforce/instructblip-vicuna-13b -->

microsoft/git-large-coco
microsoft/git-large-textcaps
microsoft/git-base
microsoft/git-base-coco
microsoft/git-base-textcaps
microsoft/git-large
microsoft/git-large-r
microsoft/git-large-r-coco
microsoft/git-large-r-textcaps

<!-- No official code -->
<!-- laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k -->
<!-- laion/CoCa-ViT-B-32-laion2B-s13B-b90k -->
<!-- laion/CoCa-ViT-L-14-laion2B-s13B-b90k -->
<!-- laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k -->

```shell
for model in \
Salesforce/blip2-opt-2.7b \
Salesforce/blip2-opt-2.7b-coco \
Salesforce/blip2-opt-6.7b \
Salesforce/blip2-opt-6.7b-coco 
do
python \
    -m src.train \
    train_data='[vg-densecap-local]' eval_data='[vg-densecap-local]' \
    +model=base_sam_captioner \
    training.do_train=False \
    training.do_eval=False \
    training.do_inference=True \
    +data.streaming=False \
    training.fp16=True \
    training.output_dir=tmp/sam_captioner/$model \
    training.dataloader_num_workers=4 \
    model.captioner_model_name_or_path=$model
done
```

## The process of batch generation of language model

`transformers/generation/utils.py:GenerationMixin:generate`



## Chunckified inference

Regional chunk size is set to 16

| SAM Model | Captioner                              | fp16 | region chunk size | Memory (GB) | Speed (s/it) |
| --------- | -------------------------------------- | ---- | ----------------- | ----------- | ------------ |
| ViT-huge  | Salesforce/blip-image-captioning-base  | Yes  | 16                | ~ 9         | ~ 5.02       |
| ViT-huge  | Salesforce/blip-image-captioning-base  | No   | 16                | ~ 8         | ~ 8.29       |
| ViT-huge  | Salesforce/blip-image-captioning-large | Yes  | 16                | ~ 10        | ~ 6.28       |
| ViT-huge  | Salesforce/blip-image-captioning-large | No   | 16                | ~ 9.7       | ~ 14.99      |
| ViT-huge  | Salesforce/blip2-opt-2.7b              | Yes  | 16                | ~ 34        | ~ 5.82       |
| ViT-huge  | Salesforce/blip2-opt-2.7b              | No   | 16                | ~ 32        | ~ 18.19      |
| ViT-huge  | Salesforce/blip2-opt-2.7b              | Yes  | 4                 | ~ 34        | ~ 11.56      |
| ViT-huge  | microsoft/git-large-coco               | Yes  | 16                | ~ 14        | ~ 7.06       |
| ViT-huge  | microsoft/git-base-coco                | Yes  | 16                | ~ 12        | ~ 3.26       |

## Bugs in SAM batch inference when transformers<=4.30.2

Remember to update the `requirements.txt` file. Otherwise we should always set batch_size=1.

Here is the fixing pr which was merged already after version 4.30.2: https://github.com/huggingface/transformers/pull/25074

## Debug the distributed training

Inside the trainer, we can access the main process by:

```python
if args.local_process_index == 0:
    breakpoint()
torch.distributed.barrier()
# the problematic line
labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
```

`try-catch` does not trigger the pdb interface:

```python
try:
    # the problematic line
    labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
except Error as e:
    if args.local_process_index == 0:
        breakpoint()
finally:
    torch.distributed.barrier()
```

## Amulet T4 instance is maintained into wrong information about the number of GPUs

Wrong T4 instance information is maintained by singularity, where `Standard_NC{4,8,16,32}as_T4_v3` only have 1, 1, 1, and 2 GPUs separately, but they are showed to have 1, 2, 4, and 4 GPUs separately.

in `amlt/helpers/sing_instances.py`, we add the below code:

```python
# add at 377, in amlt/helpers/sing_instances.py:fetch_instances_for_series
              # NOTE(xiaoke): Fix T4 wrong number of GPU
              if accelerator == "T4":
                instance_name_to_num_gpu = {
                  "NC8as_T4_v3": ["2", "1"],
                  "NC16as_T4_v3": ["4", "1"],
                  "NC32as_T4_v3": ["4", "2"],
                }
                if instance_name in instance_name_to_num_gpu:
                  description = description.replace(f"GPU x {instance_name_to_num_gpu[instance_name][0]}", f"GPU x {instance_name_to_num_gpu[instance_name][1]}")
                  info = re.search(match, description)
```

Note that we need to print the instance out explicitly. Sometimes we fail to get 4 cards while only get 1 card.

```python
# add at 422, amlt/client/sing_client.py:_setup_script_run_config
print(f"instance: {job.sku.instance}, sku: {job.sku}")
```

How to check:

```shell
amlt cache instance-types
amlt cache instance-types -s NCast4v3
```

## Debug the commands generated by amlt

```
amlt show EXP JOB
```

(deprecated)
```python
# /anaconda/envs/sca-v2/lib/python3.9/site-packages/amlt/client/aml_client.py:create_context
# At the end of this function
      inspect_amlt_job_dir = "tmp/amlt_job/"
      try:
        print(f"Copy code from {code_resource.remote_dir} to {temp_dir}.")
        if os.path.exists(inspect_amlt_job_dir):
            shutil.rmtree(inspect_amlt_job_dir, ignore_errors=True)
        shutil.copytree(temp_dir, inspect_amlt_job_dir)
      except Exception as e:
        print(f"Cannot copy code from {temp_dir} to {inspect_amlt_job_dir} due to {e}")
      yield temp_dir
```

## Test tokenizer


```python
from transformers import AutoProcessor

gpt2_large_tokenizer_cfg = dict(
    pretrained_model_name_or_path="gpt2-large",
    use_fast=True)

openllama_tokenizer_cfg = dict(
    pretrained_model_name_or_path='openlm-research/open_llama_3b_v2',
    use_fast=False)

def print_func(tokenizer, list_of_str):
    print(f"{list_of_str}: {tokenizer(list_of_str)['input_ids']}")

tokenizer = AutoProcessor.from_pretrained(**gpt2_large_tokenizer_cfg)
print_func(tokenizer, ["car", "Car", "CAR"])
print_func(tokenizer, ["tokenizer", "Tokenizer", "TOKENIZER"])

tokenizer = AutoProcessor.from_pretrained(**openllama_tokenizer_cfg)
print_func(tokenizer, ["car", "Car", "CAR"])
print_func(tokenizer, ["tokenizer", "Tokenizer", "TOKENIZER"])
```