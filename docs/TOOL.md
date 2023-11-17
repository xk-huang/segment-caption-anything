## Make statistics about the datasets

Build the database of labels:

```shell
DATASET=vg-densecap-local
python scripts/tools/build_annotation_db.py train_data='['$DATASET']' eval_data='['$DATASET']' training.dataloader_num_workers=10  train_data_overrides='[data.with_image\=False]' eval_data_overrides='[data.with_image\=False]'
# training.output_dir=tmp/data/$DATASET
```

The table schema:

```sql
CREATE TABLE IF NOT EXISTS {table_name} (  
    region_id INTEGER PRIMARY KEY,  
    image_id INTEGER,  
    width INTEGER,  
    height INTEGER,  
    file_name TEXT,  
    coco_url TEXT,  
    task_type TEXT,  
    phrases TEXT,  
    tokenized_phrases TEXT,
    x REAL,  
    y REAL,  
    region_width REAL,  
    region_height REAL  
)  
```

The distribution of VG text num tokens: `scripts/notebooks/dataset_statstics_db.ipynb`


## Build annotation database (deprecated)

```shell
# for DATASET in objects365-local coco-instance v3det-local vg-densecap-region_descriptions refcocog-google; do
for DATASET in vg-densecap-region_descriptions ; do
    python scripts/tools/build_annotation_db.py \
    train_data='['"$DATASET"']' train_data_overrides='[data.with_image\=False]' \
    eval_data='['"$DATASET"']' eval_data_overrides='[data.with_image\=False]' \
    training.output_dir='tmp/annotation_db/'"$DATASET"
done
```

### Extract nouns from the annotation database

```shell
ANNOTATION_DB_PATH=
python scripts/tools/add_pos_table_annotation_db.py --db $ANNOTATION_DB_PATH

for i in coco-instance objects365-local refcocog-google v3det-local vg-densecap-region_descriptions ; do
    python scripts/tools/add_pos_table_annotation_db.py --db tmp/annotation_db/$i/annotations.db
done
```

### Visualize annotations with the Graio app

```shell
ANNOTATION_DB_PATH=
python scripts/apps/annotation_db_app.py --db $ANNOTATION_DB_PATH
```

### Get the noun comparison

```shell
# Default output path: tmp/annotation_db_noun_stats/compare_nouns_annotation_db.xlsx
for i in coco-instance objects365-local v3det-local refcocog-google ; do
    python scripts/tools/compare_nouns_annotation_db.py --db tmp/annotation_db/$i/annotations.db --db tmp/annotation_db/vg-densecap-region_descriptions/annotations.db # -o OUTPUT_PATH
done
```


## To get dataset statistics (deprecated)

The statistics result is saved in `*/dataset_statistics.log`. Therefore, we need to parse the log file.

Load with images of which we check the sanity (slow)

- Objects365: 1742289 images, ~ 3 hours.
- V3Det: 183348 images, ~ 30 minutes.

```shell
for DATASET in objects365-local; do
    torchrun --nproc-per-node 12 --standalone  scripts/tools/dataset_statistics.py \
    train_data='['"$DATASET"']' \
    eval_data='['"$DATASET"']' \
    training.output_dir='tmp/dataset_statistics-w_image/'"$DATASET"
done
```

Only do statistics (fast, < 2 min)

```shell
for DATASET in objects365-local; do
    torchrun --nproc-per-node 12 --standalone  scripts/tools/dataset_statistics.py \
    train_data='['"$DATASET"']' train_data_overrides='[data.with_image\=False]' \
    eval_data='['"$DATASET"']' eval_data_overrides='[data.with_image\=False]' \
    training.output_dir='tmp/dataset_statistics-wo_image/'"$DATASET"
done
```

Do not use multiple eval datasts.

### Parse the log to csv

```bash
#!/bin/bash

# Input argument: base_dir
base_dir=$1
output_file="${base_dir}/dataset_statistics-full.$(date +%m%d%y).csv"
# Create a CSV file with the header
echo "dataset,split,total samples,total regions,total sents,total tokens,total words" > "$output_file"

# Find all "dataset_statistics.log" files under the base_dir
find "${base_dir}" -type f -name "dataset_statistics.log" | while read log_file; do
  # Extract the dataset name from the directory path
  dataset=$(basename $(dirname "${log_file}"))

  # Parse the log file and extract the required information
  grep -E "\[FULL\]: split name" "${log_file}" | while read line; do
    # Extract the values for each field
    split=$(echo "${line}" | grep -oP "split name: \K\w+")
    total_samples=$(echo "${line}" | grep -oP "total samples: \K\d+")
    total_regions=$(echo "${line}" | grep -oP "total regions: \K\d+")
    total_sents=$(echo "${line}" | grep -oP "total sents: \K\d+")
    total_tokens=$(echo "${line}" | grep -oP "total tokens: \K\d+")
    total_words=$(echo "${line}" | grep -oP "total words: \K\d+")

    # Append the parsed information to the CSV file
    echo "${dataset},${split},${total_samples},${total_regions},${total_sents},${total_tokens},${total_words}" >> "$output_file"
  done
done
```

## Test dataloading.

```shell
python scripts/tools/test_dataset_loading.py \
train_data='[vg-densecap-region_descriptions]' \
eval_data='[vg-densecap-region_descriptions]' \
+data_transforms=lsj-1_0-2_0 \
+model=base_sca_multitask_v2 \
training.do_train=True \
training.do_eval=True \
training.per_device_train_batch_size=1 \
training.num_masks_per_sample=16 \
training.dataloader_num_workers=10 \
```


## Get param size

```shell
for i in 2 4 8 12 24; do
    python scripts/tools/count_num_params.py \
    train_data='[vg-densecap-local]' \
    eval_data='[vg-densecap-local]' \
    training.do_train=True training.do_eval=True \
    +model=base_sca_multitask_v2 model.num_caption_tokens=8 model.additional_num_hidden_layers=$i model.num_task_tokens=6 | grep mask_decoder.additional_transformer >> tmp/mixer_size.txt
done

for i in facebook/sam-vit-huge facebook/sam-vit-large facebook/sam-vit-base ; do
    python scripts/tools/count_num_params.py \
    train_data='[vg-densecap-local]' \
    eval_data='[vg-densecap-local]' \
    training.do_train=True training.do_eval=True \
    +model=base_sca_multitask_v2 model.sam_model_name_or_path=$i | grep vision_encoder >> tmp/vision_encoder_size.txt
done
```