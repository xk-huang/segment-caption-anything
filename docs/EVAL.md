## Experiments

Please check `amlt_configs/` for the experiments configs.

## Evaluate with `vdtk`

### Install `vdtk`

Support CLIP computation with images encoded by base64.

https://github.com/xk-huang/vdtk/tree/dev

- data (e.g., jar files): https://huggingface.co/xk-huang/vdtk-data

Install with external data:

#### Docker

```shell
alias=`whoami | cut -d'.' -f2`
docker run -itd --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} -w `pwd` --name sca nvcr.io/nvidia/pytorch:22.10-py3 bash
docker exec -it sca bash

# In the docker container
# cd to the code dir
. amlt_configs/setup.sh
source ~/.bashrc
pip install pydantic==1.10.8  # https://github.com/pydantic/pydantic/issues/545#issuecomment-1573776471
. amlt_configs/setup_eval_suite.sh
```

#### Conda

```shell
ORIGINAL_DIR="$(pwd)"
REPO_DIR=/tmp/vdtk
git clone --recursive https://github.com/xk-huang/vdtk.git $REPO_DIR -b dev
cd $REPO_DIR
git submodule update --init --recursive

apt-get update
sudo apt-get update
apt-get install git-lfs
sudo apt-get install git-lfs

git lfs install
git clone https://huggingface.co/xk-huang/vdtk-data
# git submodule init && git submodule update

rsync -avP ./vdtk-data/vdtk .
rm -rf vdtk-data

pip install --upgrade pip
pip install -e . POT==0.9.0  # POT=0.9.1 will take up all the memory with tf backend
pip install tensorflow==2.12.1  # Just fix one version of tf
pip install levenshtein==0.21.1
pip install openpyxl==3.1.2

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
cd "$ORIGINAL_DIR"
```

Potential Problems:

- About Tensorflow: TF does not support CUDA 12 now (08/15/23). So we use `nvcr.io/nvidia/pytorch:22.12-py3` which contains CUDA 11.8.
- Encoding in docker image: `import locale;locale.getpreferredencoding()` is `ANSI_X3.4-1968` rather than `UTF-8` which causes error in file writing.
  - change `vdtk/metrics/tokenizer/ptbtokenizer.py:73`: `tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")`


### The format of input prediction json file

```json
[
    {
        "_id": 0,
        "split": "inference",
        "references": [
            "a man wearing a red and white shirt"
        ],
        "candidates": [
            "red and yellow",
            "red shirt guy",
            "red and yellow uniform"
        ],
        "metadata": {
            "metadata_input_boxes": [
                0,
                95,
                113,
                419
            ],
            "metadata_image_id": 266240,
            "metadata_region_id": 27287
        },
        "logits": {
            "iou_scores": [
                0.89990234375,
                0.994140625,
                0.99365234375
            ]
        }
    }
]
```

### The structure of files

```
$OUTPUT_DIR/infer/infer-visual_genome-densecap-local-densecap-test.json
# infer-{data_script_identifier}-{name}-{split}.json
```

### All-in-one script

Usage:

```shell
>>> bash scripts/tools/eval_suite.sh
# Env args:
#        DRY_RUN: 
#        ONLY_GATHER: 
#        ONLY_EVAL: 
#        SKIP_CLIP_RECALL: 
#        DEBUG: 
#         NO_POST_PROCESS: 
# Usage: [DRY_RUN=1] [ONLY_GATHER=1] [ONLY_EVAL=1] ./eval_suite.sh <INFERENCE_JSON_DIR> <JSON_FILE_NAME> <SPLIT> [<IMAGE_B64_TSV_PATH>] [<MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT>] [<POST_PROCESS_MULTI_CANDIDATES_SCRIPT>]JSON_FILE_NAME is not used, use any string like 'xxx' for it.   
```

e.g.,

```bash
DRY_RUN=1 NO_POST_PROCESS=1 ONLY_EVAL=1 SKIP_CLIP_RECALL=1 bash scripts/tools/eval_suite.sh exp/ xxx inference
```

<details>
<summary>The details about the script.</summary>

1. Replace GT captions (the tokenizer processed ones) with the real GT (`scripts/tools/replace_references_in_json_for_vdtk.py`). Please prepare the folder structure correctly as in [this](The structure of files). It requires the `.hydra` config.
2. Remove multiple predictions but keep one based on IOU score (`scripts/tools/post_process_multi_candidates_for_vdtk.py`).

If there are multiple candidate preditions, we only choose **one candidates** with highest IOU for Meteor, CIDEr-D, ROUGE, etc.:

```shell
python scripts/tools/post_process_multi_candidates_for_vdtk.py -i $INFERENCE_JSON_PATH
```

Process multiple inference json file under a certain dirctory:

```shell
INFERENCE_JSON_DIR=
find $INFERENCE_JSON_DIR -name 'infer.json' -exec python scripts/tools/post_process_multi_candidates_for_vdtk.py -i {} \;
```

3. evaluate with vdtk, and save the results in `.log` file

You need to change `PRED_JSONS_BASE_DIR`, `JSON_FILE_NAME`, `SPLIT`, and `IMAGE_B64_TSV_PATH`.

If the `infer.json` file is too large to open in vscode, you can use vim to open it and change the above variables accordingly.

Currently, `JSON_FILE_NAME` is deprecated as we `find` the `*.json` in `PRED_JSONS_BASE_DIR`.

4. Parse the results for each `*.log` and gather to one xlsx by sheets.

Parse the log. Change the `PRED_JSONS_BASE_DIR` accordingly.

5. Merge each metric into one table with `scripts/tools/merge_sheets_xlsx.py`

</details>