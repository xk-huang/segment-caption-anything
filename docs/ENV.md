## Env

### Use Docker

The docker image: `nvidia/pytorch:23.07-py3`

```shell
alias=`whoami | cut -d'.' -f2`
docker run -itd --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} -w `pwd` --name sca nvcr.io/nvidia/pytorch:22.10-py3 bash
docker exec -it sca bash

# In the docker container
# cd to the code dir
. amlt_configs/setup.sh
```

### Use Conda

```shell
conda create -n sca-v2 -y python=3.9 
conda activate sca-v2

# https://pytorch.org/, pytorch 2.0.1 (as of 07/12/2023)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -U datasets==2.16.1  # NotImplementedError(f"Loading a dataset cached in a {type(self._fs).__name__} is not supported.") https://github.com/huggingface/datasets/issues/6352
```

For dev

```shell
mkdir -p tmp/{data,code}
pip install -r requirements-dev.txt 
```

For gradio demo:

```
pip install -r requirements-app.txt 
```

`transformers` is based on v4.30.2, hash 66fd3a8d6

```shell
REPO_DIR=transformers
git clone git@github.com:huggingface/transformers.git $REPO_DIR
git fetch --all --tags --prune
git checkout v4.30.2 -b v4.30.2
git rev-parse --short HEAD
```

## Data

Replace the data file paths in `src/conf/data/*.yaml`.

The data file links:
- VG: https://homes.cs.washington.edu/~ranjay/visualgenome/index.html
- COCO: https://cocodataset.org/
- Objects365: https://www.objects365.org/
- V3Det: https://v3det.openxlab.org.cn/