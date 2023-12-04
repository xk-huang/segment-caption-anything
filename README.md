# Segment and Caption Anything

The repository contains the official implementation of "Segment and Caption Anything"

[Project Page](https://xk-huang.github.io/segment-caption-anything), [Paper](https://xk-huang.github.io/segment-caption-anything)

![teaser](./docs/teaser-github.svg)

tl;dr
1. Despite the absence of semantic labels in the training data, SAM implies high-level semantics sufficient for captioning. 
2. SCA (b) is a lightweight augmentation of SAM (a) with the ability to generate regional captions.
3. On top of SAM architecture, we add a fixed pre-trained language mode, and a optimizable lightweight hybrid feature mixture whose training is cheap and scalable.

<table>
  <tr>
    <td><img src="./docs/anything-mode-00.png.jpg" alt="anything-mode-00"></td>
    <td><img src="./docs/anything-mode-03.png.jpg" alt="anything-mode-01"></td>
  </tr>
  <tr>
    <td><img src="./docs/anything-mode-01.png.jpg" alt="anything-mode-02"></td>
    <td><img src="./docs/anything-mode-02.png.jpg" alt="anything-mode-03"></td>
  </tr>
</table>

## Environment Preparation

Please check [docs/ENV.md](docs/ENV.md).


## Model Zoo

Please check [docs/MODEL_ZOO.md](docs/MODEL_ZOO.md)


## Gradio Demo

Please check [docs/DEMO.md](docs/DEMO.md)


## Running Training and Inference

Please check [docs/USAGE.md](docs/USAGE.md).


## Experiments and Evaluation

Please check [docs/EVAL.md](docs/EVAL.md)


## Acknowledgement

Deeply appreciate these wonderful open source projects: [transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate), [deepspeed](https://github.com/microsoft/DeepSpeed), [detectron2](https://github.com/facebookresearch/detectron2), [hydra](https://github.com/facebookresearch/hydra), [timm](https://github.com/huggingface/pytorch-image-models), [gradio](https://github.com/gradio-app/gradio).

## Citation

If you find this repository useful, please consider giving a star ⭐ and citation 🦖:

```
@misc{xiaoke2023SCA,
  title={{Segment and Caption Anything}},
  author={Xiaoke, Huang and Jianfeng, Wang and Yansong, Tang and Zheng, Zhang and Han, Hu and Jiwen, Lu and Lijuan, Wang and Zicheng, Liu},
  journal={arXiv},
  year={2023},
}
```