# Model

To add a new model, you need to modify the following parts of the code:

1. Modify `configuration_*.py` file, especially the `from_*` static method.
2. Add a `modeling_*.py` files. Note ahout the `from_*` static method, which calls the `from_*` static method from `configuration_*.py`.
3. Add an argument class in `arguments.py` accordingly.
4. Add the new model in `__init__.py`.
5. Import the new model and argument class in the main script like `train.py`, and call `from_*` according to its parameters.
6. May need to add processor.

## Architecture

### Multitaskv2

`base_sca_multitask_v2`

It uses `task_type` to activate different task tokens, which are `recognition` and `caption`

### DirectDecodingv2 (MultitaskV2)

`base_sca_direct_decoding_v2`

Like Multitaskv2, but the caption tokens are the query tokens of SAM.

### SplitMixer (Multitaskv2)

`base_sca_multitask_split_mixer`

Like Multitaskv2, but it does not based on the fused tokens from SAM's feature mixer.

### ROI Pooler (Multitaskv2)


### Other Image features (Multitaskv2)


## Inputs and Outputs

SCA trainer requires that every items in  `logits` should not be `None`.
When it gathers the results across devices during inference, it calls `self._pad_across_processes` which recursively concatenates tensors.

## Attributes and Methods

TBD

## HF Trainer Adaption

TBD
