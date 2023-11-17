import transformers
from transformers import AutoModel, AutoProcessor, AutoConfig
import torch
import numpy as np
from typing import Mapping, Sequence

SEED = 42
transformers.enable_full_determinism(SEED)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CAPTION_PRETRAIN_MODELS_NAMES = [
    "Salesforce/blip-image-captioning-base",
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip2-opt-2.7b",
]
CAPTION_PRETRAIN_MODEL = CAPTION_PRETRAIN_MODELS_NAMES[0]
# NOTE: If you use BLIP2 model, you need to change the `pixel_values_shape` below accordingly.
CACHE_DIR = ".model.cache/"
# DEVICE = "cpu"
DEVICE = "cuda"

# MODEL
config = AutoConfig.from_pretrained(CAPTION_PRETRAIN_MODEL, cache_dir=CACHE_DIR)
caption_architectures = config.architectures
if len(caption_architectures) != 1:
    print(f"captioner_architectures: {caption_architectures} has to be of length 1")
caption_architecture = caption_architectures[0]

module = getattr(transformers, caption_architecture)
model = module.from_pretrained(CAPTION_PRETRAIN_MODEL, cache_dir=CACHE_DIR)
processor = AutoProcessor.from_pretrained(CAPTION_PRETRAIN_MODEL, cache_dir=CACHE_DIR)
model.to(DEVICE)

# Data
pixel_values_shape = [1, 3, 384, 384]  # shape for BLIP
# pixel_values_shape = [1, 3, 224, 224]  # shape for BLIP2
input_ids_shape = [1, 17]
attention_mask_shape = [1, 17]
labels_shape = [1, 17]


single_sample_inputs = {
    "pixel_values": torch.ones(pixel_values_shape),
    "input_ids": torch.ones(input_ids_shape, dtype=torch.long),
    "attention_mask": torch.ones(attention_mask_shape, dtype=torch.long),
    "labels": torch.ones(labels_shape, dtype=torch.long),
}

batch_size = 2
batch_sample_inputs = {
    "pixel_values": single_sample_inputs["pixel_values"].repeat(batch_size, 1, 1, 1),
    "input_ids": single_sample_inputs["input_ids"].repeat(batch_size, 1),
    "attention_mask": single_sample_inputs["attention_mask"].repeat(batch_size, 1),
    "labels": single_sample_inputs["labels"].repeat(batch_size, 1),
}
for k in single_sample_inputs:
    single_sample_inputs[k] = single_sample_inputs[k].to(DEVICE)
for k in batch_sample_inputs:
    batch_sample_inputs[k] = batch_sample_inputs[k].to(DEVICE)

with torch.no_grad():
    single_sample_outputs = model(**single_sample_inputs)
    batch_sample_outputs = model(**batch_sample_inputs)

print(f"Model: {CAPTION_PRETRAIN_MODEL} with {caption_architecture}, using {DEVICE} device")


def recursive_compare_print(outputs_1, outputs_2, tensor_slice=None, key=None, depth=0):
    if type(outputs_1) != type(outputs_2):
        raise ValueError(f"outputs_1: {type(outputs_1)} vs outputs_2: {type(outputs_2)}")
    elif isinstance(outputs_1, torch.Tensor):
        if tensor_slice is None:
            tensor_slice = slice(None)
        if len(outputs_1.shape) == 0:
            print(
                "\t" * depth
                + f"diff of {key} (shape={outputs_1.shape}): {torch.max(torch.abs(outputs_1 - outputs_2))}"
            )
        else:
            print(
                "\t" * depth
                + f"diff of {key} (shape={outputs_1.shape}): {torch.max(torch.abs(outputs_1[tensor_slice] - outputs_2[tensor_slice]))}"
            )
    elif isinstance(outputs_1, Mapping):
        print("\t" * depth + f"Mapping {key} (type {type(outputs_1)}):")
        for k in outputs_1:
            recursive_compare_print(outputs_1[k], outputs_2[k], tensor_slice=tensor_slice, key=k, depth=depth + 1)
    elif isinstance(outputs_1, Sequence):
        print("\t" * depth + f"Sequence {key} (type {type(outputs_1)}):")
        for output_1, output_2 in zip(outputs_1, outputs_2):
            recursive_compare_print(output_1, output_2, tensor_slice=tensor_slice, depth=depth + 1)
    else:
        print("\t" * depth + f"Unexpected type with {k}: {type(outputs_1)}")


recursive_compare_print(single_sample_outputs, batch_sample_outputs, slice(0, 1))

print("end")
exit()

"""
Model: Salesforce/blip-image-captioning-base with BlipForConditionalGeneration, using cpu device
Mapping: (type <class 'transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput'>)
        diff of loss (shape=torch.Size([])): 0.0
        diff of decoder_logits (shape=torch.Size([1, 17, 30524])): 1.049041748046875e-05
        diff of image_embeds (shape=torch.Size([1, 577, 768])): 0.0
        diff of last_hidden_state (shape=torch.Size([1, 577, 768])): 0.0

Model: Salesforce/blip-image-captioning-large with BlipForConditionalGeneration, using cpu device
Mapping: (type <class 'transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput'>)
        diff of loss (shape=torch.Size([])): 0.0
        diff of decoder_logits (shape=torch.Size([1, 17, 30524])): 8.106231689453125e-06
        diff of image_embeds (shape=torch.Size([1, 577, 1024])): 0.0
        diff of last_hidden_state (shape=torch.Size([1, 577, 1024])): 0.0

Model: Salesforce/blip2-opt-2.7b with Blip2ForConditionalGeneration, using cpu device
Mapping None (type <class 'transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput'>):
        diff of loss (shape=torch.Size([])): 9.5367431640625e-07
        diff of logits (shape=torch.Size([1, 17, 50272])): 2.9087066650390625e-05
        Mapping vision_outputs (type <class 'transformers.modeling_outputs.BaseModelOutputWithPooling'>):
                diff of last_hidden_state (shape=torch.Size([1, 257, 1408])): 0.0
                diff of pooler_output (shape=torch.Size([1, 1408])): 0.0
        Mapping qformer_outputs (type <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>):
                diff of last_hidden_state (shape=torch.Size([1, 32, 768])): 0.0
                diff of pooler_output (shape=torch.Size([1, 768])): 0.0
        Mapping language_model_outputs (type <class 'transformers.modeling_outputs.CausalLMOutputWithPast'>):
                diff of logits (shape=torch.Size([1, 49, 50272])): 2.9087066650390625e-05
                Sequence past_key_values (type <class 'tuple'>):
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.0265579223632812e-06
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 8.344650268554688e-07
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.52587890625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 8.344650268554688e-07
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.9073486328125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.2516975402832031e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.1444091796875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.1082738637924194e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.52587890625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.7136335372924805e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.71661376953125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.481524229049683e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.1920928955078125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.783272743225098e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.6391277313232422e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.351139068603516e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.002716064453125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.559755325317383e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.4781951904296875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 6.377696990966797e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.6689300537109375e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.662441253662109e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.7881393432617188e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 7.286667823791504e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.52587890625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 6.77257776260376e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.6450881958007812e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 8.031725883483887e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.8835067749023438e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.0907649993896484e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.0265579223632812e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.3083219528198242e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.7418136596679688e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.1146068572998047e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.1219253540039062e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.1205673217773438e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.71661376953125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.1801719665527344e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.5020370483398438e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.2040138244628906e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.8596649169921875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.1682510375976562e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.3887882232666016e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.4007091522216797e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.5497207641601562e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.5869736671447754e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.33514404296875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.9691884517669678e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.3828277587890625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.6570091247558594e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.430511474609375e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.3245811462402344e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.2576580047607422e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.276897430419922e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.430511474609375e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.014636993408203e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.52587890625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.609325408935547e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.0251998901367188e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.3947486877441406e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.0967254638671875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.6450881958007812e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.0609626770019531e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.3172626495361328e-05

Model: Salesforce/blip-image-captioning-base with BlipForConditionalGeneration, using cuda device
Mapping: (type <class 'transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput'>)
        diff of loss (shape=torch.Size([])): 7.62939453125e-06
        diff of decoder_logits (shape=torch.Size([1, 17, 30524])): 0.0015845298767089844
        diff of image_embeds (shape=torch.Size([1, 577, 768])): 0.19360780715942383
        diff of last_hidden_state (shape=torch.Size([1, 577, 768])): 0.19360780715942383

Model: Salesforce/blip-image-captioning-large with BlipForConditionalGeneration, using cuda device
Mapping: (type <class 'transformers.models.blip.modeling_blip.BlipForConditionalGenerationModelOutput'>)
        diff of loss (shape=torch.Size([])): 3.0517578125e-05
        diff of decoder_logits (shape=torch.Size([1, 17, 30524])): 0.0016885846853256226
        diff of image_embeds (shape=torch.Size([1, 577, 1024])): 0.1644446849822998
        diff of last_hidden_state (shape=torch.Size([1, 577, 1024])): 0.1644446849822998

Model: Salesforce/blip2-opt-2.7b with Blip2ForConditionalGeneration, using cuda device
Mapping None (type <class 'transformers.models.blip_2.modeling_blip_2.Blip2ForConditionalGenerationModelOutput'>):
        diff of loss (shape=torch.Size([])): 1.1444091796875e-05
        diff of logits (shape=torch.Size([1, 17, 50272])): 0.0001537799835205078
        Mapping vision_outputs (type <class 'transformers.modeling_outputs.BaseModelOutputWithPooling'>):
                diff of last_hidden_state (shape=torch.Size([1, 257, 1408])): 0.00011777877807617188
                diff of pooler_output (shape=torch.Size([1, 1408])): 4.231929779052734e-06
        Mapping qformer_outputs (type <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>):
                diff of last_hidden_state (shape=torch.Size([1, 32, 768])): 4.0531158447265625e-06
                diff of pooler_output (shape=torch.Size([1, 768])): 8.493661880493164e-07
        Mapping language_model_outputs (type <class 'transformers.modeling_outputs.CausalLMOutputWithPast'>):
                diff of logits (shape=torch.Size([1, 49, 50272])): 0.0001537799835205078
                Sequence past_key_values (type <class 'tuple'>):
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.245208740234375e-06
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.6093254089355469e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.4781951904296875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.384185791015625e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.574920654296875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 9.059906005859375e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.09808349609375e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 6.67572021484375e-06
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.1856040954589844e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.682209014892578e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.100799560546875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.4483928680419922e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.765655517578125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.5556812286376953e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.24249267578125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.4616718292236328e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.1948089599609375e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.0623207092285156e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 6.29425048828125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.3392181396484375e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.7670135498046875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 1.990795135498047e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.100799560546875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.5272369384765625e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.910064697265625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.390146255493164e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.00543212890625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.2292137145996094e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.649162292480469e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.5480985641479492e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.935264587402344e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.613663673400879e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.817413330078125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 2.8073787689208984e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.887580871582031e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.349781036376953e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.404783248901367e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.57763671875e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.887580871582031e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.9637088775634766e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.863739013671875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.892183303833008e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.953145980834961e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.4226646423339844e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.14984130859375e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.066394805908203e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.6253204345703125e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.51207160949707e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.8160552978515625e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.9604644775390625e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.297494888305664e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.9591064453125e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.029273986816406e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.316734313964844e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.601478576660156e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.207955837249756e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.824995994567871e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 5.507469177246094e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 3.981590270996094e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 6.246566772460938e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.57763671875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 6.079673767089844e-05
                        Sequence None (type <class 'tuple'>):
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 4.57763671875e-05
                                diff of None (shape=torch.Size([1, 32, 49, 80])): 6.61611557006836e-05
"""
