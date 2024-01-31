# https://huggingface.co/stabilityai/stablelm-3b-4e1t

import os
import dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)
cache_dir = ".model.cache"
# NOTE(xiaoke): load sas_key from .env for huggingface model downloading.
dotenv.load_dotenv(".env")
use_auth_token = os.getenv("USE_AUTH_TOKEN", False)

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stablelm-3b-4e1t",
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-3b-4e1t",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto",
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)

inputs = tokenizer("The weather is always wonderful", return_tensors="pt").to(model.device)
tokens = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.75,
    top_p=0.95,
    do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))


# https://huggingface.co/stabilityai/stablelm-zephyr-3b
# - modle after SFT and RLAIF
# - the tokenizer is update from `GPTNeoXTokenizer`
# - Need the latest version of transformers.

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)
cache_dir = ".model.cache"
# NOTE(xiaoke): load sas_key from .env for huggingface model downloading.
dotenv.load_dotenv(".env")
use_auth_token = os.getenv("USE_AUTH_TOKEN", False)

tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stablelm-zephyr-3b",
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-zephyr-3b",
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)

prompt = [{"role": "user", "content": 'List 3 synonyms for the word "tiny"'}]
inputs = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")

tokens = model.generate(inputs.to(model.device), max_new_tokens=1024, temperature=0.8, do_sample=True)

print(tokenizer.decode(tokens[0], skip_special_tokens=False))


# https://huggingface.co/microsoft/phi-2

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)
cache_dir = ".model.cache"
# NOTE(xiaoke): load sas_key from .env for huggingface model downloading.
dotenv.load_dotenv(".env")
use_auth_token = os.getenv("USE_AUTH_TOKEN", False)


tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto",
    cache_dir=cache_dir,
    use_auth_token=use_auth_token,
)

inputs = tokenizer(
    '''def print_prime(n):
   """
   Print all primes between 1 and n
   """''',
    return_tensors="pt",
    return_attention_mask=False,
).to(model.device)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
