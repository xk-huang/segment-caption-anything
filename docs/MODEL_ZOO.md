## Model Zoo

The checkpoints are hosted on Hugging Face:

| Method                          | C     | M    | S    | B@1  | B@2  | B@3  | B@4  | R    | Noun | Verb | Noun (F) | Verb (F) |  
|---------------------------------|-------|------|------|------|------|------|------|------|------|------|----------|----------|  
| [SCA (GPT2-large, VG)](https://huggingface.co/xk-huang/segment-caption-anything-gpt2_large-vg/tree/main)           | 148.8 | 17.4 | 31.2 | 38.0 | 23.9 | 16.6 | 12.1 | 35.5 | 41.5 | 4.8  | 65.0     | 7.6      |  
| [SCA (LLAMA-3B, VG)](https://huggingface.co/xk-huang/segment-caption-anything-ollm3bv2-vg/tree/main)             | 149.8 | 17.4 | 31.3 | 38.0 | 23.9 | 16.7 | 12.2 | 35.5 | 41.2 | 4.5  | 64.6     | 7.1      |  
| [SCA (GPT2-large, Pretrain+VG)](https://huggingface.co/xk-huang/segment-caption-anything-gpt2_large-pt_vg/tree/main)  | 149.8 | 17.5 | 31.4 | 38.2 | 24.1 | 16.8 | 12.2 | 35.7 | 41.7 | 4.8  | 65.1     | 7.5      |  

We also upload the ["Pretrain" weights](https://huggingface.co/xk-huang/segment-caption-anything-gpt2_large-pt/tree/main) for "SCA (GPT2-large, Pretrain+VG)".