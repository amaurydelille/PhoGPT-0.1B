# PhoGPT-0.1B
English-Vietnamese translator made from a boring afternoon, but it looked cool pushed it.

Simple Transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) with Mixture-of-Experts and RoPE [(Su et al., 2021)](https://arxiv.org/abs/2104.09864) of 130M parameters, trained on [VinAI/PhoMT](https://huggingface.co/datasets/vinai/PhoMT) dataset (3M pairs of sentences).

Please find the weights on the associated [HuggingFace repo](https://huggingface.co/amaury-delille/PhoGPT-20M).

## Demo
![Demo](assets/demo.gif)

## To run the code
```python
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
import sentencepiece as spm
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("amaury-delille/phogpt-0.13b", trust_remote_code=True)

en_spm_path = hf_hub_download("amaury-delille/phogpt-0.13b", "tokenizer_en/spm.model")
vi_spm_path = hf_hub_download("amaury-delille/phogpt-0.13b", "tokenizer_vi/spm.model")

en_tokenizer = spm.SentencePieceProcessor(en_spm_path)
vi_tokenizer = spm.SentencePieceProcessor(vi_spm_path)

text = "Hello, how are you?"
encoded = en_tokenizer.Encode(text)
encoded.append(3)
max_len = 128
encoded = encoded + [0] * (max_len - len(encoded))
input_ids = torch.tensor([encoded[:max_len]], dtype=torch.long)

model.eval()
outputs = model.generate(input_ids, max_length=128)

out_tokens = [t for t in outputs[0].tolist() if t not in [0, 2, 3]]
translation = vi_tokenizer.Decode(out_tokens)
print(f"English: {text}")
print(f"Vietnamese: {translation}")
```

## Citations
```bibtex
@inproceedings{vaswani2017attention,
title     = {Attention is All You Need},
author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
booktitle = {Advances in Neural Information Processing Systems},
year      = {2017}
}

@article{su2021roformer,
title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
author  = {Su, Jianlin and Lu, Yu and Pan, Shengfeng and Murtadha, Ahmed and Wen, Bo and Liu, Yunfeng},
journal = {arXiv preprint arXiv:2104.09864},
year    = {2021}
}

@inproceedings{PhoMT,
title     = {{PhoMT: A High-Quality and Large-Scale Benchmark Dataset for Vietnamese-English Machine Translation}},
author    = {Long Doan and Linh The Nguyen and Nguyen Luong Tran and Thai Hoang and Dat Quoc Nguyen},
booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
year      = {2021}
}
```
