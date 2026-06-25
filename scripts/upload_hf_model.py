import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_REPO = "ripkiiiii/ajnyana-1"

create_repo(MODEL_REPO, repo_type="model", exist_ok=True, private=False)

MODEL_CARD = """\
---
language:
- su
license: mit
tags:
- sundanese
- causal-lm
- language-model
- from-scratch
- nanogpt
---

# Ajnyana-1

First Sundanese-only language model trained from scratch. Part of the [Deflated](https://deflated.xyz) Indonesian AI stack.

> "Bringing ancient Sundanese wisdom into modern NLP."

## Model Details

| | |
|---|---|
| **Architecture** | nanoGPT (Karpathy-style) |
| **Params** | 8.95M |
| **Vocab** | BPE 16K (Sundanese-specific) |
| **Context** | 512 tokens |
| **Layers** | 6 |
| **Dim** | 256 |
| **Heads** | 4 |

## Training

| | |
|---|---|
| **Dataset** | [ajnyana-corpus](https://huggingface.co/datasets/ripkiiiii/ajnyana-corpus) |
| **Tokens** | ~122M |
| **Steps** | 10,000 |
| **Hardware** | Kaggle T4 (~83 min) |
| **Optimizer** | AdamW, cosine LR 1e-3 → 1e-4, warmup 500 |
| **Final val loss** | 3.7538 |
| **Perplexity** | 52.20 |

## Usage

```python
import torch
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")
ckpt = torch.load("latest.pt", map_location="cpu", weights_only=False)
# See github.com/ripkiiii/ajnyana for full inference code
```

## Limitations

- 9M params — syntactically valid Sundanese but semantically incoherent
- CC-100 source is noisy (SEO spam partially filtered)
- No instruction tuning

## Links

- Corpus: [ripkiiiii/ajnyana-corpus](https://huggingface.co/datasets/ripkiiiii/ajnyana-corpus)
- Code: [github.com/ripkiiii/ajnyana](https://github.com/ripkiiii/ajnyana)
- Blog: [deflated.xyz](https://deflated.xyz)
"""

with open("/tmp/model_README.md", "w") as f:
    f.write(MODEL_CARD)

print("Uploading README...")
api.upload_file(path_or_fileobj="/tmp/model_README.md",
                path_in_repo="README.md",
                repo_id=MODEL_REPO, repo_type="model")

print("Uploading latest.pt ...")
api.upload_file(path_or_fileobj=os.path.join(BASE, "checkpoints", "latest.pt"),
                path_in_repo="latest.pt",
                repo_id=MODEL_REPO, repo_type="model")

for fname in ["vocab.json", "merges.txt", "config.json"]:
    local = os.path.join(BASE, "tokenizer", fname)
    if os.path.exists(local):
        print(f"Uploading tokenizer/{fname} ...")
        api.upload_file(path_or_fileobj=local,
                        path_in_repo=f"tokenizer/{fname}",
                        repo_id=MODEL_REPO, repo_type="model")

print(f"\nModel uploaded → https://huggingface.co/ripkiiiii/ajnyana-1")
