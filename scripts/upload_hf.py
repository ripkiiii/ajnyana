"""Upload Ajnyana corpus + model to HuggingFace."""
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 1. Corpus dataset ─────────────────────────────────────────────────────────
CORPUS_REPO = "ripkiiiii/ajnyana-corpus"
print(f"\n=== Uploading corpus → {CORPUS_REPO} ===")

create_repo(CORPUS_REPO, repo_type="dataset", exist_ok=True, private=False)

CORPUS_CARD = """\
---
language:
- su
license: cc-by-4.0
tags:
- sundanese
- corpus
- nlp
- text-generation
pretty_name: Ajnyana Sundanese Corpus
size_categories:
- 100M<n<1B
---

# Ajnyana Sundanese Corpus

Cleaned Sundanese text corpus built for training [Ajnyana-1](https://huggingface.co/ripkiiiii/ajnyana-1), a small language model from scratch.

## Stats

| Split | Docs | Tokens (approx) |
|-------|------|-----------------|
| train | 261,066 | ~120M |
| val   | 5,328   | ~2.4M |
| **total** | **266,394** | **~122M** |

## Sources

| Source | Docs | Notes |
|--------|------|-------|
| Wikipedia Sundanese (wikimedia/wikipedia 20231101.su) | 30,105 | 48.9% kept after QC |
| CC-100 Sundanese (statmt/cc100) | 238,988 | 85.1% kept, noisy SEO filtered |

## Format

Plain text, one document per line. Documents separated by newline. Tokenized with BPE 16K vocab (see [ajnyana-1](https://huggingface.co/ripkiiiii/ajnyana-1) for tokenizer).

## Usage

```python
with open("train.txt") as f:
    docs = [line.strip() for line in f if line.strip()]
```

## Citation

Built by [Deflated](https://deflated.xyz) — indie Indonesian AI studio.
"""

with open("/tmp/corpus_README.md", "w") as f:
    f.write(CORPUS_CARD)

api.upload_file(path_or_fileobj="/tmp/corpus_README.md",
                path_in_repo="README.md",
                repo_id=CORPUS_REPO, repo_type="dataset")

for fname in ["train.txt", "val.txt", "stats.json"]:
    local = os.path.join(BASE, "data", "processed", fname)
    if os.path.exists(local):
        print(f"  Uploading {fname} ...")
        api.upload_file(path_or_fileobj=local,
                        path_in_repo=fname,
                        repo_id=CORPUS_REPO, repo_type="dataset")
    else:
        print(f"  SKIP {fname} (not found)")

print(f"Corpus uploaded → https://huggingface.co/datasets/{CORPUS_REPO}")

# ── 2. Model ──────────────────────────────────────────────────────────────────
MODEL_REPO = "ripkiiiii/ajnyana-1"
print(f"\n=== Uploading model → {MODEL_REPO} ===")

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

# Load tokenizer (from repo)
tokenizer = ByteLevelBPETokenizer("vocab.json", "merges.txt")

# Load model
ckpt = torch.load("latest.pt", map_location="cpu", weights_only=False)
# ... (see github.com/ripkiiii/ajnyana for full inference code)
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

api.upload_file(path_or_fileobj="/tmp/model_README.md",
                path_in_repo="README.md",
                repo_id=MODEL_REPO, repo_type="model")

# Upload checkpoint
print("  Uploading latest.pt ...")
api.upload_file(path_or_fileobj=os.path.join(BASE, "checkpoints", "latest.pt"),
                path_in_repo="latest.pt",
                repo_id=MODEL_REPO, repo_type="model")

# Upload tokenizer
for fname in ["vocab.json", "merges.txt", "config.json"]:
    local = os.path.join(BASE, "tokenizer", fname)
    if os.path.exists(local):
        print(f"  Uploading tokenizer/{fname} ...")
        api.upload_file(path_or_fileobj=local,
                        path_in_repo=f"tokenizer/{fname}",
                        repo_id=MODEL_REPO, repo_type="model")

print(f"Model uploaded → https://huggingface.co/ripkiiiii/ajnyana-1")
print("\nDone!")
