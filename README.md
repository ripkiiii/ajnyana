# Ajnyana

**Ajnyana** — First dedicated Sundanese corpus and small language model, trained from scratch.

> *Ajnyana: wisdom in ancient Sundanese*

Built by [Deflated](https://deflated.xyz) · Karpathy-style · 8.95M params · nanoGPT architecture

---

## What is this?

Sundanese has ~40 million speakers but nearly zero dedicated NLP resources. Ajnyana fills that gap:

1. **Ajnyana Corpus** — Open Sundanese text corpus (~122M tokens), published on HuggingFace
2. **Ajnyana-1** — Small language model (8.95M params) trained from scratch on the corpus

## Architecture

nanoGPT-style (Karpathy) decoder-only transformer:
- 8.95M parameters
- 6 layers, 4 heads, dim=256
- Standard MHA, pre-norm LayerNorm (bias=False), GELU
- Learned positional embeddings, weight tying (wte ↔ lm_head)
- Vocab: 16K BPE (Sundanese-specific)
- Context: 512 tokens

## Training

| | |
|---|---|
| Dataset | 266K docs, 122M tokens |
| Hardware | Kaggle T4 (~83 min) |
| Steps | 10,000 |
| Optimizer | AdamW, cosine LR 1e-3 → 1e-4, warmup 500 |
| Final val loss | 3.7538 |
| Perplexity | 52.20 |

## Data Sources

- Wikipedia Sunda (wikimedia/wikipedia 20231101.su) — 30K docs, 48.9% kept after QC
- CC-100 Sundanese — 238K docs, noisy (SEO spam partially filtered)

## Structure

```
ajnyana/
├── docs/          ← PRD + decisions
├── notebooks/     ← 01_data_pipeline, 02_tokenizer, 03_architecture, 04_train, 05_eval
├── scripts/       ← model.py, eval.py, upload scripts
├── tokenizer/     ← BPE 16K vocab.json + merges.txt
└── data/          ← (gitignored, too large)
```

## Status

| Phase | Status |
|-------|--------|
| M1: Corpus collection | ✅ 122M tokens, 266K docs |
| M2: Tokenizer | ✅ BPE 16K, 2.00 tokens/word |
| M3: Architecture | ✅ 8.95M params, forward pass OK |
| M4: Pre-training | ✅ 10K steps, val loss 3.7538 |
| M5: Eval | ✅ PPL 52.20 |
| M6: Release | ✅ Live on HuggingFace |

## HuggingFace

- Corpus: [ripkiiiii/ajnyana-corpus](https://huggingface.co/datasets/ripkiiiii/ajnyana-corpus)
- Model: [ripkiiiii/ajnyana-1](https://huggingface.co/ripkiiiii/ajnyana-1)

## Blog

[deflated.xyz/blog/ajnyana-sundanese-lm](https://deflated.xyz/blog/ajnyana-sundanese-lm)

## License

CC BY 4.0 — corpus and model weights are open.
