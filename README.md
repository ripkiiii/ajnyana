# Ajnyana

**Ajnyana** — First dedicated Sundanese corpus and small language model, trained from scratch.

> *Ajnyana: wisdom in ancient Sundanese*

Built by [Deflated](https://deflated.xyz) · Karpathy-style · ~20M params · nanoGPT architecture

---

## What is this?

Sundanese has ~40 million speakers but nearly zero dedicated NLP resources. Ajnyana fills that gap:

1. **Ajnyana Corpus** — Open Sundanese text corpus (~60-100M tokens), published on HuggingFace
2. **Ajnyana-1** — Small language model (~20M params) trained from scratch on the corpus

## Architecture

nanoGPT-style (Karpathy) decoder-only transformer:
- ~20M parameters
- 6 layers, 8 heads, dim=512
- Standard MHA, LayerNorm, GELU
- Learned positional embeddings
- Vocab: 16K BPE (Sundanese)

## Data Sources

- Wikipedia Sunda (su.wikipedia) — ~62K articles
- CC-100 Sundanese
- OSCAR Sundanese

## Structure

```
ajnyana/
├── docs/          ← 01-05 + decisions.md
├── notebooks/     ← data pipeline, training, eval
├── scripts/       ← corpus scripts
├── tokenizer/     ← BPE tokenizer
└── data/          ← (gitignored, too large)
```

## Status

| Phase | Status |
|-------|--------|
| Corpus collection | ⏳ |
| Tokenizer | ⏳ |
| Architecture | ⏳ |
| Pre-training | ⏳ |
| Eval | ⏳ |
| Release | ⏳ |

## HuggingFace

- Corpus: `ripkiiiii/ajnyana-corpus` *(coming soon)*
- Model: `ripkiiiii/ajnyana-1` *(coming soon)*

## License

CC BY 4.0 — corpus and model weights are open.
