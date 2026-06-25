# PRD — Ajnyana
**Version:** 1.0  
**Date:** 2026-06-25  
**Author:** Muhammad Rifky Firmansyah Sujana  
**Studio:** Deflated

---

## 1. Overview

**Ajnyana** adalah proyek NLP pertama yang membangun corpus Bahasa Sunda skala besar secara terbuka, sekaligus melatih small language model dari scratch menggunakan corpus tersebut.

Nama "Ajnyana" berasal dari Sunda kuno — bermakna kebijaksanaan. Project ini lahir dari gap nyata: 40 juta penutur Bahasa Sunda, tapi hampir zero dedicated NLP resource.

---

## 2. Problem Statement

- Bahasa Sunda adalah bahasa daerah terbesar kedua di Indonesia (~40 juta penutur)
- Tidak ada dedicated Sundanese corpus yang terbuka dan berskala besar
- Tidak ada dedicated Sundanese language model yang dilatih dari scratch
- Resource yang ada (CC-100, NusaX) sangat kecil dan fragmentasi
- Riset NLP Sunda selalu bergantung pada model multilingual yang tidak domain-spesifik

---

## 3. Goals

### Corpus (Ajnyana Corpus)
- Kumpulkan semua teks Bahasa Sunda yang tersedia secara publik
- Bersihkan dan publish sebagai open dataset di HuggingFace
- Target: 60-100M tokens (semua yang bisa dikumpulkan)

### Model (Ajnyana-1)
- Latih small language model dari scratch, Karpathy-style
- Arsitektur sederhana, reproducible, fully documented
- Target: 10-30M params (disesuaikan dengan ukuran corpus)
- Publish ke HuggingFace

### Dokumentasi
- Semua keputusan teknis tercatat di `decisions.md`
- Blog post di deflated.xyz tiap milestone
- Paper (opsional, post-release)

---

## 4. Non-Goals

- Bukan model production-grade atau chatbot
- Bukan fine-tuned dari model existing — harus dari scratch
- Bukan multilingual — Sunda-only
- Tidak perlu RLHF atau instruction tuning di v1

---

## 5. Target Users

1. **Peneliti NLP** — butuh Sundanese corpus untuk eksperimen
2. **Developer Indo** — mau fine-tune model Sunda untuk aplikasi lokal
3. **Komunitas Sunda** — dokumentasi digital bahasa Sunda
4. **Deflated** — porto AI studio, showcase Indo AI stack

---

## 6. Data Sources

| Source | Format | Est. Size | Priority |
|--------|--------|-----------|----------|
| Wikipedia Sunda (su.wikipedia) | Dump XML | ~50-80M tokens | 🔴 Utama |
| CC-100 Sundanese | HuggingFace dataset | ~10M tokens | 🔴 Utama |
| OSCAR Sundanese | HuggingFace dataset | TBD | 🟡 Sekunder |
| NusaX Sundanese | HuggingFace dataset | ~3,600 kalimat | 🟢 Tambahan |
| Situs web Sunda (TBD) | Web scraping | TBD | 🟡 Sekunder |

**Total target:** 60-100M tokens bersih setelah cleaning.

---

## 7. Model Spec — Ajnyana-1

**Arsitektur: nanoGPT style (Karpathy)** — simpel, clean, fully reproducible.

| Parameter | Value |
|-----------|-------|
| Arsitektur | GPT-2 style decoder-only |
| Params | ~20M |
| dim | 512 |
| n_layers | 6 |
| n_heads | 8 |
| ffn_dim | 2048 (4× dim) |
| vocab_size | 16K (BPE Sundanese) |
| max_seq_len | 512 |
| Positional | Learned embeddings |
| Norm | LayerNorm |
| Activation | GELU |
| Attention | Standard MHA (no GQA) |

**Kenapa nanoGPT bukan LLaMA-style:**
- Lebih simpel → lebih mudah dijelasin + direproduksi
- True Karpathy-style — story konsisten
- Data kecil (~60-100M tokens) → performa gap vs LLaMA-style tidak signifikan

**Training target:** sesuai data yang tersedia, ~1-2 sesi Kaggle.

---

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Corpus size | ≥60M tokens bersih |
| Corpus HuggingFace downloads | publish + live |
| Model perplexity | meaningful (jauh di bawah random baseline) |
| Model HuggingFace | publish + live |
| Blog post | ≥2 posts di deflated.xyz |
| Reproducibility | semua script publik di GitHub |

---

## 9. Milestones

| Fase | Deliverable | Status |
|------|-------------|--------|
| **M1: Corpus** | Ajnyana Corpus v1 di HuggingFace | ⏳ |
| **M2: Tokenizer** | BPE tokenizer Sunda 16-32K vocab | ⏳ |
| **M3: Architecture** | Model spec final + notebook verified | ⏳ |
| **M4: Training** | Ajnyana-1 trained, val loss logged | ⏳ |
| **M5: Eval** | Perplexity report + sample outputs | ⏳ |
| **M6: Release** | HuggingFace publish + blog post | ⏳ |

---

## 10. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Data terlalu sedikit (<30M tokens) | Medium | Turunkan model ke 5-10M params |
| Kualitas CC-100 Sunda rendah | High | Aggressive cleaning, filter pendek |
| Kaggle quota habis sebelum selesai | Low | Backup SageMaker Studio Lab (4h/hari) |
| Wikipedia Sunda banyak artikel stub | High | Filter artikel <200 kata |

---

## 11. Release Plan

- **Corpus:** `ripkiiiii/ajnyana-corpus` di HuggingFace (CC BY 4.0)
- **Model:** `ripkiiiii/ajnyana-1` di HuggingFace
- **GitHub:** `github.com/ripkiiii/ajnyana` (public)
- **Blog:** deflated.xyz/blog — post per milestone
- **Deflated page:** tambah Ajnyana ke /research

---

## 12. Stack

- **Data pipeline:** Python, HuggingFace datasets, Wikipedia dumps
- **Tokenizer:** HuggingFace tokenizers (BPE)
- **Model:** PyTorch, dari scratch
- **Training:** Kaggle (T4 x2) + SageMaker Studio Lab (backup)
- **Storage:** HuggingFace Hub
- **Docs:** Markdown di repo GitHub
