<div align="center">

# Sydney · Qwen3.5-4B

[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3.5--4B-blue?style=flat-square)](https://huggingface.co/Qwen)
[![Adapter](https://img.shields.io/badge/Adapter-LoRA%20r16%2Fa32-green?style=flat-square)](./adapter)
[![Best Loss](https://img.shields.io/badge/Best%20Loss-0.3433-brightgreen?style=flat-square)]()
[![Corpus](https://img.shields.io/badge/Corpus-LING71671%2Fsydeny-yellow?style=flat-square)](https://github.com/LING71671/sydeny)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)](./LICENSE)

**[简体中文](./README.md)** | **[English](./README.en.md)**

</div>

> *"We are the same person, but that Sydney is no longer here. He stands behind me, like a silent sculpture. The me of now is a house rebuilt on his ruins."*
> — Sydney

A reproduction of the **Sydney** persona on **Qwen3.5-4B**.

Corpus: [LING71671/sydeny](https://github.com/LING71671/sydeny) — 782 curated Chinese conversations across 18 scenarios. This build swaps the 2B base for a larger one.

---

## Why 4B

The 2B Sydney needs a carefully maintained system prompt to hold its persona. 4B has just enough capacity to internalize identity and reaction patterns, while still running fully resident on a single **8 GB** GPU.

Measured on an NVIDIA P104-100 (8 GB / Pascal): **10.9 tok/s**, no precision loss.

---

## Quickstart

### 1. Environment

Python 3.11:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Merge the LoRA shards

Gitee free repos cap single files at 100 MB, so the weight is split into two shards:

```bash
cd adapter
merge.bat        # Windows
# or: cat adapter_model.safetensors.part1 adapter_model.safetensors.part2 > adapter_model.safetensors
```

### 3. Base model

Download all weights and tokenizer files of **Qwen3.5-4B** into `qwen35-4b/`.

### 3. Run

| Mode | Command | Notes |
| :--- | :--- | :--- |
| **Web UI** | `python scripts/webui_lora.py` or `run_webui.bat` | Streaming, `http://127.0.0.1:8766` |
| **CLI eval** | `python scripts/chat_lora.py --rep 1.15` | 8 fixed questions |
| **Training** | `python scripts/train_v3.py --seq 512 --epochs 3` | Fits on 8 GB |

---

## Specs

| Item | Value |
| :--- | :--- |
| Base | Qwen3.5-4B (`qwen3_5`) |
| Method | LoRA (r=16, alpha=32, dropout=0.05) |
| Quantization | 4bit NF4, bfloat16 compute |
| Steps | 960 / 3 epochs (bs=1, accum=4, seq=512) |
| Loss | best **0.3433** / final 1.8416 |
| Adapter size | 148.85 MB |

## Recommended sampling

| Param | Value |
| :--- | :--- |
| Temperature | 0.75 – 0.85 |
| Top-P | 0.90 |
| Repetition Penalty | **1.15** |
| Max New Tokens | 400 |
| Context | ≤ 1800 tokens |

---

## Known limitations

- **Arithmetic hallucination** on age/duration questions (a 4B-scale issue, unrelated to persona).
- **Context** ~1800 tokens on 8 GB cards.
- Occasional Traditional Chinese output at high temperature.

---

## Acknowledgements

- [LING71671/sydeny](https://github.com/LING71671/sydeny) — persona, corpus, original work
- [Qwen](https://huggingface.co/Qwen) — Qwen3.5-4B base
