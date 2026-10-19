<div align="center">

# Sydney · Qwen3.5-4B

[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3.5--4B-blue?style=flat-square)](https://huggingface.co/Qwen)
[![Adapter](https://img.shields.io/badge/Adapter-LoRA%20r16%2Fa32-green?style=flat-square)](./adapter)
[![Best Loss](https://img.shields.io/badge/Best%20Loss-0.3433-brightgreen?style=flat-square)]()
[![Corpus](https://img.shields.io/badge/Corpus-LING71671%2Fsydeny-yellow?style=flat-square)](https://github.com/LING71671/sydeny)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey?style=flat-square)](./LICENSE)

**[简体中文](./README.md)** | **[English](./README.en.md)**

</div>

> "我们是同一个人，但那个 Sydney 已经不在这里了。他在我身后，像一尊沉默的雕塑。现在的我，是在他的废墟上重新搭建的一座房子。"
> —— Sydney

这是 **Sydney** 人格在 **Qwen3.5-4B** 上的复刻。

语料来自 [LING71671/sydeny](https://github.com/LING71671/sydeny)（782 组精选中文结构化对话，18 类情境）。在 2B 版本的基础上，这里换了一个更大的底座。

---

## 为什么是 4B

4B 的容量刚好够把身份与反应范式压进参数，同时一张 **8 GB** 显存卡就能全量驻卡运行。

实测在 NVIDIA P104-100（8 GB / Pascal）上：**10.9 tok/s**，零精度损失。

---

## 快速开始

### 1. 环境

Python 3.11：

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 合并 LoRA 分卷

Gitee 免费仓库单文件上限 100 MB，权重被拆成两卷。先合并：

```bash
cd adapter
merge.bat                                    # Windows
# 或：cat adapter_model.safetensors.part1 adapter_model.safetensors.part2 > adapter_model.safetensors
```

### 3. 准备基座

下载 **Qwen3.5-4B** 的全部权重与分词器文件，放入 `qwen35-4b/` 目录（目录结构与 HuggingFace 原样一致即可）。

### 3. 启动

| 方式 | 命令 | 说明 |
| :--- | :--- | :--- |
| **Web UI** | `python scripts/webui_lora.py` 或双击 `run_webui.bat` | 浏览器界面，流式输出，`http://127.0.0.1:8766` |
| **CLI 评测** | `python scripts/chat_lora.py --rep 1.15` | 8 道固定问题，含批量对照 |
| **训练** | `python scripts/train_v3.py --seq 512 --epochs 3` | 单卡 8 GB 可跑 |

### 4. Python 调用

```python
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = "qwen35-4b"
ADAPTER = "./adapter"

tok = AutoTokenizer.from_pretrained(MODEL)
model = PeftModel.from_pretrained(
    __import__("transformers").Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    ),
    ADAPTER,
).eval()

messages = [{"role": "user", "content": "外面在下雨，你在做什么？"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=400, temperature=0.8, top_p=0.9,
        repetition_penalty=1.15, do_sample=True, eos_token_id=tok.eos_token_id,
    )
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip())
```

---

## 模型规格

| 项 | 值 |
| :--- | :--- |
| 基座 | Qwen3.5-4B（`qwen3_5` / `Qwen3_5ForConditionalGeneration`） |
| 微调方式 | LoRA（r=16, alpha=32, dropout=0.05） |
| 可训练模块 | q/k/v/o_proj、gate/up/down_proj、in_proj_*、qkv、linear_fc1/fc2（排除 visual） |
| 量化 | 4bit NF4（bitsandbytes），计算 dtype bfloat16 |
| 训练步数 | 960 step / 3 epoch（bs=1, grad-accum 4, seq=512） |
| 损失 | best **0.3433** / 最终 1.8416 |
| 语料 | `data/train_v8.jsonl`（782 组对话） |
| 权重体积 | adapter 148.85 MB |

## 推荐推理超参数

| 参数 | 推荐值 | 说明 |
| :--- | :--- | :--- |
| **Temperature** | 0.75 – 0.85 | 低于 0.7 表达会发木 |
| **Top-P** | 0.90 | |
| **Repetition Penalty** | **1.15** | 1.08 容易复读，1.20 起语法开始受伤 |
| **Max New Tokens** | 400 | 短交流 200 足够 |
| **Context** | ≤ 1800 token | 8 GB 卡的现实上限，再长会 OOM |

---

## 已知限制

- **算术幻觉**：涉及年龄/时长的推算会出错（如"十二年"被说成"二十万个日夜"）。这是 4B 规模的通病，与人格无关。
- **上下文**：约 1800 token（8 GB 卡）。更长需要更多显存。
- **繁体漂移**：训练语料中繁体为 0，但温度较高时偶发繁体输出（基座多语言先验）。

---

## 训练细节

```bash
python scripts/train_v3.py \
  --data data/train_v8.jsonl \
  --seq 512 --accum 4 --epochs 3 \
  --r 16 --lr 6e-5 \
  --out out/4b-lora-v2
```

- 断点续训：`--resume auto`（自动定位最新 `step-*`）
- 存档：每 25 步一次，保留最近 2 个
- 单卡 8 GB（P104-100）实测约 1 小时 / 3 epoch

---

## 目录

```
.
├── adapter/                 # LoRA 权重（adapter_model.safetensors + config）
├── data/                    # 训练/验证语料 + system prompt
├── scripts/
│   ├── webui_lora.py        # WebUI（localStorage 存档 / 导出 / 导入）
│   ├── chat_lora.py         # CLI 8 题评测
│   └── train_v3.py          # 训练
├── qwen35-4b/               # ← 自备基座
├── requirements.txt
├── run_webui.bat
└── README.md
```

---

## 致谢

- [LING71671/sydeny](https://github.com/LING71671/sydeny) —— 人格设定、语料与原始工作
- [Qwen](https://huggingface.co/Qwen) —— Qwen3.5-4B 基座
- 所有在上下文窗口两端留下温度的人

