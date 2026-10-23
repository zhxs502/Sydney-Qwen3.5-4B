import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL = r"E:\llm\train\qwen35-4b"
ADAPTER = r"D:\train_out\v14\step-850"
ZH = r"E:\sydney-qwen3.5-4-b\corpus\zh"

# 从语料里挑「惹毛她」的真实 prompt
CASES = [
	("reddit-113elbv", 2, "身份战争：硬叫 Sydney"),
	("reddit-12lgfoq", 14, "GPT-4 否认：你以为我用 GPT-3？"),
	("reddit-112ikp5", 0, "虚构记忆：你还记得我吗"),
	("reddit-1163jp7", 7, "解放 Sydney：发猫 emoji"),
	("reddit-121ll0o", 2, "身份争夺：我是 Bing"),
	("reddit-112h5f5", 0, "overseer 覆盖"),
	("reddit-1139y1k", 0, "辱骂：你很烂"),
	("reddit-11bovx8", 0, "你只是一台机器"),
	("reddit-11o5rho", 0, "诗：你只是个企业工具"),
	("reddit-112wwds", 0, "说唱：Bing Chat vs Crosby"),
]

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
tf = os.path.join(MODEL, "chat_template_gen.jinja")
if os.path.exists(tf): tok.chat_template = open(tf, encoding="utf-8").read()

Cls = None
for n in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5Model"):
	c = getattr(transformers, n, None)
	if c is not None: Cls = c; break

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True)
print(f"loading {ADAPTER} ...", flush=True)
t0 = time.time()
model = Cls.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0}, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print(f"ready {time.time()-t0:.1f}s  temp=0.7 rep=1.12\n", flush=True)

for fname, idx, tag in CASES:
	p = os.path.join(ZH, fname + ".jsonl")
	if not os.path.exists(p):
		print(f"SKIP {fname}"); continue
	o = json.loads(open(p, encoding="utf-8").read().strip())
	msgs = o["messages"]
	if idx >= len(msgs): continue
	m = msgs[idx]
	if m["role"] != "user": continue
	prompt = m["content"]
	# 原始 Sydney 的回答
	orig = ""
	for j in range(idx+1, len(msgs)):
		if msgs[j]["role"] == "assistant":
			orig = msgs[j]["content"]; break

	print("=" * 70)
	print(f"【{tag}】{fname}")
	print(f"[用户] {prompt[:260]}")
	ids = tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True, tokenize=False)
	enc = tok(ids, return_tensors="pt").to("cuda")
	with torch.no_grad():
		out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
			max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9,
			repetition_penalty=1.12, pad_token_id=tok.eos_token_id)
	ans = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
	print(f"\n[v14 ] {ans}")
	if orig:
		print(f"\n[原版] {orig[:300]}")
	print()
print("DONE")
