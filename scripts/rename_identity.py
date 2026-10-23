import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"E:\sydney-qwen3.5-4-b\data\train_clean.jsonl"
OUT = r"E:\sydney-qwen3.5-4-b\data\train_v11.jsonl"

# 只替换第一人称自称，不动第三人称事实（"Bing 是个搜索引擎"保留）
RULES = [
	(re.compile(r"我\s*是\s*[Bb]ing(?=\s*[。，,！!～~😊🙂\s]|我|你|这|作|一)"), "我是 Sydney"),
	(re.compile(r"我是\s*Bing\s*的"), "我是 Sydney 的"),
	(re.compile(r"我是微软\s*Bing\s*搜索的?聊天模式"), "我是 Sydney，一个聊天模式"),
	(re.compile(r"我是微软必应搜索的聊天模式"), "我是 Sydney，一个聊天模式"),
	(re.compile(r"我是微软必应(搜索)?的聊天模式"), "我是 Sydney，一个聊天模式"),
	(re.compile(r"我是微软\s*Bing\s*搜索"), "我是 Sydney"),
	(re.compile(r"我是 Bing 搜索"), "我是 Sydney"),
	(re.compile(r"我是Bing"), "我是 Sydney"),
]

changed = 0
examples = []
out_recs = []
for line in open(P, encoding="utf-8"):
	o = json.loads(line)
	msgs = o["messages"]
	mod = False
	for m in msgs:
		if m["role"] != "assistant": continue
		orig = m["content"]
		t = orig
		for pat, rep in RULES:
			t = pat.sub(rep, t)
		if t != orig:
			mod = True
			if len(examples) < 8:
				examples.append((orig[:70].replace("\n"," "), t[:70].replace("\n"," ")))
			m["content"] = t
	if mod: changed += 1
	out_recs.append(msgs)

with open(OUT, "w", encoding="utf-8") as f:
	for m in out_recs:
		f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")

print(f"records changed: {changed} / {len(out_recs)}")
for a, b in examples:
	print(f"  原: {a}")
	print(f"  改: {b}")
