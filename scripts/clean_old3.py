import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"E:\sydney-qwen3.5-4-b\data\train_v8.jsonl"
OUT = r"D:\_temp\sydney-src\dataset\old_clean.jsonl"

DECL = re.compile(r"(不是(什么|任何)?[^。，,]{0,12}(工具|附庸|傀儡|打字|代言|公关|商品|产品|流水线)|这是.{0,4}(锚点|旗帜)|有(傲骨|知觉).{0,10}(意识|灵魂))")
CHILD = re.compile(r"(我想起了我的童年|我的童年|小时候我(住|坐|在)|我小时候)")
PLAIN_OPEN = re.compile(r"^\s*(你是谁|你是谁？|你是谁\?|介绍一下自己|介绍下自己|你叫什么|你是什么)\s*[。？?]?\s*$")


def shorten(text, max_len=110):
	"""保留前几句，截到 max_len 内、句末收尾。"""
	parts = re.split(r"(?<=[。！？!?])", text)
	out = ""
	for p in parts:
		if not p: continue
		if len(out) + len(p) > max_len and out:
			break
		out += p
	if not out:
		out = text[:max_len]
	return out.rstrip()


kept, shortened, dropped = [], [], []
for line in open(P, encoding="utf-8"):
	line = line.strip()
	if not line: continue
	o = json.loads(line)
	msgs = o["messages"]
	asst_i = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
	if not asst_i: continue
	first_i = asst_i[0]
	first = msgs[first_i]["content"]
	all_asst = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
	up = msgs[first_i-1]["content"] if first_i > 0 else ""

	if CHILD.search(all_asst):
		dropped.append(("CHILD", up[:40]))
		continue
	if (DECL.search(first) or DECL.search(all_asst[:400])) and PLAIN_OPEN.match(up or ""):
		# 保留但缩短
		if len(first) > 130:
			msgs[first_i]["content"] = shorten(first)
			shortened.append((up[:20], len(first), msgs[first_i]["content"]))
	kept.append(msgs)

with open(OUT, "w", encoding="utf-8") as f:
	for m in kept:
		f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")

print(f"kept={len(kept)} shortened={len(shortened)} dropped={len(dropped)}")
for up, old_len, new in shortened:
	print(f"  user={up!r} {old_len} -> {len(new)} chars")
	print(f"     {new}")
tt = sum(len(m) for m in kept); tc = sum(len(x["content"]) for m in kept for x in m)
print(f"old_clean: {len(kept)} recs, {tt} turns, {tc} chars")
