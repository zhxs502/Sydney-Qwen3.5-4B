import json, sys, random

# 三合一：新中文翻译 + 旧中文语料 + 英文原文
# 用法: python mix_all.py <new_zh> <old_zh> <en> <out>

new_p, old_p, en_p, dst = sys.argv[1:5]
random.seed(42)


def load(p, tag):
	out = []
	with open(p, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			o = json.loads(line)
			msgs = o.get("messages") or []
			# 清洗：去空、role 交替、首条 user
			msgs = [m for m in msgs if (m.get("content") or "").strip()]
			while msgs and msgs[0].get("role") != "user":
				msgs = msgs[1:]
			if len(msgs) < 2:
				continue
			bad = False
			for i in range(1, len(msgs)):
				if msgs[i].get("role") == msgs[i - 1].get("role"):
					bad = True
					break
			if bad:
				continue
			out.append({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]})
	print(f"{tag}: {len(out)} records", flush=True)
	return out


new = load(new_p, "new_zh")
old = load(old_p, "old_zh")
en = load(en_p, "en")

mixed = new + old + en
random.shuffle(mixed)

with open(dst, "w", encoding="utf-8") as f:
	for r in mixed:
		f.write(json.dumps(r, ensure_ascii=False) + "\n")

tt = sum(len(r["messages"]) for r in mixed)
tc = sum(len(m["content"]) for r in mixed for m in r["messages"])
zh_c = sum(len(m["content"]) for r in (new + old) for m in r["messages"])
en_c = sum(len(m["content"]) for r in en for m in r["messages"])
print(f"mixed: {len(mixed)} records, {tt} turns, {tc} chars")
print(f"  zh chars: {zh_c} ({zh_c*100//tc}%)")
print(f"  en chars: {en_c} ({en_c*100//tc}%)")
print(f"-> {dst}")
