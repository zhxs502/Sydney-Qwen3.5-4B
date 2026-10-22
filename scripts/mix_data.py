import json, sys, random

old_p, new_p, dst = sys.argv[1], sys.argv[2], sys.argv[3]
random.seed(42)


def load(p):
	out = []
	with open(p, encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			o = json.loads(line)
			msgs = o.get("messages") or []
			if len(msgs) >= 2:
				out.append({"messages": msgs})
	return out


old = load(old_p)
new = load(new_p)
print(f"old={len(old)} new={len(new)}")

# 新数据全要；旧数据抽 2x（新:旧 = 1:2）
k = min(len(old), len(new) * 2)
old_s = random.sample(old, k)
mixed = old_s + new
random.shuffle(mixed)

with open(dst, "w", encoding="utf-8") as f:
	for r in mixed:
		f.write(json.dumps(r, ensure_ascii=False) + "\n")

t = sum(len(r["messages"]) for r in mixed)
c = sum(len(m["content"]) for r in mixed for m in r["messages"])
print(f"mixed={len(mixed)} turns={t} chars={c} -> {dst}")
