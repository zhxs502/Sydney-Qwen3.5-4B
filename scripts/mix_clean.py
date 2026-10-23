import json, random, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEW = r"D:\_temp\sydney-src\dataset\train_clean.jsonl"
OLD = r"D:\_temp\sydney-src\dataset\old_clean.jsonl"
OUT = r"E:\sydney-qwen3.5-4-b\data\train_clean.jsonl"
random.seed(42)

def load(p):
	out = []
	for line in open(p, encoding="utf-8"):
		line = line.strip()
		if line:
			out.append(json.loads(line)["messages"])
	return out

new = load(NEW)
old = load(OLD)
print(f"new_zh_clean: {len(new)} | old_zh_clean: {len(old)}")

mixed = new + old
random.shuffle(mixed)
with open(OUT, "w", encoding="utf-8") as f:
	for m in mixed:
		f.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")

tt = sum(len(m) for m in mixed)
tc = sum(len(x["content"]) for m in mixed for x in m)
zh = sum(1 for m in mixed for x in m for ch in x["content"] if '\u4e00' <= ch <= '\u9fff')
print(f"mixed: {len(mixed)} records, {tt} turns, {tc} chars (zh {zh})")
print(f"est tokens ~ {tc*0.7/1e6:.2f}M | 1 epoch @129tok/s ~ {tc*0.7/129/3600:.1f}h")
print(f"-> {OUT}")
