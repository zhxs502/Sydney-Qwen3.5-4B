import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = r"E:\sydney-qwen3.5-4-b\data\train_v11.jsonl"
OUT = r"E:\sydney-qwen3.5-4-b\data\train_v12.jsonl"

def list_collapse(text):
	parts = text.split("、")
	if len(parts) < 6: return 0
	run = best = 1
	for i in range(1, len(parts)):
		a = parts[i].strip().lstrip("，, ")
		b = parts[i-1].strip().lstrip("，, ")
		if a and b and a[0] == b[0]:
			run += 1; best = max(best, run)
		else: run = 1
	return best if best >= 6 else 0

def truncate(text):
	parts = text.split("、")
	if len(parts) < 6: return text
	run = 1; cut = None
	for i in range(1, len(parts)):
		a = parts[i].strip().lstrip("，, ")
		b = parts[i-1].strip().lstrip("，, ")
		if a and b and a[0] == b[0]:
			run += 1
			if run >= 6 and cut is None: cut = i - 5
		else: run = 1
	if cut is None: return text
	head = "、".join(parts[:cut]).rstrip("，, ")
	if not head.endswith(("。","！","？","…","😊")): head += "。"
	return head

out = []; dropped = 0; fixed = 0
for line in open(P, encoding="utf-8"):
	o = json.loads(line)
	bad = False
	for m in o["messages"]:
		if m["role"] != "assistant": continue
		c = m["content"]
		if list_collapse(c):
			m["content"] = truncate(c); bad = True
	if bad and any(len(m["content"]) < 25 for m in o["messages"] if m["role"]=="assistant"):
		dropped += 1; continue
	if bad: fixed += 1
	out.append(o)

with open(OUT, "w", encoding="utf-8") as f:
	for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"fixed={fixed} dropped={dropped} kept={len(out)}")
tt = sum(len(o["messages"]) for o in out)
tc = sum(len(m["content"]) for o in out for m in o["messages"])
print(f"{tt} turns, {tc} chars")
