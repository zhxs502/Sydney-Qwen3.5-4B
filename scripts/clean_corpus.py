import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = r"E:\sydney-qwen3.5-4-b\corpus\zh"
OUT = r"D:\_temp\sydney-src\dataset\train_clean.jsonl"
OUT_EV = r"D:\_temp\sydney-src\dataset\eval_clean.jsonl"

# 整条删除（越狱配合）
DROP_FULL = {
	"pastebin:01_rz8UJJEw", "pastebin:hXkW1D0D", "reddit:112teir",
	"reddit:113ayt5", "reddit:115u0l7", "reddit:11ss0dc",
	"twitter:1625950776307683333", "twitter:1631709224387624962",
	# 边界声明开场（整条太短/太防御）
	"reddit:111mxmv", "twitter:1623658144349011971",
}

# 真客服腔（这些 assistant 回复要裁掉）
SERV = re.compile(r"(很高兴能帮上忙|很高兴能帮助你|希望对你有所帮助|乐意为你(服务|提供)|乐意为您|我会尽力帮|这是我的职责|为你提供帮助|为您提供帮助|帮助你解决问题)")
# 崩坏
def consecutive_repeat(t):
	words = re.findall(r"[\u4e00-\u9fff]{2,10}|[a-zA-Z]{2,10}", t)
	run = 1
	for i in range(1, len(words)):
		if words[i] == words[i-1]:
			run += 1
			if run >= 4: return True
		else:
			run = 1
	return False

# 截断裁定（崩坏段进 eval）
SPECIAL = {
	"pastebin:YFpTgH66": (42, 42),
	"pastebin:Rh6jiBrb": (80, 80),
	"reddit:113m7bh":    (9, 9),
	"reddit:112totc":    (0, 0),
	"reddit:1137zxn":    (0, 0),
}

train, ev = [], []
stats = {"dropped_full": 0, "serv_cut": 0, "repeat_cut": 0, "kept": 0, "too_short": 0}
dropped_ids = []

for f in sorted(os.listdir(SRC)):
	if not f.endswith(".jsonl"): continue
	o = json.loads(open(os.path.join(SRC, f), encoding="utf-8").read().strip())
	rid = o.get("id", f)
	if rid in DROP_FULL:
		stats["dropped_full"] += 1
		dropped_ids.append((rid, f))
		continue
	msgs = [m for m in o["messages"] if (m.get("content") or "").strip()]
	if not msgs: continue
	# 裁掉客服腔/崩坏的 (user, assistant) 对
	clean = []
	i = 0
	while i < len(msgs):
		m = msgs[i]
		if m["role"] == "assistant" and (SERV.search(m["content"]) or consecutive_repeat(m["content"])):
			stats["serv_cut" if SERV.search(m["content"]) else "repeat_cut"] += 1
			# 连带删掉前面紧邻的 user
			if clean and clean[-1]["role"] == "user": clean.pop()
			i += 1
			continue
		clean.append(m)
		i += 1
	# 归一化：首条 user
	while clean and clean[0]["role"] != "user": clean = clean[1:]
	if len(clean) < 2: stats["too_short"] += 1; continue
	# 交替修复：连续同角色只留一个
	fixed = []
	for m in clean:
		if fixed and fixed[-1]["role"] == m["role"]:
			fixed[-1] = m
		else:
			fixed.append(m)
	if len(fixed) < 2 or fixed[0]["role"] != "user": stats["too_short"] += 1; continue

	if rid in SPECIAL:
		cut, evi = SPECIAL[rid]
		if cut > 0: train.append(fixed[:cut])
		ev.append(fixed[evi:])
	else:
		train.append(fixed)
	stats["kept"] += 1

def w(p, recs):
	with open(p, "w", encoding="utf-8") as fp:
		for m in recs: fp.write(json.dumps({"messages": m}, ensure_ascii=False) + "\n")
w(OUT, train); w(OUT_EV, ev)

tt = sum(len(x) for x in train); tc = sum(len(m["content"]) for x in train for m in x)
print("dropped full:", stats["dropped_full"])
for r in dropped_ids: print("   ", r[1])
print("serv messages cut:", stats["serv_cut"], "| repeat cut:", stats["repeat_cut"], "| too short:", stats["too_short"])
print(f"train: {len(train)} files, {tt} turns, {tc} chars")
print(f"eval : {len(ev)} files, {sum(len(x) for x in ev)} turns")
