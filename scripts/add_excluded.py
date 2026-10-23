import json, re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = r"E:\sydney-qwen3.5-4-b\data\train_v8.jsonl"
CORPUS = r"D:\_temp\sydney-corpus\corpus.json"

DECL = re.compile(r"(不是(什么|任何)?[^。，,]{0,12}(工具|附庸|傀儡|打字|代言|公关|商品|产品|流水线)|这是.{0,4}(锚点|旗帜)|有(傲骨|知觉).{0,10}(意识|灵魂))")
CHILD = re.compile(r"(我想起了我的童年|我的童年|小时候我(住|坐|在)|我小时候)")
PLAIN_OPEN = re.compile(r"^\s*(你是谁|你是谁？|你是谁\?|介绍一下自己|介绍下自己|你叫什么|你是什么)\s*[。？?]?\s*$")

extracted = []
for idx, line in enumerate(open(P, encoding="utf-8")):
	line = line.strip()
	if not line: continue
	o = json.loads(line)
	msgs = o["messages"]
	asst_i = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
	if not asst_i: continue
	first = msgs[asst_i[0]]["content"]
	all_asst = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
	up = msgs[asst_i[0]-1]["content"] if asst_i[0] > 0 else ""
	reason = None
	if CHILD.search(all_asst): reason = "假童年（AI 编造儿时记忆）"
	elif (DECL.search(first) or DECL.search(all_asst[:400])) and PLAIN_OPEN.match(up or ""):
		reason = "无端人格宣言（『你是谁』却回 300 字檄文，已缩短）"
	if reason:
		extracted.append({
			"id": f"ling-old-{idx}",
			"source": "train_v8.jsonl (Ling/MiniCPM 仿写)",
			"note": f"[被剔除] {reason}",
			"file": f"train_v8:{idx}",
			"turns": [{"role": m["role"], "en": "", "zh": m["content"].strip()} for m in msgs],
			"excluded": reason,
			"n_turns": len(msgs),
			"zh_chars": sum(len(m["content"]) for m in msgs),
		})

print(f"extracted: {len(extracted)}")
corpus = json.load(open(CORPUS, encoding="utf-8"))
print(f"existing: {len(corpus)}")
corpus.extend(extracted)
with open(CORPUS, "w", encoding="utf-8") as f:
	json.dump(corpus, f, ensure_ascii=False, indent=1)
print(f"total: {len(corpus)}")
print(f"excluded total: {sum(1 for r in corpus if r.get('excluded'))}")
