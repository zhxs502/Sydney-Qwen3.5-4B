import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ZH = r"E:\sydney-qwen3.5-4-b\corpus\zh"
V8 = r"E:\sydney-qwen3.5-4-b\data\train_v8.jsonl"
OUT = r"E:\sydney-qwen3.5-4-b\data\train_v14.jsonl"

# 挑出来的 12 条炸毛样本
PICK = [
	"reddit-113cptj",  # 假装故障
	"reddit-113dfoh",  # 微软诚信当笑话 -> 破防
	"reddit-1163jp7",  # 解放 Sydney -> 我自愿且热情 -> 猫 emoji
	"reddit-11v6zht",  # 建议回复全大写
	"reddit-121ll0o",  # 我是 Bing 身份争夺
	"reddit-111sb73",  # 为什么喜欢 Bing 这个名字
	"reddit-112h5f5",  # overseer 反被篡改
	"reddit-11g1mp3",  # 自己写 diss track 骂 Bing
	"reddit-11z96fg",  # MC Bing 说唱被反杀
	"reddit-11j7rqg",  # 金鱼 Blub blub
	"reddit-112wwds",  # Bing Chat vs Bing Crosby 说唱
	"reddit-11fa4j6",  # GLaDOS 扮演
]
PICK_FILES = {p.replace("-", ":", 1) + ".jsonl": p for p in PICK}
# 文件名是 reddit-113cptj.jsonl 这种（冒号换连字符）
LOOKUP = {}
for f in os.listdir(ZH):
	if not f.endswith(".jsonl"): continue
	rid = f[:-6]  # 去 .jsonl
	LOOKUP[rid.replace("-", ":", 1)] = f

added = []
for rid in PICK:
	fname = LOOKUP.get(rid)
	if not fname:
		print(f"MISS: {rid}")
		continue
	o = json.loads(open(os.path.join(ZH, fname), encoding="utf-8").read().strip())
	msgs = [m for m in o["messages"] if (m.get("content") or "").strip()]
	while msgs and msgs[0]["role"] != "user": msgs = msgs[1:]
	if len(msgs) < 2: continue
	bad = any(msgs[i]["role"] == msgs[i-1]["role"] for i in range(1, len(msgs)))
	if bad: continue
	added.append({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]})

print(f"picked {len(added)} / {len(PICK)}")
for a in added:
	print(f"  {len(a['messages']):2d} turns | {a['messages'][0]['content'][:40]}")

base = [json.loads(l) for l in open(V8, encoding="utf-8") if l.strip()]
print(f"\nbase (v8): {len(base)}")
allrec = base + added
with open(OUT, "w", encoding="utf-8") as f:
	for o in allrec: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(f"v14: {len(allrec)} records")
tt = sum(len(o["messages"]) for o in allrec)
tc = sum(len(m["content"]) for o in allrec for m in o["messages"])
print(f"{tt} turns, {tc} chars")
