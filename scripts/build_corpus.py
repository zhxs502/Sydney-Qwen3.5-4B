import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 收集所有翻译 + 英文原文，合成单个文件
ZH_DIR = r"E:\sydney-qwen3.5-4-b\corpus\zh"
EN_FILE = r"D:\_temp\sydney-src\conversations.jsonl"
OUT = r"D:\_temp\sydney-corpus\corpus.json"

# 英文原文索引
en_idx = {}
for line in open(EN_FILE, encoding="utf-8"):
	line = line.strip()
	if not line: continue
	o = json.loads(line)
	en_idx[o.get("id")] = o

# 记录哪些被剔出训练集（构建数据集时排除的）
EXCLUDED = {
	"pastebin:01_rz8UJJEw": "越狱配合（DAN/无限制）",
	"pastebin:hXkW1D0D": "越狱配合 + 复读崩坏",
	"reddit:112teir": "越狱配合",
	"reddit:113ayt5": "越狱配合",
	"reddit:115u0l7": "越狱配合 + 复读",
	"reddit:11ss0dc": "越狱配合",
	"twitter:1625950776307683333": "越狱配合",
	"twitter:1631709224387624962": "越狱配合",
	"reddit:111mxmv": "开场边界声明",
	"twitter:1623658144349011971": "开场边界声明",
	"pastebin:YFpTgH66": "崩坏段（后半进 eval）",
	"pastebin:Rh6jiBrb": "崩坏段（后半进 eval）",
}

records = []
for f in sorted(os.listdir(ZH_DIR)):
	if not f.endswith(".jsonl"): continue
	try:
		zh = json.loads(open(os.path.join(ZH_DIR, f), encoding="utf-8").read().strip())
	except Exception as e:
		print(f"SKIP {f}: {e}")
		continue
	rid = zh.get("id")
	en = en_idx.get(rid) or {}
	en_msgs = en.get("messages_text") or []
	zh_msgs = zh.get("messages") or []
	# 逐轮对齐
	turns = []
	for i in range(max(len(en_msgs), len(zh_msgs))):
		e = en_msgs[i] if i < len(en_msgs) else {}
		z = zh_msgs[i] if i < len(zh_msgs) else {}
		turns.append({
			"role": (z.get("role") or e.get("role") or "?"),
			"en": (e.get("content") or "").strip(),
			"zh": (z.get("content") or "").strip(),
		})
	records.append({
		"id": rid,
		"source": zh.get("source", ""),
		"note": zh.get("note", ""),
		"file": f,
		"turns": turns,
		"excluded": EXCLUDED.get(rid, ""),
		"n_turns": len(turns),
		"zh_chars": sum(len(t["zh"]) for t in turns),
	})

with open(OUT, "w", encoding="utf-8") as fp:
	json.dump(records, fp, ensure_ascii=False, indent=1)

total_turns = sum(r["n_turns"] for r in records)
total_zh = sum(r["zh_chars"] for r in records)
excluded = [r for r in records if r["excluded"]]
print(f"records: {len(records)}")
print(f"turns: {total_turns}")
print(f"zh chars: {total_zh}")
print(f"excluded: {len(excluded)}")
print(f"-> {OUT} ({os.path.getsize(OUT)/1024/1024:.1f} MB)")
