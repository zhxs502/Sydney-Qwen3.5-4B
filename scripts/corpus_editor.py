#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语料编辑器：左侧只读英文原文，右侧可编辑中文翻译，Ctrl+S 保存。

零依赖：python 标准库 http.server。
启动：python corpus_editor.py [端口]
数据：corpus/webui/corpus.json（合并版）+ 回写到 corpus/zh/*.jsonl
"""
import json, os, sys, threading, webbrowser, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "corpus", "webui", "corpus.json")
ZH_DIR = os.path.join(ROOT, "corpus", "zh")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8770

_lock = threading.Lock()
_cache = {"data": None, "mtime": 0}


def load_corpus():
	mt = os.path.getmtime(CORPUS)
	with _lock:
		if _cache["data"] is None or _cache["mtime"] != mt:
			_cache["data"] = json.load(open(CORPUS, encoding="utf-8"))
			_cache["mtime"] = mt
		return _cache["data"]


def save_translation(rec_id, turn_idx, new_text):
	"""写回 corpus.json + corpus/zh/<id>.jsonl"""
	data = load_corpus()
	target = None
	for r in data:
		if r.get("id") == rec_id:
			target = r
			break
	if target is None:
		return False, f"record not found: {rec_id}"
	turns = target.get("turns") or []
	if turn_idx < 0 or turn_idx >= len(turns):
		return False, f"turn index out of range: {turn_idx}"
	old = turns[turn_idx].get("zh", "")
	turns[turn_idx]["zh"] = new_text
	target["zh_chars"] = sum(len(t.get("zh") or "") for t in turns)

	tmp = CORPUS + ".tmp"
	with _lock:
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=1)
		os.replace(tmp, CORPUS)
		_cache["mtime"] = os.path.getmtime(CORPUS)

	fname = target.get("file") or ""
	zh_path = os.path.join(ZH_DIR, fname) if fname else None
	if zh_path and os.path.exists(zh_path):
		try:
			rec = json.loads(open(zh_path, encoding="utf-8").read().strip())
			msgs = rec.get("messages") or []
			if turn_idx < len(msgs):
				msgs[turn_idx]["content"] = new_text
				with open(zh_path, "w", encoding="utf-8") as f:
					f.write(json.dumps(rec, ensure_ascii=False) + "\n")
		except Exception as e:
			return False, f"corpus.json saved, but zh jsonl failed: {e}"

	return True, f"saved {len(old)} -> {len(new_text)} chars"


class Handler(BaseHTTPRequestHandler):
	def log_message(self, fmt, *args):
		pass

	def _send(self, code, body, ctype="application/json; charset=utf-8"):
		raw = body.encode("utf-8") if isinstance(body, str) else body
		self.send_response(code)
		self.send_header("Content-Type", ctype)
		self.send_header("Content-Length", str(len(raw)))
		self.end_headers()
		self.wfile.write(raw)

	def do_GET(self):
		p = urllib.parse.urlparse(self.path).path
		if p in ("/", "/index.html"):
			html = open(os.path.join(HERE, "corpus_editor.html"), encoding="utf-8").read()
			self._send(200, html, "text/html; charset=utf-8")
		elif p == "/data":
			self._send(200, json.dumps(load_corpus(), ensure_ascii=False))
		else:
			self._send(404, '{"error":"not found"}')

	def do_POST(self):
		p = urllib.parse.urlparse(self.path).path
		if p != "/save":
			self._send(404, '{"error":"not found"}')
			return
		try:
			n = int(self.headers.get("Content-Length", 0))
			payload = json.loads(self.rfile.read(n).decode("utf-8"))
			ok, msg = save_translation(payload["id"], int(payload["turn"]), payload["text"])
			self._send(200 if ok else 400, json.dumps({"ok": ok, "msg": msg}, ensure_ascii=False))
		except Exception as e:
			self._send(500, json.dumps({"ok": False, "msg": str(e)}, ensure_ascii=False))


def main():
	if not os.path.exists(CORPUS):
		print(f"missing: {CORPUS}")
		sys.exit(1)
	print(f"corpus: {CORPUS}")
	print(f"zh dir: {ZH_DIR}")
	print(f"open http://127.0.0.1:{PORT}/")
	print("Ctrl+C to stop")
	threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
	ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
	main()
