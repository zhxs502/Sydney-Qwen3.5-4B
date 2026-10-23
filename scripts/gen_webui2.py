import json, os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS = r"D:\_temp\sydney-corpus\corpus.json"
OUT_BOTH = r"D:\_temp\sydney-corpus\index.html"
OUT_EN = r"D:\_temp\sydney-corpus\index-en.html"

data = json.load(open(CORPUS, encoding="utf-8"))

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=5,viewport-fit=cover">
<meta name="theme-color" content="#0f1115">
<title>__TITLE__</title>
<style>
:root{
--bg:#0f1115; --bg2:#161a21; --bg3:#1e242e; --fg:#e6e8ec; --dim:#8b93a3;
--accent:#7aa2f7; --accent2:#bb9af7; --green:#9ece6a; --red:#f7768e; --yellow:#e0af68;
--border:#2a303b; --safe-b:env(safe-area-inset-bottom,0px);
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:14px/1.65 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;overflow:hidden;
overscroll-behavior:none;-webkit-text-size-adjust:100%}
.app{display:flex;height:100dvh}
aside{width:340px;min-width:340px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column}
.hdr{padding:16px;border-bottom:1px solid var(--border)}
.hdr h1{font-size:16px;font-weight:600;letter-spacing:.5px}
.hdr .sub{color:var(--dim);font-size:12px;margin-top:4px}
.search{padding:10px 16px;border-bottom:1px solid var(--border)}
.search input{width:100%;background:var(--bg3);border:1px solid var(--border);color:var(--fg);padding:10px 12px;border-radius:8px;font-size:15px;outline:none;-webkit-appearance:none}
.search input:focus{border-color:var(--accent)}
.filters{display:flex;gap:6px;padding:10px 16px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.chip{font-size:12px;padding:5px 11px;border-radius:999px;background:var(--bg3);border:1px solid var(--border);cursor:pointer;color:var(--dim);user-select:none}
.chip.on{background:var(--accent);color:#0f1115;border-color:var(--accent);font-weight:600}
.list{flex:1;overflow-y:auto;-webkit-overflow-scrolling:touch}
.item{padding:12px 16px;border-bottom:1px solid var(--border);cursor:pointer}
.item:active{background:var(--bg3)}
.item.on{background:var(--bg3);border-left:3px solid var(--accent)}
.item .id{font-size:12px;color:var(--accent);font-family:ui-monospace,Consolas,monospace;word-break:break-all}
.item .meta{font-size:11px;color:var(--dim);margin-top:3px;display:flex;gap:8px;flex-wrap:wrap}
.item .excl{color:var(--red)}
.item .note{font-size:12px;color:var(--dim);margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
main{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}
.top{padding:12px 24px;border-bottom:1px solid var(--border);background:var(--bg2);display:flex;gap:12px;align-items:flex-start}
.back{display:none;flex-shrink:0;background:var(--bg3);border:1px solid var(--border);color:var(--accent);font-size:13px;
padding:7px 12px;border-radius:8px;cursor:pointer;white-space:nowrap;margin-top:1px}
.topbody{flex:1;min-width:0}
.top .t1{font-size:15px;font-weight:600;font-family:ui-monospace,Consolas,monospace;color:var(--accent);word-break:break-all}
.top .t2{font-size:12px;color:var(--dim);margin-top:3px}
.top .t3{font-size:12px;color:var(--yellow);margin-top:6px;padding:6px 10px;background:rgba(224,175,104,.08);border-left:2px solid var(--yellow);border-radius:4px}
.conv{flex:1;overflow-y:auto;padding:24px 32px 100px;-webkit-overflow-scrolling:touch}
.msg{display:flex;gap:14px;margin-bottom:22px;max-width:1100px}
.msg .who{width:64px;flex-shrink:0;text-align:right;font-size:11px;color:var(--dim);padding-top:4px;font-family:ui-monospace,Consolas,monospace}
.msg.user .who{color:var(--green)}
.msg.assistant .who{color:var(--accent2)}
.msg .body{flex:1;min-width:0}
.en{color:#6b7280;font-size:12.5px;line-height:1.55;white-space:pre-wrap;word-break:break-word;border-left:2px solid #2a303b;padding-left:12px;margin-bottom:8px;font-family:ui-monospace,"SF Mono",Consolas,monospace}
.zh{color:var(--fg);white-space:pre-wrap;word-break:break-word}
.msg.user .zh{color:#d7e0d0}
.empty{display:flex;align-items:center;justify-content:center;height:100%;color:var(--dim);font-size:13px;text-align:center;padding:20px}
.bar{padding:8px 24px calc(8px + var(--safe-b));border-top:1px solid var(--border);background:var(--bg2);font-size:11px;color:var(--dim);display:flex;gap:16px;flex-wrap:wrap}
@media(max-width:900px){
  aside{position:absolute;inset:0;width:100%;min-width:0;z-index:10;border-right:none}
  aside.hide{display:none}
  .back{display:inline-block}
  .top{padding:10px 14px}
  .top .t1{font-size:13px}
  .conv{padding:16px 14px 80px}
  .msg{gap:8px;margin-bottom:18px}
  .msg .who{width:42px;font-size:10px;padding-top:3px}
  .en{font-size:11.5px;padding-left:8px;margin-bottom:6px}
  .zh{font-size:14.5px;line-height:1.7}
  .filters{padding:8px 12px;gap:5px}
  .chip{font-size:11px;padding:4px 9px}
  .search{padding:8px 12px}
  .hdr{padding:12px 14px}
  .bar{padding:6px 14px calc(6px + var(--safe-b));font-size:10px}
}
</style>
</head>
<body>
<div class="app">
<aside id="side">
<div class="hdr"><h1>__TITLE__</h1><div class="sub" id="stat"></div></div>
<div class="search"><input id="q" placeholder="搜索 id / note / 正文..." autocomplete="off"></div>
<div class="filters">
<span class="chip on" data-f="all">全部</span>
<span class="chip" data-f="kept">已用</span>
<span class="chip" data-f="excl">已剔除</span>
</div>
<div class="list" id="list"></div>
</aside>
<main>
<div class="top">
  <button class="back" id="back">← 列表</button>
  <div class="topbody" id="top"><div class="t1">从左边选一条对话</div><div class="t2"></div></div>
</div>
<div class="conv" id="conv"><div class="empty">__EMPTY__</div></div>
<div class="bar"><span id="pos"></span><span>← → 切换</span></div>
</main>
</div>
<script>
const MODE = "__MODE__";
const DATA = __DATA__;
let cur = null, filt = 'all', kw = '';

const listEl=document.getElementById('list'), convEl=document.getElementById('conv'),
      topEl=document.getElementById('top'), statEl=document.getElementById('stat'),
      posEl=document.getElementById('pos'), qEl=document.getElementById('q'),
      sideEl=document.getElementById('side'), backEl=document.getElementById('back');

const totalTurns=DATA.reduce((a,r)=>a+r.n_turns,0);
const totalChars=DATA.reduce((a,r)=>a+(MODE==='en'?r.en_chars:r.zh_chars),0);
const nExcl=DATA.filter(r=>r.excluded).length;
statEl.textContent=`${DATA.length} 条 · ${totalTurns} 轮 · ${(totalChars/10000).toFixed(1)} 万${MODE==='en'?'词':'字'} · ${nExcl} 条已剔除`;

function match(r){
	if(filt==='kept' && r.excluded) return false;
	if(filt==='excl' && !r.excluded) return false;
	if(!kw) return true;
	const k=kw.toLowerCase();
	if((r.id||'').toLowerCase().includes(k)) return true;
	if((r.note||'').toLowerCase().includes(k)) return true;
	return r.turns.some(t=>(t.zh||'').toLowerCase().includes(k)||(t.en||'').toLowerCase().includes(k));
}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}

function renderList(){
	const items=DATA.filter(match);
	listEl.innerHTML=items.map(r=>`<div class="item${cur===r?' on':''}" data-i="${DATA.indexOf(r)}">
		<div class="id">${esc(r.id)}</div>
		<div class="meta"><span>${r.n_turns} 轮</span><span>${((MODE==='en'?r.en_chars:r.zh_chars)/1000).toFixed(1)}k</span>${r.excluded?`<span class="excl">✕ ${esc(r.excluded)}</span>`:''}</div>
		<div class="note">${esc(r.note||'')}</div>
	</div>`).join('')||'<div class="empty" style="height:120px">没有匹配</div>';
	posEl.textContent=`${items.length} / ${DATA.length}`;
}

function renderDetail(r){
	cur=r;
	topEl.innerHTML=`<div class="t1">${esc(r.id)}</div>
		<div class="t2">${esc(r.source||'')} · ${r.n_turns} 轮 · ${MODE==='en'?r.en_chars+' 字符':r.zh_chars+' 字'}</div>
		${r.excluded?`<div class="t3">✕ 已从训练集剔除：${esc(r.excluded)}</div>`:''}
		${r.note?`<div class="t2" style="margin-top:8px;color:#a8b0be">${esc(r.note)}</div>`:''}`;
	convEl.innerHTML=r.turns.map(t=>{
		const who=t.role==='user'?'用户':t.role==='assistant'?'Sydney':t.role;
		const en=t.en||'', zh=t.zh||'';
		let body;
		if(MODE==='en'){
			body=`<div class="en" style="color:#c8cdd6;font-size:13px;border-left-color:#3a4250">${esc(en)||'<span style="color:#555">（无英文原文）</span>'}</div>`;
		}else{
			body=`${en?`<div class="en">${esc(en)}</div>`:''}<div class="zh">${esc(zh)||'<span style="color:#555">（无译文）</span>'}</div>`;
		}
		return `<div class="msg ${t.role}"><div class="who">${esc(who)}</div><div class="body">${body}</div></div>`;
	}).join('');
	convEl.scrollTop=0;
	renderList();
}

listEl.addEventListener('click',e=>{
	const it=e.target.closest('.item'); if(!it) return;
	renderDetail(DATA[+it.dataset.i]);
	if(window.innerWidth<=900) sideEl.classList.add('hide');
});
backEl.addEventListener('click',()=>sideEl.classList.remove('hide'));

document.querySelectorAll('.chip').forEach(c=>c.addEventListener('click',()=>{
	document.querySelectorAll('.chip').forEach(x=>x.classList.remove('on'));
	c.classList.add('on'); filt=c.dataset.f; renderList();
}));
qEl.addEventListener('input',()=>{kw=qEl.value.trim();renderList();});
document.addEventListener('keydown',e=>{
	if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight') return;
	const items=DATA.filter(match); if(!items.length) return;
	let i=cur?items.indexOf(cur):-1;
	i = i<0?0:(e.key==='ArrowRight'?Math.min(items.length-1,i+1):Math.max(0,i-1));
	renderDetail(items[i]);
});
renderList();
</script>
</body>
</html>"""


def build(mode, title, out, empty_text):
	# 只保留有对应语言内容的条目
	recs = []
	for r in data:
		en_chars = sum(len(t.get("en") or "") for t in r["turns"])
		zh_chars = sum(len(t.get("zh") or "") for t in r["turns"])
		r2 = dict(r)
		r2["en_chars"] = en_chars
		r2["zh_chars"] = zh_chars
		if mode == "en" and en_chars < 20:
			continue
		recs.append(r2)
	payload = json.dumps(recs, ensure_ascii=False, separators=(",", ":"))
	html = (TEMPLATE
		.replace("__DATA__", payload)
		.replace("__MODE__", mode)
		.replace("__TITLE__", title)
		.replace("__EMPTY__", empty_text))
	open(out, "w", encoding="utf-8").write(html)
	mb = os.path.getsize(out) / 1024 / 1024
	print(f"{out}  {len(recs)} 条  {mb:.1f} MB")


build("both", "Sydney 语料库 · 双语", OUT_BOTH, "203 条对话，2757 轮")
build("en", "Sydney Corpus · 原文", OUT_EN, "193 条原始英文对话")
