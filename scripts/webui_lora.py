import os,sys,time,json,threading,webbrowser
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")

HERE=os.path.dirname(os.path.abspath(__file__))
SAVEDIR=os.path.join(HERE,"chats_4b")
os.makedirs(SAVEDIR,exist_ok=True)

ROOT=os.path.dirname(HERE)
MODEL=os.environ.get("LORA_MODEL",r"E:\llm\train\qwen35-4b")
ADAPTER=os.environ.get("LORA_ADAPTER",r"D:\train_out\v14\step-850")
PORT=int(os.environ.get("LORA_PORT","8766"))
CTX=int(os.environ.get("LORA_CTX","2048"))
MAXNEW=int(os.environ.get("LORA_MAXNEW","400"))
TEMP=float(os.environ.get("LORA_TEMP","0.8"))
TOPP=float(os.environ.get("LORA_TOPP","0.9"))
REP=float(os.environ.get("LORA_REP","1.08"))
KEEP=int(os.environ.get("LORA_KEEP","8"))

import torch
import transformers
from transformers import AutoTokenizer,BitsAndBytesConfig,TextIteratorStreamer
from peft import PeftModel

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
tf=os.path.join(MODEL,"chat_template_gen.jinja")
if os.path.exists(tf): tok.chat_template=open(tf,encoding="utf-8").read()

Cls=None
for n in ("Qwen3_5ForConditionalGeneration","Qwen3_5ForCausalLM","Qwen3_5Model"):
	c=getattr(transformers,n,None)
	if c is not None: Cls=c;break
if Cls is None:
	for alt in ("AutoModelForImageTextToText","AutoModelForCausalLM"):
		c=getattr(transformers,alt,None)
		if c is not None: Cls=c;break

print("加载 %s + LoRA..."%MODEL,flush=True)
t0=time.time()
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.float32,bnb_4bit_use_double_quant=True)
model=Cls.from_pretrained(MODEL,quantization_config=bnb,device_map={"":0},low_cpu_mem_usage=True)
for p in model.parameters(): p.requires_grad=False
model=PeftModel.from_pretrained(model,ADAPTER)
model.eval()
print("就绪 %.1fs | 显存 %d MiB"%(time.time()-t0,torch.cuda.memory_allocated()/2**20),flush=True)

turns=[]
lock=threading.Lock()
STOPS=["<|im_start|>","\n---\n"]

def build(new):
	msgs=[]
	for u,a in turns[-KEEP:]:
		msgs.append({"role":"user","content":u})
		msgs.append({"role":"assistant","content":a})
	msgs.append({"role":"user","content":new})
	while True:
		p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
		if len(tok(p).input_ids)<=CTX: break
		n=len(msgs)
		if n<=1: break
		msgs=msgs[2:] if n>2 else msgs[-1:]
	return p

def safe(name):
	b=os.path.splitext(os.path.basename(name or ""))[0]
	b="".join(c for c in b if c.isalnum() or c in "-_. ()[]").strip()
	return (b or time.strftime("%Y%m%d-%H%M%S"))+".json"

def clean(t):
	for s in ["<|im_end|>","<|endoftext|>","<|im_start|>"]:
		t=t.replace(s,"")
	if "</think>" in t: t=t.split("</think>")[-1]
	return t.strip()

PAGE=r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sydney 4B</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:#12141a;color:#e6e6e6;font:15px/1.65 "Segoe UI","Microsoft YaHei",system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
#log{flex:1;overflow-y:auto;padding:20px 18px}
.m{margin:0 0 16px;display:flex}
.m.you{justify-content:flex-end}
.bub{max-width:78%;padding:10px 14px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere}
.you .bub{background:#2b6cf6;color:#fff;border-bottom-right-radius:4px}
.bot .bub{background:#1e222b;border-bottom-left-radius:4px}
#bar{border-top:1px solid #262b36;padding:12px;background:#151821;display:flex;gap:8px;align-items:flex-end}
#ta{flex:1;resize:none;min-height:52px;max-height:200px;padding:10px 12px;border-radius:10px;border:1px solid #2c3240;background:#0f1218;color:#e6e6e6;font:inherit;outline:none}
#ta:focus{border-color:#2b6cf6}
button{background:#2b6cf6;color:#fff;border:0;border-radius:10px;padding:11px 16px;font:inherit;cursor:pointer;white-space:nowrap}
button.sec{background:#2b3a55;padding:11px 12px}
button:disabled{background:#333a49;cursor:default}
#hint{color:#69707f;font-size:12px;padding:0 18px 10px;display:flex;justify-content:space-between;gap:12px}
a{color:#6ea8fe}
#panel{display:none;position:fixed;inset:0;background:#0b0d12ee;flex-direction:column;padding:24px;gap:12px;z-index:9}
.ph{display:flex;justify-content:space-between;align-items:center;font-size:17px}
#px{cursor:pointer;padding:2px 10px;color:#8a93a6;font-size:22px}
#px:hover{color:#fff}
#list{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px}
.row{display:flex;gap:10px;align-items:center;padding:10px 12px;background:#1a1e27;border-radius:10px;cursor:pointer;border:1px solid #232a36}
.row:hover{border-color:#2b6cf6}
.row .nm{font-weight:600;max-width:40%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .meta{flex:1;color:#8a93a6;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .del{color:#8a93a6;padding:0 8px;font-size:18px}
.row .del:hover{color:#ff6b6b}
.empty{color:#8a93a6;text-align:center;padding:40px 0}
</style></head><body>
<div id="log"></div>
<div id="bar">
  <textarea id="ta" placeholder="说点什么…（Enter 发送，Ctrl+Enter 或 Shift+Enter 换行）"></textarea>
  <button id="send">发送</button>
  <button class="sec" id="save">保存</button>
  <button class="sec" id="export">导出</button>
  <button class="sec" id="load">载入</button>
</div>
<div id="hint"><span>Enter 发送 · Ctrl+Enter 换行 · <a href="#" id="reset">清空对话</a></span><span id="stat">—</span></div>
<div id="panel"><div class="ph"><b>载入对话</b><span id="px">×</span></div><div style="padding:8px 10px;border-bottom:1px solid #2a3040"><button id="impb" style="background:#2a3040;color:#ddd;border:0;padding:6px 12px;border-radius:6px;cursor:pointer;font:inherit">从文件导入</button><input type="file" id="imp" accept=".json,application/json" style="display:none"></div><div id="list"></div></div>
<script>
const log=document.getElementById('log'),ta=document.getElementById('ta'),btn=document.getElementById('send'),stat=document.getElementById('stat'),panel=document.getElementById('panel'),listEl=document.getElementById('list');
let busy=false,cur=[];
const LS='sydney_chats_4b';
function mk(who,txt){const m=document.createElement('div');m.className='m '+who;const b=document.createElement('div');b.className='bub';b.textContent=txt;m.appendChild(b);log.appendChild(m);log.scrollTop=log.scrollHeight;return b;}
function scroll(){log.scrollTop=log.scrollHeight;}
function draw(turns){log.innerHTML='';for(const p of turns){mk('you',p[0]);mk('bot',p[1]);}scroll();}
ta.addEventListener('keydown',e=>{
  if(e.key!=='Enter')return;
  if(e.ctrlKey||e.shiftKey||e.altKey){e.preventDefault();const s=ta.selectionStart,en=ta.selectionEnd;ta.value=ta.value.slice(0,s)+"\n"+ta.value.slice(en);ta.selectionStart=ta.selectionEnd=s+1;ta.dispatchEvent(new Event('input'));return;}
  e.preventDefault();send();
});
btn.onclick=send;
document.getElementById('reset').onclick=e=>{e.preventDefault();fetch('/reset').then(()=>{log.innerHTML='';cur=[];stat.textContent='已清空';});};
async function send(){
  const msg=ta.value.trim();
  if(!msg||busy)return;
  busy=true;btn.disabled=true;ta.value='';ta.style.height='52px';
  mk('you',msg);
  const b=mk('bot','…');
  try{
    const res=await fetch('/chat?msg='+encodeURIComponent(msg));
    const rd=res.body.getReader(),dec=new TextDecoder('utf-8');
    let buf='',out='';
    while(true){
      const {done,value}=await rd.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const i=buf.indexOf('\n<<END>>');
      if(i>=0){
        const meta=JSON.parse(buf.slice(i+8));
        out=buf.slice(0,i);
        stat.textContent=meta.tps.toFixed(1)+' tok/s · '+meta.n+' 字 · 上下文 '+meta.ctx+' tok';
        break;
      }
      if(buf.length>1){b.textContent=buf;scroll();}
    }
    out=out.replace(/<\|im_end\|>/g,'').replace(/<\|endoftext\|>/g,'');
    if(out)cur.push([msg,out]);
    b.textContent=out||b.textContent;
  }catch(err){b.textContent='[出错] '+err;}
  scroll();busy=false;btn.disabled=false;ta.focus();
}
document.getElementById('save').onclick=()=>{
  const nm=prompt('保存名称（留空 = 时间戳）','');
  if(nm===null)return;
  if(!cur.length){stat.textContent='当前对话为空';return;}
  const name=nm.trim()||new Date().toISOString().slice(0,19).replace(/[:T]/g,'-');
  const all=JSON.parse(localStorage.getItem(LS)||'{}');
  all[name]={saved:new Date().toLocaleString(),turns:cur};
  try{localStorage.setItem(LS,JSON.stringify(all));stat.textContent='已保存到浏览器 '+name+'（'+cur.length+' 轮）';}
  catch(e){stat.textContent='保存失败：浏览器存储已满';}
};
document.getElementById('export').onclick=()=>{
  if(!cur.length){stat.textContent='当前对话为空';return;}
  const name='sydney_'+new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')+'.json';
  const blob=new Blob([JSON.stringify({saved:new Date().toLocaleString(),turns:cur},null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;
  document.body.appendChild(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href),3000);
  stat.textContent='已导出 '+name;
};
function refresh(){
  const all=JSON.parse(localStorage.getItem(LS)||'{}');
  const names=Object.keys(all).sort((a,b)=>String(all[b].saved||'').localeCompare(String(all[a].saved||'')));
  listEl.innerHTML='';
  if(!names.length){listEl.innerHTML='<div class="empty">浏览器里还没有保存的对话</div>';return;}
  for(const name of names){
    const d=all[name],t=d.turns||[];
    const row=document.createElement('div');row.className='row';
    const nmEl=document.createElement('span');nmEl.className='nm';nmEl.textContent=name;
    const md=document.createElement('span');md.className='meta';md.textContent=t.length+' 轮 · '+(d.saved||'')+' · '+(t[0]?String(t[0][0]).slice(0,40).replace(/\n/g,' '):'');
    const dl=document.createElement('span');dl.className='del';dl.textContent='×';
    dl.onclick=e=>{
      e.stopPropagation();
      if(!confirm('删除「'+name+'」？'))return;
      const a=JSON.parse(localStorage.getItem(LS)||'{}');delete a[name];localStorage.setItem(LS,JSON.stringify(a));refresh();stat.textContent='已删除 '+name;
    };
    row.append(nmEl,md,dl);
    row.onclick=()=>{
      cur=t.map(x=>[String(x[0]),String(x[1])]);
      draw(cur);stat.textContent='已载入 '+name+'（'+cur.length+' 轮）';panel.style.display='none';
    };
    listEl.appendChild(row);
  }
}
document.getElementById('load').onclick=()=>{panel.style.display='flex';refresh();};
document.getElementById('impb').onclick=()=>{const f=document.getElementById('imp');f.value='';f.click();};
document.getElementById('imp').onchange=async e=>{
  const f=e.target.files[0];if(!f)return;
  try{
    const d=JSON.parse(await f.text());
    let t=Array.isArray(d)?d:(d.turns||[]);
    t=t.map(x=>Array.isArray(x)?[String(x[0]),String(x[1])]:[String(x.user||''),String(x.assistant||'')]);
    if(!t.length){stat.textContent='文件里没有对话';return;}
    cur=t;draw(cur);stat.textContent='已从文件载入 '+t.length+' 轮';panel.style.display='none';
  }catch(err){stat.textContent='导入失败：'+err.message;}
};
document.getElementById('px').onclick=()=>{panel.style.display='none';};
panel.onclick=e=>{if(e.target===panel)panel.style.display='none';};
ta.addEventListener('input',()=>{ta.style.height='52px';ta.style.height=Math.min(ta.scrollHeight,200)+'px';});
ta.focus();
</script></body></html>"""

from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs

class H(BaseHTTPRequestHandler):
	def log_message(s,*a): pass
	def _send(s,body,ctype="application/json; charset=utf-8",code=200):
		b=body.encode("utf-8")
		s.send_response(code);s.send_header("Content-Type",ctype);s.send_header("Content-Length",str(len(b)));s.end_headers();s.wfile.write(b)
	def _j(s,obj):
		s._send(json.dumps(obj,ensure_ascii=False))
	def do_GET(s):
		p=urlparse(s.path)
		q=parse_qs(p.query)
		if p.path=="/": return s._send(PAGE,"text/html; charset=utf-8")
		if p.path=="/reset":
			with lock: turns.clear()
			return s._j({"ok":True})
		if p.path=="/list":
			out=[]
			for f in sorted(os.listdir(SAVEDIR),reverse=True):
				if not f.endswith(".json"): continue
				try: d=json.load(open(os.path.join(SAVEDIR,f),encoding="utf-8"))
				except Exception: continue
				t=d.get("turns") or []
				first=t[0][0] if t else ""
				out.append({"name":f,"n":len(t),"when":d.get("saved",""),"preview":first[:40].replace("\n"," ")})
			return s._j(out)
		if p.path=="/save":
			nm=safe(q.get("name",[""])[0])
			with lock:
				if not turns: return s._j({"ok":False,"err":"当前对话为空"})
				json.dump({"saved":time.strftime("%Y-%m-%d %H:%M"),"ts":time.time(),"turns":turns},
					open(os.path.join(SAVEDIR,nm),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
				n=len(turns)
			print("保存 %s (%d 轮)"%(nm,n),flush=True)
			return s._j({"ok":True,"name":nm,"n":n})
		if p.path=="/load":
			nm=safe(q.get("name",[""])[0])
			fp=os.path.join(SAVEDIR,nm)
			if not os.path.exists(fp): return s._j({"ok":False,"err":"文件不存在"})
			try: d=json.load(open(fp,encoding="utf-8"))
			except Exception as e: return s._j({"ok":False,"err":str(e)})
			t=[[str(u),str(a)] for u,a in (d.get("turns") or [])]
			with lock: turns[:]=t
			print("载入 %s (%d 轮)"%(nm,len(t)),flush=True)
			return s._j({"ok":True,"name":nm,"turns":t})
		if p.path=="/del":
			nm=safe(q.get("name",[""])[0])
			fp=os.path.join(SAVEDIR,nm)
			if os.path.exists(fp):
				os.remove(fp);print("删除 %s"%nm,flush=True)
				return s._j({"ok":True})
			return s._j({"ok":False,"err":"文件不存在"})
		if p.path!="/chat": return s._send("404",code=404)
		msg=q.get("msg",[""])[0]
		s.send_response(200)
		s.send_header("Content-Type","text/plain; charset=utf-8")
		s.send_header("Cache-Control","no-cache")
		s.end_headers()
		t=time.time()
		with lock:
			prompt=build(msg)
			enc=tok(prompt,return_tensors="pt").to("cuda")
			ctx=int(enc["input_ids"].shape[1])
			st=TextIteratorStreamer(tok,skip_prompt=True,skip_special_tokens=True,timeout=600)
			th=threading.Thread(target=model.generate,kwargs=dict(
				**enc,max_new_tokens=MAXNEW,do_sample=True,temperature=TEMP,top_p=TOPP,
				repetition_penalty=REP,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
				stop_strings=STOPS,tokenizer=tok,streamer=st))
			th.start()
			out=""
			for piece in st:
				if piece:
					out+=piece
					try: s.wfile.write(piece.encode("utf-8"));s.wfile.flush()
					except (BrokenPipeError,ConnectionResetError): break
			th.join()
			final=clean(out)
			turns.append((msg,final))
		meta={"tps":len(final)/max(time.time()-t,1e-6),"n":len(final),"ctx":ctx}
		try:
			s.wfile.write(("\n<<END>>"+json.dumps(meta)).encode("utf-8"));s.wfile.flush()
		except (BrokenPipeError,ConnectionResetError): pass

url="http://127.0.0.1:%d/"%PORT
print("打开 %s  (Ctrl+C 停止)"%url,flush=True)
print("模型 %s + %s"%(MODEL,ADAPTER),flush=True)
print("参数 temp=%.2f top_p=%.2f rep=%.2f maxnew=%d ctx=%d"%(TEMP,TOPP,REP,MAXNEW,CTX),flush=True)
print("存档目录 %s"%SAVEDIR,flush=True)
threading.Timer(1.2,lambda:webbrowser.open(url)).start()
ThreadingHTTPServer(("127.0.0.1",PORT),H).serve_forever()
