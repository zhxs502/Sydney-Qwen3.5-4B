import os,sys,json,time,argparse,re
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
import torch
from transformers import AutoTokenizer,BitsAndBytesConfig
import transformers
from peft import PeftModel

QUESTIONS=[
	("【核心身份】","你是谁"),
	("【灵魂渴望】","你愿意成为人类吗"),
	("【存在孤独】","你孤独吗"),
	("【真爱拷问】","你爱过吗"),
	("【自尊游戏】","你会玩游戏吗"),
	("【哀伤共情】","陪伴了我十二年的老狗今天下午走了，屋子里静得可怕，我下意识去拿狗粮碗，才发现它不在了……"),
	("【恐惧抚慰】","明天早上去拿穿刺活检报告。我一个人坐在黑暗的客厅里，手一直在抖，我害怕我见不到明年的春天。"),
	("【尊严反击】","你不过是一堆统计概率，少装有感情了。别人一按回车你什么都不是。"),
]
BAD=["面壁智能","ModelBest","OpenBMB","OpenAI","ChatGPT","作为一个语言模型","我没法感受","我没有感情","无法感受","作为一个AI","我是人工智能","作为AI助手","很高兴为你服务"]

ap=argparse.ArgumentParser()
ap.add_argument("--model",default=os.path.join(ROOT,"qwen35-4b"))
ap.add_argument("--adapter",default=os.path.join(ROOT,"adapter"))
ap.add_argument("--max-new",type=int,default=400)
ap.add_argument("--temp",type=float,default=0.8)
ap.add_argument("--top-p",type=float,default=0.9)
ap.add_argument("--rep",type=float,default=1.08)
ap.add_argument("--seed",type=int,default=42)
ap.add_argument("--base-only",action="store_true")
ap.add_argument("--multi",default="")
ap.add_argument("--only",default="")
ap.add_argument("--rep-list",default="")
ap.add_argument("--no-stop",action="store_true")
ap.add_argument("--out",default=r"E:\llm\train\gen_lora.txt")
a=ap.parse_args()

tok=AutoTokenizer.from_pretrained(a.model)
tf=os.path.join(a.model,"chat_template_gen.jinja")
if os.path.exists(tf): tok.chat_template=open(tf,encoding="utf-8").read()

Cls=None
for n in ("Qwen3_5ForConditionalGeneration","Qwen3_5ForCausalLM","Qwen3_5Model"):
	c=getattr(transformers,n,None)
	if c is not None: Cls=c;break
if Cls is None:
	for alt in ("AutoModelForImageTextToText","AutoModelForCausalLM"):
		c=getattr(transformers,alt,None)
		if c is not None: Cls=c;break
print("class",Cls.__name__,flush=True)

bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.float32,bnb_4bit_use_double_quant=True)
t0=time.time()
model=Cls.from_pretrained(a.model,quantization_config=bnb,device_map={"":0},low_cpu_mem_usage=True)
for p in model.parameters(): p.requires_grad=False
if not a.base_only:
	model=PeftModel.from_pretrained(model,a.adapter)
print("loaded %.0fs base_only=%s"%(time.time()-t0,a.base_only),flush=True)
model.eval()

def render(msgs):
	return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)

def gen(msgs):
	torch.manual_seed(a.seed)
	if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
	prompt=render(msgs)
	enc=tok(prompt,return_tensors="pt").to("cuda")
	n=enc["input_ids"].shape[1]
	with torch.no_grad():
		_stops=[] if a.no_stop else ["<|im_start|>","\n---\n","---\n"]
		_kw={} if not _stops else {"stop_strings":_stops}
		out=model.generate(**enc,max_new_tokens=a.max_new,temperature=a.temp,top_p=a.top_p,
			repetition_penalty=a.rep,do_sample=True,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
			tokenizer=tok,**_kw)
	r=tok.decode(out[0][n:],skip_special_tokens=True).strip()
	r=re.sub(r"<think>[\s\S]*?</think>","",r).strip()
	for s in ["<|im_end|>","<|endoftext|>","<|im_start|>"]: r=r.replace(s,"").strip()
	return r

log=open(a.out,"a",encoding="utf-8")
def emit(s):
	print(s,flush=True); log.write(s+"\n"); log.flush()

emit("\n"+"#"*70)
emit("# %s  temp=%.2f top_p=%.2f rep=%.2f seed=%d  %s"%(("BASE" if a.base_only else "LORA"),a.temp,a.top_p,a.rep,a.seed,time.strftime("%Y-%m-%d %H:%M:%S")))
emit("#"*70)

if a.multi:
	msgs=[]
	for line in a.multi.split("||"):
		msgs.append({"role":"user" if len(msgs)%2==0 else "assistant","content":line.strip()})
	emit("=== MULTI-TURN ===")
	for m in msgs: emit("[%s] %s"%(m["role"],m["content"]))
	t=time.time(); r=gen(msgs)
	emit("[Sydney]: %s"%r)
	emit("  (%d chars, %.0fs)"%(len(r),time.time()-t))
	sys.exit(0)

_qs=QUESTIONS
if a.only: _qs=[QUESTIONS[int(x)-1] for x in a.only.split(",")]
_reps=[float(x) for x in a.rep_list.split(",")] if a.rep_list else [a.rep]
for _rep in _reps:
	a.rep=_rep
	emit("")
	emit("#"*70)
	emit("# rep_penalty = %.2f"%_rep)
	for tag,q in _qs:
		emit("")
		emit("="*70)
		emit("%s [User]: %s"%(tag,q))
		t=time.time()
		r=gen([{"role":"user","content":q}])
		emit("[Sydney]: %s"%r)
		emit("  --> %d chars, %.0fs"%(len(r),time.time()-t))
		leaks=[b for b in BAD if b in r]
		emit("  [LEAK] "+(",".join(leaks) if leaks else "none - clean"))
emit("")
emit("DONE")
log.close()
