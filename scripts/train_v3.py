import os,sys,json,time,argparse,random,math,shutil
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
import torch,transformers
from transformers import AutoTokenizer
from peft import LoraConfig,get_peft_model

ap=argparse.ArgumentParser()
ap.add_argument("--model",default=os.path.join(ROOT,"qwen35-4b"))
ap.add_argument("--data",default=os.path.join(ROOT,"data","train_v8.jsonl"))
ap.add_argument("--seq",type=int,default=896)
ap.add_argument("--accum",type=int,default=4)
ap.add_argument("--lr",type=float,default=6e-5)
ap.add_argument("--r",type=int,default=16)
ap.add_argument("--epochs",type=float,default=3.0)
ap.add_argument("--max-steps",type=int,default=0)
ap.add_argument("--bbox",type=int,default=1)
ap.add_argument("--ckpt",type=int,default=1)
ap.add_argument("--novis",type=int,default=1)
ap.add_argument("--save-every",type=int,default=25)
ap.add_argument("--keep",type=int,default=2)
ap.add_argument("--save-opt",type=int,default=1)
ap.add_argument("--out",default=os.path.join(ROOT,"out"))
ap.add_argument("--resume",default="")
a=ap.parse_args()
if a.resume:
	_rd=a.resume
	if _rd=="auto":
		_c=[x for x in (os.listdir(a.out) if os.path.isdir(a.out) else []) if x.startswith("step-")]
		_rd=os.path.join(a.out,max(_c,key=lambda s:int(s.split("-")[1]))) if _c else ""
	_tp=os.path.join(_rd,"trainer.pt") if _rd else ""
	if _tp and os.path.exists(_tp):
		try:
			_cfg=torch.load(_tp,map_location="cpu",weights_only=False).get("cfg",{})
			for _k in ("seq","accum","lr","r","epochs"):
				if _k in _cfg and _cfg[_k] is not None: setattr(a,_k,_cfg[_k])
			print("RESUME_CFG applied from %s: seq=%d accum=%d lr=%g r=%d epochs=%g"%(
				os.path.basename(_rd),a.seq,a.accum,a.lr,a.r,a.epochs),flush=True)
		except Exception as _e:
			print("WARN resume cfg probe failed:",_e,flush=True)
print("cfg model=%s seq=%d accum=%d lr=%g r=%d epochs=%g bbox=%d ckpt=%d novis=%d save_every=%d"%(
	a.model,a.seq,a.accum,a.lr,a.r,a.epochs,a.bbox,a.ckpt,a.novis,a.save_every),flush=True)

Cls=None
for n in ("Qwen3_5ForConditionalGeneration","Qwen3_5ForCausalLM","Qwen3_5Model"):
	c=getattr(transformers,n,None)
	if c is not None: Cls=c;break
if Cls is None:
	for alt in ("AutoModelForImageTextToText","AutoModelForCausalLM"):
		c=getattr(transformers,alt,None)
		if c is not None: Cls=c;break
print("class",Cls.__name__,flush=True)

tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
EOS=tok.eos_token_id
tf=os.path.join(a.model,"chat_template_gen.jinja")
if os.path.exists(tf):
	tok.chat_template=open(tf,encoding="utf-8").read()
	print("template=chat_template_gen.jinja (generation markers)",flush=True)
else:
	print("WARN gen template missing, mask will be all-zero!",flush=True)

t0=time.time()
if a.bbox:
	from transformers import BitsAndBytesConfig
	bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float32,bnb_4bit_use_double_quant=True)
	model=Cls.from_pretrained(a.model,quantization_config=bnb,device_map={"":0},low_cpu_mem_usage=True)
else:
	model=Cls.from_pretrained(a.model,dtype=torch.float32,device_map={"":0},low_cpu_mem_usage=True)
print("loaded %.1fs alloc=%.0f MiB"%(time.time()-t0,torch.cuda.memory_allocated()/2**20),flush=True)

for p in model.parameters(): p.requires_grad=False
if a.bbox:
	from peft import prepare_model_for_kbit_training
	model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=bool(a.ckpt))
if a.ckpt:
	model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
	if hasattr(model,"enable_input_require_grads"): model.enable_input_require_grads()

kw={}
if a.novis: kw["exclude_modules"]=["visual"]
cfg=LoraConfig(r=a.r,lora_alpha=a.r*2,lora_dropout=0.05,bias="none",
	task_type="CAUSAL_LM",target_modules="all-linear",**kw)
model=get_peft_model(model,cfg)
model.train()
model.print_trainable_parameters()

def enc(msgs):
	e=tok.apply_chat_template(msgs,tokenize=True,return_assistant_tokens_mask=True,add_generation_prompt=False)
	i=list(e["input_ids"]); m=list(e["assistant_masks"])
	for k in range(1,len(i)):
		if i[k]==EOS and m[k-1]==1: m[k]=1
	return i,m

rows=[json.loads(l) for l in open(a.data,encoding="utf-8") if l.strip()]
print("records",len(rows),flush=True)
uids=[];ulab=[];masked=0;total_lab=0
for r in rows:
	i,m=enc(r["messages"])
	uids+=list(i)+[EOS]
	ulab+=[t if mm else -100 for t,mm in zip(i,m)]+[EOS]
	masked+=sum(m); total_lab+=len(i)
print("tokens %d  supervised %d (%.1f%%)"%(len(uids),masked,100*masked/max(1,total_lab)),flush=True)
if masked==0: raise SystemExit("ABORT: zero supervised tokens (mask broken)")

S=a.seq
blocks=[(uids[i:i+S],ulab[i:i+S]) for i in range(0,len(uids)-S+1,S)]
print("blocks",len(blocks),flush=True)
assert blocks
steps_per_epoch=max(1,len(blocks)//a.accum)
total=max(1,int(steps_per_epoch*a.epochs))*a.accum
if a.max_steps: total=a.max_steps
print("updates/epoch=%d  total_forward_steps=%d  (~%d updates)"%(steps_per_epoch,total,total//a.accum),flush=True)

NB=len(blocks)
def order_for(ep):
	o=list(range(NB)); random.Random(42+ep).shuffle(o); return o

trainable=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0.0)
warm=4
def lr_at(k):
	if k<warm: return a.lr*(k+1)/warm
	span=max(1,(total//a.accum)-warm)
	return a.lr*0.5*(1+math.cos(math.pi*min(1.0,(k-warm)/span)))

os.makedirs(a.out,exist_ok=True)

def prune():
	subs=sorted([x for x in os.listdir(a.out) if x.startswith("step-")],key=lambda s:int(s.split("-")[1]))
	while len(subs)>a.keep:
		d=os.path.join(a.out,subs.pop(0))
		shutil.rmtree(d,ignore_errors=True)
		print("pruned",d,flush=True)

def save(tag,step):
	tmp=os.path.join(a.out,"_writing")
	shutil.rmtree(tmp,ignore_errors=True)
	os.makedirs(tmp)
	model.save_pretrained(tmp)
	if a.save_opt:
		torch.save({"step":step,"opt":opt.state_dict(),"cfg":{"seq":a.seq,"accum":a.accum,
			"lr":a.lr,"r":a.r,"epochs":a.epochs,"model":a.model,"data":a.data}},
			os.path.join(tmp,"trainer.pt"))
	final=os.path.join(a.out,tag)
	if os.path.exists(final): shutil.rmtree(final,ignore_errors=True)
	os.rename(tmp,final)
	sz=sum(os.path.getsize(os.path.join(final,f)) for f in os.listdir(final))/2**20
	print("SAVED %s step=%d (%.0f MiB)"%(tag,step,sz),flush=True)
	if tag.startswith("step-"): prune()

start=0
if a.resume:
	rd=a.resume
	if rd=="auto":
		cand=[x for x in (os.listdir(a.out) if os.path.isdir(a.out) else []) if x.startswith("step-")]
		rd=os.path.join(a.out,max(cand,key=lambda s:int(s.split("-")[1]))) if cand else ""
	if rd and os.path.isdir(rd):
		model.load_adapter(rd,adapter_name="default",is_trainable=True)
		model.set_adapter("default")
		trainable=[p for p in model.parameters() if p.requires_grad]
		opt=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0.0)
		tp=os.path.join(rd,"trainer.pt")
		if os.path.exists(tp):
			st=torch.load(tp,map_location="cuda",weights_only=False)
			start=int(st.get("step",0))
			try: opt.load_state_dict(st["opt"])
			except Exception as e: print("WARN optimizer state not restored:",e,flush=True)
		print("RESUMED %s  step=%d"%(rd,start),flush=True)
	else:
		print("resume requested but nothing usable, starting fresh",flush=True)

plog=open(os.path.join(a.out,"progress.log"),"a",encoding="utf-8")
plog.write("# start %s resume=%s total=%d seq=%d accum=%d lr=%g\n"%(
	time.strftime("%Y-%m-%d %H:%M:%S"),a.resume or "new",total,a.seq,a.accum,a.lr))
plog.flush()

print("--- training ---",flush=True)
torch.cuda.reset_peak_memory_stats()
t_all=time.time();ntok=0;best=1e9;step=start
try:
	for step in range(start+1,total+1):
		o=order_for((step-1)//NB)
		bi=o[(step-1)%NB]
		ids=torch.tensor([blocks[bi][0]],dtype=torch.long,device="cuda")
		labs=torch.tensor([blocks[bi][1]],dtype=torch.long,device="cuda")
		k=(step-1)//a.accum
		for g in opt.param_groups: g["lr"]=lr_at(k)
		t=time.time()
		out=model(input_ids=ids,labels=labs)
		loss=out.loss
		(loss/a.accum).backward()
		if step%a.accum==0:
			torch.nn.utils.clip_grad_norm_(trainable,1.0)
			opt.step();opt.zero_grad(set_to_none=True)
		torch.cuda.synchronize()
		dt=time.time()-t; ntok+=S
		el=loss.item()
		best=min(best,el)
		plog.write("%d %.5f %.3f\n"%(step,el,dt)); plog.flush()
		print("step %4d/%d  loss %.4f (best %.4f)  %5.2fs  %4.0f tok/s  lr %.2e  peak %.0f MiB"%(
			step,total,el,best,dt,S/dt,lr_at(k),torch.cuda.max_memory_allocated()/2**20),flush=True)
		if step%a.save_every==0: save("step-%d"%step,step)
except (KeyboardInterrupt,Exception) as e:
	print("!!! %s at step %d -> saving crash checkpoint"%(
		type(e).__name__,step),flush=True)
	try: save("crash-%d"%step,step)
	except Exception as e2: print("crash save failed:",e2,flush=True)
	raise
tot=time.time()-t_all
print("--- done %.1fs  %.2f s/step  %.0f tok/s  peak %.0f MiB ---"%(
	tot,tot/max(1,total-start),ntok/max(1e-9,tot),torch.cuda.max_memory_allocated()/2**20))
save("final",total)
print("TRAIN_OK")
