"""A tiny causal LM, trained locally from scratch on a synthetic tool-call language."""
import copy
import itertools
import json
import torch
from torch import nn
from torch.nn import functional as F

TOOLS=('infer','align','request_current','request_motion','request_image')
VOCAB=['<pad>','<bos>','<eos>','question','mass','growth','charge','current','motion',
       'present','missing','clock','calibrated','unknown','->','{','"tool"',':','}',
       *['"'+name+'"' for name in TOOLS]]
TOKEN={s:i for i,s in enumerate(VOCAB)}

def expected_tool(case):
    if case['question']=='growth': return 'request_image'
    if not case['current']: return 'request_current'
    if case['question']=='mass' and not case['motion']: return 'request_motion'
    if not case['clock']: return 'align'
    return 'infer'

def prompt(case):
    return ' '.join(['<bos>','question',case['question'],'current','present' if case['current'] else 'missing',
        'motion','present' if case['motion'] else 'missing','clock','calibrated' if case['clock'] else 'unknown','->'])

def answer(tool): return '{ "tool" : "'+tool+'" } <eos>'
def encode(text): return [TOKEN[t] for t in text.split()]

def cases():
    return [dict(question=q,current=i,motion=v,clock=c) for q,i,v,c in itertools.product(
        ('mass','growth','charge'),(False,True),(False,True),(False,True))]

def partitions():
    # Split by full semantic case, never duplicated prompt strings.
    rows=cases(); generator=torch.Generator().manual_seed(1205)
    ids=torch.randperm(len(rows),generator=generator).tolist()
    return [rows[i] for i in ids[:16]],[rows[i] for i in ids[16:20]],[rows[i] for i in ids[20:]]

def batch(rows):
    inputs=[]; targets=[]
    for case in rows:
        prefix=encode(prompt(case)); full=prefix+encode(answer(expected_tool(case)))
        x=full[:-1]; y=full[1:]
        y[:len(prefix)-1]=[-100]*(len(prefix)-1)
        inputs.append(x); targets.append(y)
    return torch.tensor(inputs),torch.tensor(targets)

class TinyToolLM(nn.Module):
    def __init__(self,width=32):
        super().__init__(); self.embedding=nn.Embedding(len(VOCAB),width)
        self.position=nn.Embedding(32,width)
        layer=nn.TransformerEncoderLayer(width,2,64,dropout=0,batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.head=nn.Linear(width,len(VOCAB))
    def forward(self,ids):
        n=ids.shape[1]; positions=torch.arange(n,device=ids.device)
        causal=torch.triu(torch.ones(n,n,device=ids.device,dtype=torch.bool),diagonal=1)
        return self.head(self.transformer(self.embedding(ids)+self.position(positions),mask=causal))

def lm_loss(logits,targets):
    return F.cross_entropy(logits.reshape(-1,len(VOCAB)),targets.reshape(-1),ignore_index=-100)

def train_lm(steps=220,seed=12):
    torch.manual_seed(seed); model=TinyToolLM(); opt=torch.optim.AdamW(model.parameters(),lr=.004)
    train,dev,_=partitions(); x,y=batch(train); xd,yd=batch(dev)
    best=float('inf'); checkpoint=copy.deepcopy(model.state_dict()); history=[]
    for step in range(steps):
        model.train(); opt.zero_grad(); loss=lm_loss(model(x),y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
        if step%10==0 or step==steps-1:
            model.eval()
            with torch.no_grad(): score=float(lm_loss(model(xd),yd))
            history.append((step,float(loss.detach()),score))
            if score<best: best=score; checkpoint=copy.deepcopy(model.state_dict())
    model.load_state_dict(checkpoint); model.eval()
    return model,history

@torch.no_grad()
def generate(model,case,max_tokens=8):
    prefix=encode(prompt(case)); ids=prefix.copy()
    for _ in range(max_tokens):
        nxt=int(model(torch.tensor([ids]))[0,-1].argmax()); ids.append(nxt)
        if nxt==TOKEN['<eos>']: break
    return ' '.join(VOCAB[i] for i in ids[len(prefix):] if i!=TOKEN['<eos>'])

def parse_call(text):
    try: call=json.loads(text)
    except (ValueError,TypeError): return None
    if not isinstance(call,dict) or set(call)!= {'tool'} or call['tool'] not in TOOLS: return None
    return call

@torch.no_grad()
def constrained_call(model,case):
    # Rank only valid complete calls by response-token log probability.
    # This guarantees syntax, not the scientific suitability of the selected action.
    prefix=encode(prompt(case)); scores=[]
    for tool in TOOLS:
        full=prefix+encode(answer(tool)); x=torch.tensor([full[:-1]])
        logits=model(x)[0].log_softmax(-1); y=torch.tensor(full[1:])
        score=logits[torch.arange(len(y)),y][len(prefix)-1:].sum()
        scores.append(float(score))
    return {'tool':TOOLS[max(range(len(TOOLS)),key=lambda i:scores[i])]}
