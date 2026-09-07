"""Reusable experiment bookkeeping; model definitions remain visible in lessons."""
import copy
import hashlib
import json
import platform
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .world import summaries, SUMMARY_SIGMA, MachineConfig

def seed_all(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)

def fit(model, x, y, xv, yv, steps=200, lr=.003, loss_fn=None):
    loss_fn = loss_fn or nn.MSELoss(); opt=torch.optim.AdamW(model.parameters(),lr=lr)
    best=float('inf'); state=copy.deepcopy(model.state_dict()); history=[]
    for step in range(steps):
        model.train(); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step % 10 == 0 or step == steps-1:
            model.eval()
            with torch.no_grad(): val=float(loss_fn(model(xv),yv))
            history.append((step,float(loss.detach()),val))
            if val<best: best=val; state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state); model.eval()
    return np.array(history)

def grid_posterior(observed, channels=None, n=41, sigma=None, resistance=.22):
    """Discrete uniform prior over charge,mass; nuisance resistance fixed by contract."""
    q=torch.linspace(.75,1.25,n); m=torch.linspace(.65,1.35,n)
    qq,mm=torch.meshgrid(q,m,indexing='ij')
    theta=torch.stack([qq.flatten(),mm.flatten(),qq.flatten()*0+resistance,qq.flatten()*0+.6],1)
    idx=torch.arange(12) if channels is None else torch.as_tensor(channels)
    sigma=SUMMARY_SIGMA if sigma is None else sigma
    with torch.no_grad():
        predictions=summaries(theta)
        logp=-.5*(((predictions[:,idx]-observed[idx])/sigma[idx])**2).sum(1)
        weights=torch.softmax(logp,0)
    return {'theta':theta, 'weights':weights,'predictions':predictions,'q':q,'m':m,
            'mean':(weights[:,None]*theta).sum(0)}

def weighted_interval(values,weights,level=.9):
    order=torch.argsort(values); cdf=weights[order].cumsum(0)
    probs=torch.tensor([(1-level)/2,(1+level)/2],dtype=weights.dtype)
    ids=torch.searchsorted(cdf,probs).clamp_max(len(values)-1)
    return values[order][ids]

def entropy(weights):
    return float(-(weights*torch.log(weights.clamp_min(1e-30))).sum())

def save_run(root, lesson, metrics, config=None):
    root=Path(root); run=root/'runs'/'world-model'/f'{lesson}-{uuid.uuid4().hex[:10]}'
    run.mkdir(parents=True,exist_ok=False)
    source={}
    for pattern in ('lm_testbed/*.py',f'notebooks/{lesson}_*.ipynb'):
        for p in root.glob(pattern):
            # Notebook outputs change during execution; hash source cells only.
            if p.suffix=='.ipynb':
                n=json.loads(p.read_text()); data=json.dumps([c['source'] for c in n['cells']],sort_keys=True).encode()
            else: data=p.read_bytes()
            source[str(p.relative_to(root))]=hashlib.sha256(data).hexdigest()
    report={'lesson':lesson,'created_unix':time.time(),'python':platform.python_version(),
        'torch':torch.__version__,'numpy':np.__version__,'device':'cpu',
        'scope':'exposed synthetic development evidence','config':config or {},'metrics':metrics,'source_sha256':source}
    (run/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return run
