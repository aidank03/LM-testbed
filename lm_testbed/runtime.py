"""Bounded observation-only tool interface for the teaching agent."""
import hashlib
import math
import json
import torch
from .language import parse_call
from .learning import grid_posterior, weighted_interval

def evidence_id(record):
    def canonical(value):
        if isinstance(value,torch.Tensor): return canonical(value.tolist())
        if isinstance(value,list): return [canonical(x) for x in value]
        if isinstance(value,dict): return {k:canonical(v) for k,v in value.items()}
        if isinstance(value,float) and not math.isfinite(value): return None
        return value
    public=canonical(record)
    return 'obs-'+hashlib.sha256(json.dumps(public,sort_keys=True,allow_nan=False).encode()).hexdigest()[:16]

def dispatch(call,record):
    """No evaluator truth, arbitrary execution, filesystem paths or facility control."""
    if parse_call(json.dumps(call)) is None:
        return {'status':'rejected','reason':'invalid tool schema'}
    tool=call['tool']; case=record['case']; evidence=evidence_id(record)
    if tool.startswith('request_'):
        return {'status':'needs_measurement','requested':tool.removeprefix('request_'),'evidence':[evidence]}
    if tool=='align':
        return {'status':'needs_calibration','reason':'supply independent timing fiducial; no alignment inferred by matching peaks',
                'evidence':[evidence]}
    if case['question']=='growth':
        return {'status':'unresolved','reason':'growth absent from summary operator','evidence':[evidence]}
    if not case['clock'] or not case['current'] or (case['question']=='mass' and not case['motion']):
        return {'status':'unresolved','reason':'required information missing','evidence':[evidence]}
    obs=record.get('summaries')
    if not isinstance(obs,torch.Tensor) or obs.shape!=(12,):
        return {'status':'rejected','reason':'expected 12 summary slots, NaN in missing slots'}
    # Thermal is not exposed to this agent. Motion exposes speed and radius together.
    indices=[0,1,2]+([3,4,5,6,7,8] if case['motion'] else [])
    if not torch.isfinite(obs[indices]).all():
        return {'status':'rejected','reason':'available channels must be finite'}
    p=grid_posterior(obs,channels=indices,n=31)
    k=0 if case['question']=='charge' else 1
    interval=weighted_interval(p['theta'][:,k],p['weights'])
    prediction=(p['weights'][:,None]*p['predictions']).sum(0)
    from .world import SUMMARY_SIGMA
    residual=float(((obs[indices]-prediction[indices])/SUMMARY_SIGMA[indices]).abs().max())
    if residual>4:
        return {'status':'unresolved','reason':'posterior predictive residual exceeds teaching threshold',
                'max_residual_sigma':residual,'evidence':[evidence]}
    return {'status':'conditional_estimate','parameter':case['question'],'mean':float(p['mean'][k]),
            'interval90':interval.tolist(),'max_residual_sigma':residual,'evidence':[evidence],
            'assumptions':['dimensionless reduced model','resistance fixed at .22','uniform bounded prior',
                           'known gains and aligned clocks','Gaussian reconstructed-summary likelihood']}

def run_agent(policy,record,max_calls=2):
    if type(max_calls) is not int or max_calls<1: raise ValueError('positive tool budget required')
    trace=[]
    # One proposal and one guarded call are sufficient for the current request language.
    # A missing observation is a terminal request, never magically filled with truth.
    proposal=policy(dict(record['case']))
    parsed=parse_call(proposal) if isinstance(proposal,str) else proposal
    if parsed is None:
        return {'final':{'status':'rejected','reason':'invalid generated JSON'},'trace':[],'calls':0}
    output=dispatch(parsed,record); trace.append({'call':parsed,'result':output})
    return {'final':output,'trace':trace,'calls':len(trace),'budget':max_calls}
