from pathlib import Path
import textwrap
import nbformat as nb
from add_world_model_visuals import enhance
ROOT=Path(__file__).resolve().parents[1]

def md(s): return nb.v4.new_markdown_cell(textwrap.dedent(s).strip())
def code(s): return nb.v4.new_code_cell(textwrap.dedent(s).strip())
SETUP='''
from pathlib import Path
import sys, json, copy, time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from dataclasses import asdict, replace
from IPython.display import display
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'lm_testbed').is_dir()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook from the LM-testbed repo, with lm_testbed/ beside notebooks/.')
sys.path.insert(0, str(ROOT))
from lm_testbed.world import *
from lm_testbed.learning import seed_all, fit, save_run, grid_posterior, weighted_interval, entropy
seed_all(42)
plt.rcParams.update({'figure.figsize': (10, 4), 'axes.grid': True, 'grid.alpha': .2})
DEVICE = torch.device('cpu')
print('PyTorch', torch.__version__, '| CPU | dimensionless teaching world v1')
'''

def write(num,slug,title,intro,skills,cells,exercises):
    first=[md(f'# LM-testbed {num} — {title}\n\n{intro}'),
      md(f'''## How to use this lesson
Run from the top with **Python (Factor AI)** or the environment in `docs/WORLD_MODEL_COURSE.md`. Each notebook runs independently; later notebooks reuse the small `lm_testbed` package. No API calls or model downloads occur. Training uses the CPU to make the default run portable.

**Skills:** {skills}

**Evidence contract:** all saved results are exposed synthetic development evidence. Train, tuning, calibration and evaluation draws have separate seeds where used. Re-running and tuning on these evaluation draws makes them development data; they are not a secret final test. Never split frames from one shot across train and evaluation.

**Scope:** a dimensionless, pre-stagnation Z-inspired teaching model. It is not a calibrated digital twin of Sandia's Z machine. The thermal and perturbation variables are proxies; they do not identify phase, ETI, MRTI, or fusion performance.'''),code(SETUP)]
    final=md('## Your next experiments\n\nWrite a prediction, change one thing, and save both successes and failures.\n\n'+'\n'.join(f'{i+1}. {x}' for i,x in enumerate(exercises)))
    n=nb.v4.new_notebook(cells=first+cells+[final],metadata={'kernelspec':{'display_name':'Python (Factor AI)','language':'python','name':'factor-ai'},'language_info':{'name':'python','version':'3.12'},'lm_testbed':{'course_version':'0.1.0','lesson':num,'evaluation':'public development'}})
    enhance(n,num)
    nb.validate(n); nb.write(n,ROOT/'notebooks'/f'{num}_{slug}.ipynb')

write('05','experiment_contract_and_config','from one diagnostic to an experiment world',
'''Factor becomes the application layer: it asks questions, calls tools and records evidence. LM-testbed is the workshop where we learn to build the models and interfaces behind those tools. This lesson turns “the experiment” into an explicit software contract.

Think of a model train set: the tracks are the simulator, the cameras are diagnostics, and the language model is a dispatcher. The dispatcher needs a map and a timetable before it can make useful decisions.''',
'dataclasses, validation, tensor dimensions, provenance, reproducible configuration, interface design',[
md(r'''## 1. Draw the information flow
Machine configuration and uncertain material parameters produce a hidden trajectory. Each diagnostic observes that trajectory through its own response. Inference combines observations with assumptions. An LM can choose tools and explain their outputs; it does not get the hidden answer.

A **world model** predicts how state evolves under a configuration or action. A learned state-transition model arrives in lesson 10. The present simulator supplies a small world in which to learn.'''),
code('''fig, ax = plt.subplots(figsize=(13,3)); ax.axis('off')
labels=['Machine\\nconfiguration','Hidden state\\ntrajectory','Diagnostic\\nresponses','Recorded\\nobservations','Inference +\\nLM tools']
for i,label in enumerate(labels):
    ax.text(i*2.5,0,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.7',fc='#dceaf7',ec='#37658c'))
    if i<4: ax.annotate('',xy=(i*2.5+1.7,0),xytext=(i*2.5+.8,0),arrowprops=dict(arrowstyle='->'))
ax.set(xlim=(-1.3,11.4),ylim=(-1,1)); plt.show()'''),
md(r'''## 2. Separate machine, acquisition and inference settings
A PDV STFT window belongs to processing. Charge, load mass and circuit resistance belong to the physical model. Image exposure and channel clock offsets belong to acquisition. Priors and uncertainty models belong to inference. Keeping them separate lets us change one layer without silently changing the others.'''),
code('''machine=MachineConfig()
diagnostic=DiagnosticConfig(pdv_window=64, image_exposure=.04)
request={'schema_version':'world-request-v1','question':'infer charge and mass',
         'machine':asdict(machine),'diagnostics':asdict(diagnostic),
         'inference':{'method':'discrete_grid','parameters':['charge','mass'],
                      'prior':'uniform within declared bounds','calibration':'assumed known'},
         'agent':{'max_tool_calls':3,'allowed_tools':['infer','align','request_image']}}
print(json.dumps(request,indent=2)); print('Machine SHA256:',fingerprint(machine))'''),
md(r'''## 3. Make invalid configurations fail visibly
An invalid sample count or nonpositive mass should fail at the boundary. A valid configuration only means the software accepts it; it does not establish that the physical assumptions are sound.'''),
code('''for bad in [dict(mass=-1),dict(samples=4),dict(end_time=5)]:
    try: MachineConfig(**bad)
    except ValueError as exc: print('Rejected:',bad,'→',exc)
changed=replace(machine,mass=1.15)
assert fingerprint(changed)!=fingerprint(machine)
print('Changing mass changes the configuration fingerprint.')'''),
md(r'''## 4. Inspect one shot as tensors
`states.shape = [batch, time, variables]`. A batch is a collection of independent shots, not repeated frames from one shot. We will keep the same convention throughout the course.'''),
code('''t,states=simulate()
print('time:',tuple(t.shape),'states:',tuple(states.shape))
print(dict(zip(STATE_NAMES,states[0,-1].tolist())))
fig,axes=plt.subplots(2,3,figsize=(12,6))
for j,ax in enumerate(axes.flat):
    ax.plot(t,states[0,:,j]); ax.set(title=STATE_NAMES[j],xlabel='normalized time')
plt.tight_layout(); plt.show()'''),
md(r'''## 5. Give the candidate observations, keep truth in the evaluator
The notebook author can inspect both objects. The tool interface accepts only the observation object. This is an API boundary, not an operating-system security boundary. True adversarial isolation would need separate processes and access controls.'''),
code('''public=observe(t.numpy(),states[0].numpy(),diagnostic,seed=501)
evaluator={'theta':asdict(machine),'state':states[0].numpy()}
assert 'state' not in public and 'theta' not in public
print('Public keys:',list(public)); print('Channels:',list(public['diagnostics']))
print('Image tensor:',public['diagnostics']['image']['intensity'].shape)
print('Candidate is never passed evaluator.')'''),
md(r'''## 6. Record a reproducible contract
Each run gets a fresh directory, a configuration and source fingerprints. An experiment record should eventually include raw-data hashes, calibration versions, training data, model revision, tool trace, and the decision. This lesson starts that contract.'''),
code('''run=save_run(ROOT,'05',{'channels':list(public['diagnostics']),'state_shape':list(states.shape)},request)
(run/'request.json').write_text(json.dumps(request,indent=2))
print('Saved:',run)''')],
['Add a voltage-monitor configuration field and decide which layer owns it.','Change two machine settings and explain why their diagnostic effects might be confused.','Design an observation schema for a second laboratory without changing the agent interface.'])

write('06','differentiable_machine_and_liner','a differentiable machine and liner',
'''We replace a prescribed motion trace with a coupled circuit and moving load. PyTorch now differentiates through a numerical solver. That makes the simulator useful for parameter fitting and later for training a world model.''',
'coupled ODEs, RK4 integration, autograd through time, finite-difference checks, conservation accounting',[
md(r'''## 1. Read the reduced equations
The state is capacitor voltage $V$, current $I$, radius $r$, inward speed $u$, thermal proxy $T$, and generic perturbation amplitude $a$.

$$L(r)=L_0+k\log(1/r),\quad \dot L=ku/r,$$
$$\dot V=-I/C,\quad \dot I=(V-RI-I\dot L)/L,$$
$$\dot r=-u,\quad m\dot u=kI^2/(2r)-du,$$
$$\dot T=RI^2-\ell(T-0.1),\quad \dot a=(gI^2-0.2)a.$$

All quantities and constants are dimensionless. The same inductance coupling appears in circuit and force, which lets us check an energy balance. The heat variable has an arbitrary unit heat capacity. The growth law is deliberately generic: it is not an ETI or MRTI dispersion relation. We stop before convergence becomes singular.'''),
code('''import inspect
print(inspect.getsource(rhs))'''),
md(r'''## 2. Integrate multiple configurations together
`simulate` uses explicit fourth-order Runge–Kutta. There is no neural network yet: gradients arise because the numerical arithmetic is written in PyTorch. Charge sets the initial capacitor voltage; it is not a facility charging prescription.'''),
code('''theta=torch.tensor([[.85,1.,.22,.6],[1.,1.,.22,.6],[1.15,1.,.22,.6]])
t,state=simulate(theta)
fig,axes=plt.subplots(1,3,figsize=(12,3.5))
for j,param in enumerate(theta):
    for ax,col in zip(axes,[1,2,5]): ax.plot(t,state[j,:,col],label=f'charge={param[0]:.2f}')
for ax,title in zip(axes,['Current','Radius','Generic perturbation']): ax.set(title=title,xlabel='normalized time'); ax.legend()
plt.tight_layout(); plt.show()'''),
md(r'''## 3. Differentiate a scientific output
The question is: how does final radius respond to each input? A gradient is local sensitivity under these equations. It is not a causal estimate from experimental data.'''),
code('''theta0=torch.tensor([[1.,1.,.22,.6]],dtype=torch.float64,requires_grad=True)
_,s=simulate(theta0)
r_final=s[0,-1,2]
grad,=torch.autograd.grad(r_final,theta0)
print('final radius:',float(r_final.detach()))
print(dict(zip(PARAM_NAMES,grad[0].tolist())))
# Perturbation growth does not feed back onto mean radius in this toy model.
assert abs(float(grad[0,3]))<1e-12'''),
md(r'''## 4. Verify the derivative independently
Finite differences perturb one parameter and rerun the solver. Agreement checks implementation. It cannot verify whether these are the right equations for a real liner.'''),
code('''eps=1e-4; fd=[]
for k in range(4):
    plus=theta0.detach().clone(); minus=plus.clone(); plus[0,k]+=eps; minus[0,k]-=eps
    fd.append(float((simulate(plus)[1][0,-1,2]-simulate(minus)[1][0,-1,2])/(2*eps)))
fd=np.array(fd); error=float(np.max(np.abs(fd-grad.numpy()[0])))
print('finite difference:',fd,'maximum error:',error)
assert error<1e-5'''),
md(r'''## 5. Check time-step convergence
Compare final state at three resolutions. A small error between successive grids supports solver convergence over this interval. It does not justify longer runs through stagnation or make the perturbation law physical.'''),
code('''resolutions=[48,96,192]
ends=[simulate(theta0.detach(),replace(MachineConfig(),samples=n))[1][0,-1].numpy() for n in resolutions]
coarse=float(np.max(np.abs(ends[0]-ends[2]))); fine=float(np.max(np.abs(ends[1]-ends[2])))
print({'48_vs_192':coarse,'96_vs_192':fine})
assert fine<coarse'''),
md(r'''## 6. Account for energy
For this circuit and mean-motion model,
$$E=CV^2/2+LI^2/2+mu^2/2,\qquad \dot E=-RI^2-du^2.$$
The thermal proxy is omitted from this mechanical/electrical balance because the Joule term is counted as a loss here. The passive perturbation variable carries no modeled energy. Do not add its growth to an energy budget without changing the equations.'''),
code('''cfg=MachineConfig(); td,sd=simulate(theta0.detach(),replace(cfg,samples=192)); v,i,r,u,heat,a=sd[0].T
L=cfg.inductance+cfg.coupling*torch.log(1/r)
energy=.5*cfg.capacitance*v*v+.5*L*i*i+.5*theta0.detach()[0,1]*u*u
loss=theta0.detach()[0,2]*i*i+cfg.damping*u*u
integral=torch.cat([torch.zeros(1,dtype=td.dtype),torch.cumsum(.5*(loss[1:]+loss[:-1])*torch.diff(td),0)])
balance=energy+integral
balance_error=float((balance-balance[0]).abs().max())
plt.plot(td,energy,label='stored energy'); plt.plot(td,balance,label='stored + dissipated'); plt.legend(); plt.xlabel('time'); plt.show()
print('Maximum balance residual:',balance_error)
assert balance_error<1e-4
run=save_run(ROOT,'06',{'gradient_error':error,'grid_error':fine,'energy_residual':balance_error},asdict(cfg))
print(run)''')],
['Remove the motional-inductance term in a copied RHS and observe the energy residual.','Differentiate final perturbation amplitude and compare sensitivity to charge and growth.','Add an explicit applied pulse or second circuit section, document its units, and repeat convergence and balance checks.'])

write('07','synthetic_diagnostic_suite','build the instruments as well as the machine',
'''A state trajectory is not a diagnostic. Here one hidden shot becomes a B-dot-like voltage, a PDV waveform, a sequence of projected edge images, a radiation-response trace and a toy spectrum. We reconstruct useful quantities and watch instrument settings change the answer.''',
'forward operators, signal processing, imaging, timing, calibration, synthetic diagnostic configuration',[
md(r'''## 1. What is public inspiration and what is invented?
Sandia's public [diagnostic overview](https://www.sandia.gov/app/uploads/sites/129/2022/06/Z_Diagnostics.pdf) lists electrical, velocimetry, imaging, radiation and spectroscopy families. It describes B-dot monitors as derivative-responding current diagnostics. We borrow those **families**, not instrument specifications.

Our B-dot-like response is proportional to $dI/dt$; the PDV phase integrates $2u/\lambda$; the image is a projected edge proxy; radiation is an arbitrary $T^2$ response; the spectrum is an arbitrary exponential proxy. The last three are not calibrated radiation transport or temperature diagnostics. These channels are an illustrative bundle, not a claim that they are fielded together in one Z experiment.'''),
code('''t,s=simulate(); cfg=DiagnosticConfig(noise=.012)
record=observe(t.numpy(),s[0].numpy(),cfg,seed=701)
raw=record['diagnostics']
print('Acquisition config:',record['acquisition'])
fig,axes=plt.subplots(1,3,figsize=(13,3))
for ax,name in zip(axes,['bdot','pdv','radiation']):
    ax.plot(raw[name]['time'],raw[name]['voltage']); ax.set(title=name,xlabel='normalized time',ylabel='recorded proxy voltage')
plt.tight_layout(); plt.show()'''),
md(r'''## 2. B-dot integration and its baseline problem
Voltage proportional to a derivative becomes current only after calibration and integration. A small constant voltage offset integrates into a ramp. These records are already synthetic gain-calibrated apart from the configured gain; the reconstruction uses the known initial current.'''),
code('''from scipy.integrate import cumulative_trapezoid
b=raw['bdot']; reconstructed=cumulative_trapezoid(b['voltage']/cfg.gain,b['time'],initial=0)
biased=cumulative_trapezoid((b['voltage']+.06)/cfg.gain,b['time'],initial=0)
plt.plot(t,s[0,:,1],label='simulator current'); plt.plot(t,reconstructed,label='integrated B-dot'); plt.plot(t,biased,label='voltage offset +.06')
plt.xlabel('normalized time'); plt.ylabel('current'); plt.legend(); plt.show()
bdot_error=float(np.mean(np.abs(reconstructed-s[0,:,1].numpy())))
print('B-dot reconstruction MAE:',bdot_error)'''),
md(r'''## 3. Revisit the PDV window, now in a multi-diagnostic context
For this idealized normal-incidence waveform, $f=2u/\lambda$. Real-valued cosine data gives a speed magnitude without a sign convention or extra information. Window length trades time localization against frequency resolution. The wavelength here is dimensionless. The earlier PDV notebooks remain the more detailed waveform benchmark.'''),
code('''from scipy.signal import stft
p=raw['pdv']; fs=1/np.diff(p['time']).mean(); pdv_rows=[]
fig,axes=plt.subplots(1,2,figsize=(12,4))
for window in [32,64,128]:
    freq,frames,z=stft(p['voltage'],fs=fs,nperseg=window,noverlap=window*3//4,boundary=None,padded=False)
    ridge=freq[np.abs(z).argmax(axis=0)]*cfg.pdv_wavelength/2
    truth=np.interp(frames,t,s[0,:,3]); err=float(np.abs(ridge-truth).mean())
    pdv_rows.append({'window':window,'MAE':err})
    axes[0].plot(frames,ridge,label=f'window {window}')
axes[0].plot(t,s[0,:,3],'k--',label='truth'); axes[0].legend(); axes[0].set(xlabel='time',ylabel='speed magnitude')
axes[1].pcolormesh(frames,freq*cfg.pdv_wavelength/2,np.abs(z),shading='auto'); axes[1].set(ylim=(0,.15),xlabel='time',ylabel='speed',title='Last window spectrogram')
plt.tight_layout(); plt.show(); display(pdv_rows)'''),
md(r'''## 4. Imaging and perturbation detectability
The image contains a sinusoidal edge perturbation. We extract the positive edge at a fixed intensity threshold, then estimate the known Fourier mode. This is a deliberately simple baseline: changing blur or exposure can conceal a real perturbation. Observing a mode does not establish its mechanism.'''),
code('''im=raw['image']; positive=im['x']>=0; xp=im['x'][positive]
def edge_summary(image_record):
    frame=image_record['intensity'][-1][:,positive]
    edge=xp[np.abs(frame-.675).argmin(axis=1)]
    mode=np.cos(2*np.pi*4*image_record['z'])
    design=np.stack([np.ones_like(mode),mode],axis=1)
    radius,amplitude=np.linalg.lstsq(design,edge,rcond=None)[0]
    return float(radius),float(abs(amplitude)),edge
radius,amplitude,edge=edge_summary(im)
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].imshow(im['intensity'][-1],extent=[-1.2,1.2,1,0],aspect='auto',cmap='gray'); axes[0].set(title='Synthetic projected edge',xlabel='x',ylabel='z')
axes[1].plot(im['z'],edge,label='extracted edge'); axes[1].set(xlabel='z',ylabel='radius'); axes[1].legend()
plt.tight_layout(); plt.show()
print({'reconstructed_radius':radius,'mode_amplitude':amplitude,'simulator_amplitude_at_frame':float(np.interp(im['time'][-1],t,s[0,:,5]))})'''),
code('''blur_rows=[]
for blur in [.25,1.,3.,6.]:
    observed=observe(t.numpy(),s[0].numpy(),replace(cfg,image_psf_pixels=blur),seed=702)
    rr,aa,_=edge_summary(observed['diagnostics']['image'])
    blur_rows.append({'PSF_pixels':blur,'radius':rr,'amplitude':aa})
display(blur_rows)'''),
md(r'''## 5. Radiation and spectral nuisance parameters
The same thermal proxy feeds two different observation operators. Detector response and source intensity can be confounded. The spectrum below is a Poisson-count lesson, not a physical temperature inversion. To make it physical, supply emissivity, opacity, spectral response, geometry and calibration models.'''),
code('''spec=raw['spectrum']; e=spec['energy']; c=spec['counts']
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
axes[0].plot(raw['radiation']['time'],raw['radiation']['voltage'],label='noisy proxy response'); axes[0].plot(t,s[0,:,4].numpy()**2,label='T proxy squared'); axes[0].legend()
axes[1].errorbar(e,c,yerr=np.sqrt(c+1),fmt='.',label='counts ± sqrt(counts+1)'); axes[1].set(xlabel='proxy energy',ylabel='counts'); axes[1].legend()
plt.tight_layout(); plt.show()'''),
md(r'''## 6. Timing is part of the measurement model
Identical states observed with an offset clock can appear to disagree. Our convention is `measurement(t) = response(state(t + clock_shift))`. Positive shift means the sampled state is later than the recorded time. Near the boundaries interpolation holds the endpoint, another explicit simplification.'''),
code('''late=observe(t.numpy(),s[0].numpy(),replace(cfg,clock_shift=.08,missing=('spectrum',)),seed=703)
plt.plot(t,raw['radiation']['voltage'],label='aligned'); plt.plot(t,late['diagnostics']['radiation']['voltage'],label='clock shift +.08'); plt.xlabel('recorded time'); plt.legend(); plt.show()
assert 'spectrum' not in late['diagnostics']
run=save_run(ROOT,'07',{'bdot_MAE':bdot_error,'pdv_window_scan':pdv_rows,'image_blur_scan':blur_rows},asdict(cfg))
print(run)''')],
['Add a shared clock error and independent per-channel errors. What can a timing fiducial identify?','Add PDV dropout or a secondary return from notebook 04 and propagate its uncertainty into a reconstruction.','Replace the edge proxy with a line-integrated attenuation model; document what physical inputs it requires.'])

write('08','joint_inference_and_identifiability','infer the machine through several diagnostics',
'''The machine is hidden; diagnostics constrain it. We ask which parameters can be distinguished, then fit several parameters jointly. More channels help only if they add independent information and their calibration is modeled.''',
'likelihoods, parameter grids, Jacobians, singular values, joint optimization, nuisance parameters',[
md(r'''## 1. State the inverse problem precisely
For speed, this lesson starts from **12 reconstructed summaries**, not raw waveforms: current, inward speed, radius and thermal proxy, each at three times. `SUMMARY_SIGMA` declares independent Gaussian errors. This is a simplified replacement for the reconstruction layer from lesson 07; its uncertainties are authored, not measured from those reconstructions.

First infer charge and mass with resistance fixed at .22. Growth is absent from these mean-state summaries. Later fit current-channel gain too. These are conditional inferences, not complete inference over every possible uncertainty.'''),
code('''truth=torch.tensor([[1.08,.88,.22,.65]])
seed_all(801)
with torch.no_grad(): obs=summaries(truth)[0]+torch.randn(12)*SUMMARY_SIGMA
current_only=grid_posterior(obs,channels=[0,1,2])
joint=grid_posterior(obs)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,p,title in zip(axes,[current_only,joint],['Current only','All four summary families']):
    ax.imshow(p['weights'].reshape(41,41).T,origin='lower',extent=[.75,1.25,.65,1.35],aspect='auto')
    ax.scatter(float(truth[0,0]),float(truth[0,1]),marker='x',c='red'); ax.set(title=title,xlabel='charge',ylabel='mass')
plt.tight_layout(); plt.show()
print('Joint posterior mean:',joint['mean'][:2].tolist())'''),
md(r'''## 2. Inspect local identifiability
A Jacobian measures how observations change with parameters. Dividing by measurement sigma puts changes in noise units; scaling parameter columns by their prior ranges makes singular values more comparable. A zero column means a parameter cannot be inferred through this operator, regardless of optimizer or LM size.'''),
code('''z=truth[0].clone().requires_grad_()
jac=torch.autograd.functional.jacobian(lambda p:summaries(p[None])[0],z)
scaled=jac/SUMMARY_SIGMA[:,None]*torch.tensor(HIGH-LOW)[None,:]
sv=torch.linalg.svdvals(scaled)
print('Scaled singular values:',sv.detach().numpy())
print('Growth column norm:',float(torch.linalg.vector_norm(jac[:,3])))
assert torch.all(jac[:,3]==0)
plt.imshow(scaled.detach().numpy(),aspect='auto',cmap='coolwarm'); plt.xticks(range(4),PARAM_NAMES); plt.yticks(range(12),SUMMARY_NAMES); plt.colorbar(label='change / sigma per prior range'); plt.show()'''),
md(r'''## 3. Fit an uncertain calibration gain
The synthetic current summaries are now multiplied by an unknown gain. Fitting charge alone can absorb calibration error. We fit charge, mass and gain using all channels and a Gaussian gain calibration prior. Resistance remains fixed, and observations are assumed synchronized. Timing uncertainty is treated in the negative control below, not silently optimized away.'''),
code('''GAIN_TRUE=1.12
with torch.no_grad():
    gain_obs=summaries(truth)[0].clone(); gain_obs[:3]*=GAIN_TRUE
    gain_obs+=torch.randn(12)*SUMMARY_SIGMA

def fit_gain(free_gain=True,steps=100):
    raw=nn.Parameter(torch.zeros(3)); opt=torch.optim.Adam([raw],lr=.06); losses=[]
    for step in range(steps):
        opt.zero_grad()
        charge=.75+.5*torch.sigmoid(raw[0]); mass=.65+.7*torch.sigmoid(raw[1])
        gain=.8+.4*torch.sigmoid(raw[2]) if free_gain else torch.tensor(1.)
        theta=torch.stack([charge,mass,charge*0+.22,charge*0+.6])[None]
        pred=summaries(theta)[0]
        pred=pred*torch.cat([gain.expand(3),torch.ones(9)])
        loss=.5*(((pred-gain_obs)/SUMMARY_SIGMA)**2).sum()+.5*((gain-1)/.15)**2
        loss.backward(); opt.step(); losses.append(float(loss.detach()))
    return {'charge':float(charge.detach()),'mass':float(mass.detach()),'gain':float(gain.detach()),'loss':losses[-1]},losses
fixed,h0=fit_gain(False); free,h1=fit_gain(True)
print('Fixed gain:',fixed); print('Inferred gain:',free)
plt.semilogy(h0,label='gain fixed at 1'); plt.semilogy(h1,label='gain with calibration prior'); plt.xlabel('step'); plt.ylabel('negative log posterior'); plt.legend(); plt.show()'''),
md(r'''## 4. A timing nuisance creates correlated errors
A clock shift changes many samples together. Here we linearize its contribution as $\Sigma = D + \sigma_t^2 jj^T$, where $j=dy/dt$. This covariance is local and first-order. For a large shift, nonlinear delay fitting would be needed. Independent-error likelihoods can overstate the amount of information.'''),
code('''tt,ss=simulate(truth); ids=[32,64,95]
derivative=torch.gradient(ss[0],spacing=(tt,),dim=0)[0]
jtime=torch.cat([derivative[ids,j] for j in (1,3,2,4)])
clock_sigma=.035
cov=torch.diag(SUMMARY_SIGMA.square())+clock_sigma**2*torch.outer(jtime,jtime)
shifted_obs=obs+.05*jtime
pred=joint['predictions']; residual=pred-shifted_obs
logp=-.5*(residual*torch.linalg.solve(cov,residual.T).T).sum(1)
wc=torch.softmax(logp,0)
naive=grid_posterior(shifted_obs)
print('Mass interval ignoring shared timing:',weighted_interval(naive['theta'][:,1],naive['weights']).tolist())
print('Mass interval including timing covariance:',weighted_interval(joint['theta'][:,1],wc).tolist())
plt.imshow(cov.numpy(),cmap='magma'); plt.colorbar(label='summary covariance'); plt.title('Shared clock couples errors across channels'); plt.show()'''),
md(r'''## 5. Ask whether the assumed model predicts the observations
Posterior predictive residuals are a model check. They are not independent validation because these data were used to fit. A discrepancy can arise from wrong physics, calibration, noise assumptions, or reconstruction errors. A small residual does not establish that the model is unique.'''),
code('''mean_prediction=(joint['weights'][:,None]*joint['predictions']).sum(0)
res=(obs-mean_prediction)/SUMMARY_SIGMA
plt.axhspan(-2,2,alpha=.1); plt.plot(res,'o-'); plt.xticks(range(12),SUMMARY_NAMES,rotation=70); plt.ylabel('residual / sigma'); plt.tight_layout(); plt.show()
run=save_run(ROOT,'08',{'posterior_mean':joint['mean'].tolist(),'scaled_singular_values':sv.tolist(),
                       'fixed_gain_fit':fixed,'free_gain_fit':free,'max_standardized_residual':float(res.abs().max())},
             {'likelihood':'reconstructed summaries, independent Gaussian except timing control','resistance_fixed':.22})
print(run)''')],
['Add an image-derived perturbation summary. Does the growth column become nonzero, and is it still confused with seed amplitude?','Jointly fit resistance and current gain. Plot the tradeoff before choosing a larger network.','Replace the linearized timing covariance with an explicit latent clock offset and compare inferred uncertainty.'])

write('09','amortized_probabilistic_inference','learn an inverse model with uncertainty',
'''Instead of solving a fresh grid for every shot, train a network to predict a distribution over machine parameters. This is amortized inference: pay for simulations and training once, then make cheap predictions. We check its intervals on new shots and under model mismatch.''',
'conditional density estimation, Gaussian NLL, train-only normalization, calibration, coverage and width',[
md(r'''## 1. Define the information and parameter contract
Infer charge, mass and resistance from the same 12 summary measurements. Growth remains unobservable and is excluded from the target. Training samples come from explicit uniform parameter ranges. The network outputs a diagonal Gaussian in normalized parameter coordinates; it cannot express all posterior correlations or multiple modes. Do not mistake its mean for a complete posterior.

Independent seeds: train 9001, tuning 9101, calibration 9201, evaluation 9301, misspecified evaluation 9401. These are public development splits.'''),
code('''x,y=summary_dataset(384,9001); xv,yv=summary_dataset(96,9101)
xc,yc=summary_dataset(128,9201); xe,ye=summary_dataset(128,9301)
xshift,yshift=summary_dataset(128,9401,cfg=replace(MachineConfig(),coupling=.18,thermal_loss=.3))
xmean=x.mean(0); xscale=x.std(0).clamp_min(.01)
lo=torch.tensor(LOW[:3]); span=torch.tensor(HIGH[:3]-LOW[:3])
normx=lambda a:(a-xmean)/xscale
normy=lambda a:(a[:,:3]-lo)/span
print('Train',x.shape,'evaluation',xe.shape)'''),
md(r'''## 2. Predict mean and scale, then minimize negative log likelihood
`softplus` makes the scale positive. Large uncertainty can reduce the cost of a residual, but the log-scale term penalizes making every interval huge. The minimum scale is a modeling choice. The data likelihood and simulation prior determine what the network learns.'''),
code('''class GaussianInverse(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(12,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,6))
    def forward(self,x):
        raw=self.net(x); return torch.cat([raw[:,:3],.02+torch.nn.functional.softplus(raw[:,3:])],1)
def nll(output,target):
    mu,sigma=output[:,:3],output[:,3:]
    return (.5*((target-mu)/sigma).square()+torch.log(sigma)).mean()
seed_all(9); model=GaussianInverse()
history=fit(model,normx(x),normy(y),normx(xv),normy(yv),steps=280,lr=.003,loss_fn=nll)
plt.plot(history[:,0],history[:,1],label='train NLL'); plt.plot(history[:,0],history[:,2],label='tuning NLL'); plt.xlabel('step'); plt.legend(); plt.show()'''),
md(r'''## 3. Calibrate intervals on a separate set
For each parameter, score absolute error divided by predicted scale. Use the finite-sample split-conformal rank $\lceil(n+1)0.9\rceil$. This gives marginal 90% coverage under exchangeability with calibration shots; it is not conditional coverage for every regime or simultaneous coverage of all parameters. Shifted physics breaks that assumption.'''),
code('''with torch.no_grad():
    pc=model(normx(xc)); score=(normy(yc)-pc[:,:3]).abs()/pc[:,3:]
rank=int(np.ceil((len(xc)+1)*.9)); multiplier=score.sort(dim=0).values[min(rank,len(xc))-1]
print('90% calibration multipliers:',multiplier.tolist())
def evaluate_inverse(xx,yy):
    with torch.no_grad(): p=model(normx(xx))
    mu=p[:,:3]*span+lo; width=multiplier*p[:,3:]*span
    covered=(yy[:,:3]-mu).abs()<=width
    return {'MAE':(yy[:,:3]-mu).abs().mean(0).tolist(),'coverage90':covered.float().mean(0).tolist(),
            'mean_width90':(2*width).mean(0).tolist()},mu,width
matched,mu,width=evaluate_inverse(xe,ye); shifted,mus,ws=evaluate_inverse(xshift,yshift)
print('Matched:',matched); print('Changed coupling + thermal loss:',shifted)
prior_mae=(ye[:,:3]-(lo+span/2)).abs().mean(0)
print('Prior-midpoint baseline MAE:',prior_mae.tolist())'''),
code('''fig,axes=plt.subplots(1,3,figsize=(13,4))
for j,ax in enumerate(axes):
    order=torch.argsort(ye[:,j])[:24]
    ax.errorbar(ye[order,j],mu[order,j],yerr=width[order,j],fmt='.',alpha=.7)
    ax.plot([LOW[j],HIGH[j]],[LOW[j],HIGH[j]],'k--'); ax.set(title=PARAM_NAMES[j],xlabel='truth',ylabel='prediction ± calibrated half-width')
plt.tight_layout(); plt.show()'''),
md(r'''## 4. Check coverage and sharpness together
An interval that always spans the whole prior may cover well but teach us little. A narrow interval that misses is overconfident. The mismatch evaluation changes the forward model without changing the parameter range; this tests a different failure from simple parameter extrapolation.'''),
code('''fig,axes=plt.subplots(1,2,figsize=(11,4)); pos=np.arange(3)
for offset,stats,label in [(-.17,matched,'matched'),(.17,shifted,'model mismatch')]:
    axes[0].bar(pos+offset,stats['coverage90'],width=.34,label=label)
    axes[1].bar(pos+offset,np.array(stats['mean_width90'])/span.numpy(),width=.34,label=label)
axes[0].axhline(.9,color='k',ls='--'); axes[0].set(ylim=(0,1),ylabel='empirical coverage')
axes[1].set(ylabel='interval width / prior range')
for ax in axes: ax.set_xticks(pos,PARAM_NAMES[:3]); ax.legend()
plt.tight_layout(); plt.show()
run=save_run(ROOT,'09',{'matched':matched,'model_mismatch':shifted,'prior_midpoint_MAE':prior_mae.tolist(),'calibration_multiplier':multiplier.tolist()},
             {'train':384,'tuning':96,'calibration':128,'evaluation':128,'steps':280})
torch.save({'state_dict':model.state_dict(),'xmean':xmean,'xscale':xscale,'lo':lo,'span':span,'multiplier':multiplier},run/'inverse_model.pt')
print(run)''')],
['Replace the diagonal Gaussian with a Cholesky covariance head and visualize correlations.','Stratify coverage by charge quartile; marginal coverage may hide a weak region.','Train on multiple forward-model families and compare interval width and mismatch coverage.'])

write('10','learned_dynamics_world_model','learn a world that can roll forward',
'''A world model should predict a sequence, not just a final label. Train a residual network to advance the machine state one step, then repeatedly feed its predictions back into itself. Small one-step errors may compound into a poor future.''',
'state transitions, teacher forcing, autoregressive rollout, train-only scaling, extrapolation and baselines',[
md(r'''## 1. Define what this world model gets to see
This is supervised system identification using **full simulator states** and known machine parameters during training. It is easier than inferring hidden states from diagnostics. Lesson 11 addresses observation encoding; the capstone links inference to future predictions. The step length is fixed by this notebook's time grid and is part of the model contract.'''),
code('''cfg=replace(MachineConfig(),samples=32)
train_theta=draw_parameters(96,10001); dev_theta=draw_parameters(24,10101)
eval_theta=draw_parameters(24,10201); shift_theta=draw_parameters(24,10301,shifted=True)
with torch.no_grad():
    t,train_s=simulate(train_theta,cfg); _,dev_s=simulate(dev_theta,cfg)
    _,eval_s=simulate(eval_theta,cfg); _,shift_s=simulate(shift_theta,cfg)
center=train_s.mean((0,1)); scale=train_s.std((0,1)).clamp_min(.005)
pcenter=train_theta.mean(0); pscale=train_theta.std(0).clamp_min(.01)
def transitions(s,p):
    state=(s-center)/scale
    controls=((p-pcenter)/pscale)[:,None,:].expand(-1,s.shape[1]-1,-1)
    features=torch.cat([state[:,:-1],controls],-1).reshape(-1,10)
    delta=(state[:,1:]-state[:,:-1]).reshape(-1,6)
    return features,delta
x,y=transitions(train_s,train_theta); xv,yv=transitions(dev_s,dev_theta)
print('Training transitions:',x.shape,'independent training shots:',len(train_theta))'''),
md(r'''## 2. Learn a residual update
The network predicts a change in normalized state. The identity path already preserves the current state. This tends to be easier to learn than predicting the entire next state from scratch. A residual architecture alone does not enforce energy conservation or positive radius.'''),
code('''seed_all(10)
net=nn.Sequential(nn.Linear(10,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,6))
h=fit(net,x,y,xv,yv,steps=260,lr=.003)
plt.semilogy(h[:,0],h[:,1],label='train delta MSE'); plt.semilogy(h[:,0],h[:,2],label='tuning delta MSE'); plt.legend(); plt.xlabel('step'); plt.show()
@torch.no_grad()
def rollout(model,initial,parameters,steps):
    current=(initial-center)/scale; path=[initial]
    p=(parameters-pcenter)/pscale
    for _ in range(steps-1):
        current=current+model(torch.cat([current,p],1))
        path.append(current*scale+center)
    return torch.stack(path,1)
pred=rollout(net,eval_s[:,0],eval_theta,len(t)); shifted_pred=rollout(net,shift_s[:,0],shift_theta,len(t))'''),
md(r'''## 3. Compare free rollout to useful baselines
Persistence predicts no change. Euler uses the generating RHS with a cheaper first-order solver, so it has privileged equation knowledge. RK4 is our reference. A learned model is not automatically useful if a cheap known-equation solver is better; report that result.'''),
code('''@torch.no_grad()
def euler(initial,parameters):
    state=initial.clone(); path=[state]; dt=t[1]-t[0]
    for _ in range(len(t)-1):
        state=state+dt*rhs(state,parameters,cfg); path.append(state)
    return torch.stack(path,1)
euler_pred=euler(eval_s[:,0],eval_theta)
persist=eval_s[:,0:1].expand_as(eval_s)
def normalized_rmse(a,b): return float(torch.mean(((a-b)/scale)**2).sqrt())
metrics={'learned_matched':normalized_rmse(pred,eval_s),'Euler_matched':normalized_rmse(euler_pred,eval_s),
         'persistence_matched':normalized_rmse(persist,eval_s),'learned_shifted':normalized_rmse(shifted_pred,shift_s)}
print(metrics)'''),
code('''fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,j in zip(axes,[1,2,5]):
    ax.plot(t,eval_s[0,:,j],'k',label='RK4 reference'); ax.plot(t,pred[0,:,j],label='learned rollout'); ax.plot(t,euler_pred[0,:,j],'--',label='Euler')
    ax.set(title=STATE_NAMES[j],xlabel='time'); ax.legend()
plt.tight_layout(); plt.show()'''),
md(r'''## 4. Measure accumulation and domain failures
One-step evaluation gives the model true states at every step. Free rollout uses its own history. Plot error versus prediction horizon. Record unphysical radius predictions instead of clipping them out of the score.'''),
code('''with torch.no_grad():
    xe,ye=transitions(eval_s,eval_theta); one_step=float(nn.functional.mse_loss(net(xe),ye))
horizon=((pred-eval_s)/scale).square().mean((0,2)).sqrt()
shift_horizon=((shifted_pred-shift_s)/scale).square().mean((0,2)).sqrt()
plt.plot(t,horizon,label='matched'); plt.plot(t,shift_horizon,label='shifted'); plt.xlabel('rollout horizon'); plt.ylabel('normalized RMSE'); plt.legend(); plt.show()
metrics['one_step_delta_MSE']=one_step
metrics['invalid_radius_fraction']=float(((shifted_pred[:,:,2]<=0)|(shifted_pred[:,:,2]>1.05)).float().mean())
print('one-step MSE:',one_step,'invalid radius fraction:',metrics['invalid_radius_fraction'])
run=save_run(ROOT,'10',metrics,{'dt':float(t[1]-t[0]),'steps':260,'state_information':'full simulator state'})
torch.save({'state_dict':net.state_dict(),'state_center':center,'state_scale':scale,'parameter_center':pcenter,'parameter_scale':pscale,'config':asdict(cfg)},run/'world_model.pt')
print(run)''')],
['Train on five-step rollouts and compare long-horizon error at equal compute budget.','Add an energy-balance penalty and check whether it improves both matched and shifted predictions.','Introduce a time-varying control, then pass it to the transition model explicitly rather than hiding it in the state.'])

write('11','multimodal_attention_and_missing_data','attention over diagnostic evidence',
'''Build an encoder that treats diagnostic families as tokens. It must combine information and handle missing instruments. This introduces embeddings and attention before the next lesson turns those ideas into a causal language model.''',
'modality embeddings, attention masks, missingness, masked pooling, linear baselines, channel ablation',[
md(r'''## 1. Each diagnostic family is a token
This lesson uses reconstructed summaries: four tokens contain current, speed, radius and thermal proxy at three times. It does **not** encode raw images or waveform patches yet. A modality embedding tells the network which instrument a token represents.

An unavailable diagnostic is masked, not replaced by a convincing-looking zero measurement. The same mask also goes into the conventional regression baseline. Training and evaluation use identical permitted observations.'''),
code('''x,y=summary_dataset(384,11001); xv,yv=summary_dataset(96,11101); xe,ye=summary_dataset(128,11201)
mean=x.mean(0); scale=x.std(0).clamp_min(.01); lo=torch.tensor(LOW[:3]); span=torch.tensor(HIGH[:3]-LOW[:3])
normalize=lambda xx:((xx-mean)/scale).reshape(-1,4,3)
xt,vt,et=normalize(x),normalize(xv),normalize(xe)
yt=(y[:,:3]-lo)/span; yvt=(yv[:,:3]-lo)/span

def missing_mask(n,seed):
    gen=torch.Generator().manual_seed(seed); mask=torch.rand(n,4,generator=gen)<.25
    # Always retain at least one token; all-missing records require abstention.
    mask[mask.all(1),0]=False
    return mask
mt=missing_mask(len(x),11); mv=missing_mask(len(xv),12); me=missing_mask(len(xe),13)
print('Missing-token fraction:',float(mt.float().mean()))'''),
md(r'''## 2. Build an attention encoder
The mask prevents missing tokens from serving as keys/values. Masked pooling excludes their outputs. Attention weights are a computation inside a predictive model; they are not causal explanations of the experiment.'''),
code('''class DiagnosticTransformer(nn.Module):
    def __init__(self):
        super().__init__(); self.project=nn.Linear(3,32); self.modality=nn.Embedding(4,32)
        layer=nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.head=nn.Linear(32,3)
    def forward(self,values,mask):
        if mask.all(1).any(): raise ValueError('all diagnostics missing: abstain')
        values=values.masked_fill(mask[:,:,None],0)
        tokens=self.project(values)+self.modality(torch.arange(4))[None]
        hidden=self.encoder(tokens,src_key_padding_mask=mask)
        keep=(~mask).float()[:,:,None]
        return self.head((hidden*keep).sum(1)/keep.sum(1))
seed_all(11); model=DiagnosticTransformer(); optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
best=float('inf'); history=[]; checkpoint=copy.deepcopy(model.state_dict())
for step in range(220):
    model.train(); optimizer.zero_grad(); loss=nn.functional.mse_loss(model(xt,mt),yt); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
    if step%10==0 or step==219:
        model.eval()
        with torch.no_grad(): val=float(nn.functional.mse_loss(model(vt,mv),yvt))
        history.append((step,float(loss.detach()),val))
        if val<best: best=val; checkpoint=copy.deepcopy(model.state_dict())
model.load_state_dict(checkpoint); model.eval()
h=np.array(history); plt.plot(h[:,0],h[:,1],label='train'); plt.plot(h[:,0],h[:,2],label='tuning'); plt.legend(); plt.xlabel('step'); plt.ylabel('normalized parameter MSE'); plt.show()'''),
md(r'''## 3. Give ridge regression the same inputs
The regression sees zero-filled standardized summaries plus a four-bit availability vector. Its regularization is fixed at .1 for this lesson. A nonlinear Transformer is not guaranteed to beat this baseline on a small smooth inverse problem.'''),
code('''def flat_features(values,mask):
    return torch.cat([values.masked_fill(mask[:,:,None],0).flatten(1),(~mask).float(),torch.ones(len(values),1)],1)
a=flat_features(xt,mt); ridge=torch.linalg.solve(a.T@a+.1*torch.eye(a.shape[1]),a.T@yt)
@torch.no_grad()
def score_mask(mask):
    neural=model(et,mask)*span+lo; linear=flat_features(et,mask)@ridge*span+lo
    return {'attention_MAE':(neural-ye[:,:3]).abs().mean(0).tolist(),'ridge_MAE':(linear-ye[:,:3]).abs().mean(0).tolist()}
results={'all_available':score_mask(torch.zeros_like(me)),'random_missing':score_mask(me)}
for j,name in enumerate(['current','speed','radius','thermal_proxy']):
    mask=torch.zeros_like(me); mask[:,j]=True; results['without_'+name]=score_mask(mask)
display(results)'''),
md(r'''## 4. Verify that masked values cannot leak into the prediction
Corrupt missing-token values by a huge amount. Predictions should not move, because those values are erased and attention masks remove their influence. This is an interface invariant, not a benchmark score.'''),
code('''with torch.no_grad():
    corrupted=et.clone(); corrupted[me]=1e5
    difference=float((model(et,me)-model(corrupted,me)).abs().max())
print('Maximum change from corrupted masked inputs:',difference)
assert difference<1e-6
try: model(et[:1],torch.ones(1,4,dtype=torch.bool))
except ValueError as exc: print('All-missing record:',exc)
fig,ax=plt.subplots(); names=list(results)
ax.barh(names,[np.mean(np.array(results[k]['attention_MAE'])/span.numpy()) for k in names],label='attention')
ax.set(xlabel='MAE / prior range, averaged across parameters',title='Diagnostic ablation: inspect per-parameter numbers above'); plt.show()
run=save_run(ROOT,'11',{'ablation':results,'masked_value_invariance':difference},{'train_shots':384,'steps':220,'inputs':'four summary tokens'})
print(run)''')],
['Replace the summary-token encoder with waveform CNNs and image patches, while preserving the observation boundary.','Change missingness to depend on signal strength; random-dropout training may not generalize.','Predict a probability distribution rather than a point estimate and recalibrate separately for missing-channel regimes.'])

write('12','tiny_language_model_for_tool_calls','train a small language model to call tools',
'''Now LM means language model. Train a tiny causal Transformer from scratch to produce a structured tool call from an experiment request. Learn tokenization, embeddings, causal attention, next-token loss, answer masking, generation and constrained output.

This is a deliberately small command language, not a pretrained assistant. Its limitations are useful because the complete training loop fits on the screen.''',
'causal language modeling, token vocabulary, response-only loss, autoregressive generation, structured outputs',[
md(r'''## 1. The language and its supervision
A prompt encodes a question, available current/motion channels, and timing calibration. A completion is JSON containing one tool name. Twenty-four semantic cases are partitioned into 16 training, four tuning, and four evaluation cases. These small counts are for learning the machinery, not establishing general language understanding.

The labels are an authored workflow policy: obtain missing information, request a timing calibration when required, then infer. A growth question needs imaging because the summary operator cannot observe growth.'''),
code('''from lm_testbed.language import TinyToolLM, VOCAB, TOKEN, partitions, batch, lm_loss, prompt, answer, expected_tool, generate, parse_call, constrained_call
train,dev,evaluation=partitions()
print('Vocabulary:',VOCAB)
print('Prompt:',prompt(train[0])); print('Answer:',answer(expected_tool(train[0])))
x,y=batch(train); xd,yd=batch(dev)
print('Token tensor shape:',x.shape,'| scored response positions:',int((y[0]!=-100).sum()))
print('Target tokens (-100 are ignored):',y[0].tolist())'''),
md(r'''## 2. Causal attention and response masking do different jobs
The causal mask prevents a position from looking at future answer tokens. The `-100` loss labels stop us from scoring prompt reconstruction. The prompt still supplies context. During generation, no correct response tokens are supplied.

Unlike a normal text tokenizer, this vocabulary uses whitespace-separated tokens and treats each tool name as one token. That keeps the lesson small but limits transfer to real prose.'''),
code('''import inspect
print(inspect.getsource(TinyToolLM))
model=TinyToolLM()
print('Trainable parameters:',sum(p.numel() for p in model.parameters()))
model.eval()
with torch.no_grad():
    original=model(x[:1]); altered=x[:1].clone(); altered[0,-1]=TOKEN['missing']; changed=model(altered)
causal_error=float((original[:,:-1]-changed[:,:-1]).abs().max())
assert causal_error<1e-6
print('Earlier logits unchanged by changing final input token:',causal_error)'''),
md(r'''## 3. The language-model training loop
Forward computes token logits. Cross entropy scores only response tokens. Backward computes gradients. AdamW adjusts the Transformer. Select the checkpoint using tuning loss, then generate on semantic cases absent from training. This is actual local LM training; no external model or service is used.'''),
code('''seed_all(12); model=TinyToolLM(); optimizer=torch.optim.AdamW(model.parameters(),lr=.004)
best=float('inf'); checkpoint=copy.deepcopy(model.state_dict()); history=[]
STEPS=220
for step in range(STEPS):
    model.train(); optimizer.zero_grad(); logits=model(x); loss=lm_loss(logits,y)
    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
    if step%10==0 or step==STEPS-1:
        model.eval()
        with torch.no_grad(): val=float(lm_loss(model(xd),yd))
        history.append((step,float(loss.detach()),val))
        if val<best: best=val; checkpoint=copy.deepcopy(model.state_dict())
model.load_state_dict(checkpoint); model.eval()
h=np.array(history); plt.plot(h[:,0],h[:,1],label='train response loss'); plt.plot(h[:,0],h[:,2],label='tuning response loss'); plt.legend(); plt.xlabel('step'); plt.show()'''),
md(r'''## 4. Compare unconstrained generation with valid-call scoring
Unconstrained greedy generation can emit malformed JSON or the wrong action. Constrained scoring ranks only complete valid JSON calls. It guarantees the limited syntax, not the right decision. We keep both scores and retain failures. The baseline always calls `infer`; it may look successful on easy cases while failing missing-data cases.'''),
code('''rows=[]
for case in evaluation:
    text=generate(model,case); parsed=parse_call(text); constrained=constrained_call(model,case)
    expected=expected_tool(case)
    rows.append({'case':case,'expected':expected,'generated':text,'valid':parsed is not None,
                 'raw_correct':parsed is not None and parsed['tool']==expected,
                 'constrained_correct':constrained['tool']==expected,'constrained_tool':constrained['tool'],
                 'always_infer_correct':expected=='infer'})
display(rows)
metrics={key:sum(int(r[key]) for r in rows)/len(rows) for key in ['valid','raw_correct','constrained_correct','always_infer_correct']}
metrics['evaluation_cases']=len(rows)
print(metrics)'''),
md(r'''## 5. Save and reload the actual learned weights
A reproducible checkpoint includes vocabulary and model configuration, not just floating-point weights. Loading this file in another process should preserve predictions. The saved weights are a teaching model, not Factor's existing Qwen adapter.'''),
code('''run=save_run(ROOT,'12',metrics,{'steps':STEPS,'seed':12,'train_cases':16,'tuning_cases':4,'evaluation_cases':4})
torch.save({'state_dict':model.state_dict(),'vocabulary':VOCAB,'width':32},run/'tiny_tool_lm.pt')
loaded=TinyToolLM(); checkpoint=torch.load(run/'tiny_tool_lm.pt',map_location='cpu',weights_only=True)
assert checkpoint['vocabulary']==VOCAB
loaded.load_state_dict(checkpoint['state_dict']); loaded.eval()
assert generate(loaded,evaluation[0])==generate(model,evaluation[0])
(run/'generated_evaluation.json').write_text(json.dumps(rows,indent=2))
print('Reload verified:',run)'''),
md(r'''## Bridge to a pretrained LM
The same next-token loss, prompt masking and evaluation logic also apply to a pretrained model. Factor's existing PyTorch lab uses a small Qwen model and LoRA; that is a natural next experiment after this notebook. Keep the tool schema fixed while changing model/provider. A real text tokenizer, chat template, adapter target modules and context length then become explicit configuration. Do not claim transfer until the new model is evaluated on fresh requests.''')],
['Repeat with three initialization seeds and report every held-out decision, not just average loss.','Add voltage and imaging availability to the language, then split by semantic request families.','Compare base and LoRA-tuned local models using the same tool interface and a fresh authored evaluation set.'])

write('13','language_model_tools_and_evaluation','connect the LM to inference tools',
'''The language model proposes an action. A bounded runtime validates it, executes a numerical tool, and returns a traceable result. We separate syntax, tool choice, numerical execution and scientific adequacy, because success at one does not guarantee success at the others.''',
'agent interfaces, JSON schema validation, tool boundaries, evidence references, behavioral evaluation, failure records',[
md(r'''## 1. The candidate receives only permitted observations
This notebook uses the tiny LM from lesson 12, retrained locally with the same fixed recipe so it runs independently. Each record contains the request and reconstructed summaries. Missing values are actually removed from the available information by replacing their slots with NaN. The thermal family is not exposed in this agent task. Evaluator truth stays separate.

This lesson runs a bounded single-decision episode. Asking for a missing diagnostic ends the episode as `needs_measurement`; the system never invents the requested measurement.'''),
code('''from lm_testbed.language import train_lm, partitions, expected_tool, generate, constrained_call
from lm_testbed.runtime import run_agent, dispatch
seed_all(12); lm,lm_history=train_lm()
_,_,cases_eval=partitions()
records=[]; truths=[]
for i,case in enumerate(cases_eval):
    theta=draw_parameters(1,13000+i); theta[:,2]=.22
    with torch.no_grad(): summary=summaries(theta)[0]+torch.randn(12)*SUMMARY_SIGMA
    summary[9:]=float('nan')
    if not case['current']: summary[:3]=float('nan')
    if not case['motion']: summary[3:9]=float('nan')
    records.append({'case':case,'summaries':summary,'shot_id':f'development-{i}'})
    truths.append(theta[0])
print('First permitted request:',records[0]['case'])
print('First permitted summary slots:',records[0]['summaries'])'''),
md(r'''## 2. Inspect the tool contract
`infer` returns a conditional grid estimate or abstains. It requires the relevant channels and known timing. Resistance is fixed at .22. The tool rejects a large posterior predictive residual using a teaching threshold of four sigma. That residual gate is heuristic and has no guaranteed false-alarm rate.

`align` requests an independent calibration. It does not line up unrelated peaks. `request_*` returns a measurement request. Every valid result links to a hash of the permitted observation record. An evidence ID establishes provenance, not physical correctness.'''),
code('''import inspect
print(inspect.getsource(dispatch))'''),
md(r'''## 3. Compare policies on the same records
The conventional rule is the authored policy used to label the toy LM data. It is therefore a strong baseline here. “Always infer” tests runtime guard behavior. Raw LM and constrained LM differ in output syntax handling. Model choice is scored independently of the runtime's ability to prevent an invalid estimate.'''),
code('''policies={'rule':lambda case:{'tool':expected_tool(case)},
          'always infer':lambda case:{'tool':'infer'},
          'raw LM':lambda case:generate(lm,case),
          'constrained LM':lambda case:constrained_call(lm,case)}
rows=[]; traces=[]
for name,policy in policies.items():
    for record,truth in zip(records,truths):
        start=time.perf_counter(); result=run_agent(policy,record,max_calls=2)
        call=result['trace'][0]['call']['tool'] if result['trace'] else None
        final=result['final']; expected=expected_tool(record['case'])
        row={'policy':name,'shot':record['shot_id'],'tool':call,'expected_tool':expected,
             'tool_correct':call==expected,'status':final['status'],'calls':result['calls'],
             'seconds':time.perf_counter()-start,'truth_covered':None}
        if final['status']=='conditional_estimate':
            index=0 if record['case']['question']=='charge' else 1
            row['truth_covered']=final['interval90'][0]<=float(truth[index])<=final['interval90'][1]
        rows.append(row); traces.append({'policy':name,'shot':record['shot_id'],'episode':result})
display(rows)
metrics={name:{'tool_accuracy':np.mean([r['tool_correct'] for r in rows if r['policy']==name]),
               'estimated_cases':sum(r['status']=='conditional_estimate' for r in rows if r['policy']==name)} for name in policies}
print(metrics)'''),
md(r'''## 4. Test the boundary separately from the model
Malformed or unlisted calls must fail. A plausible request with missing information must not receive an unconditional estimate. The runtime never executes code produced by the model. This is a closed-schema teaching interface, not a full natural-language prompt-injection defense.'''),
code('''bad=run_agent(lambda c:'{"tool":"run_shell","command":"anything"}',records[0])
assert bad['final']['status']=='rejected'
missing={'case':{'question':'mass','current':False,'motion':False,'clock':False},'summaries':torch.full((12,),float('nan')),'shot_id':'missing'}
blocked=run_agent(lambda c:{'tool':'infer'},missing)
assert blocked['final']['status']=='unresolved'
print('Malformed call:',bad['final']); print('Missing-data call:',blocked['final'])
run=save_run(ROOT,'13',metrics,{'max_calls':2,'policy_model':'tiny causal Transformer','evaluation_cases':len(records)})
(run/'tool_traces.json').write_text(json.dumps(traces,indent=2))
(run/'evaluation.json').write_text(json.dumps(rows,indent=2))
print(run)'''),
md(r'''## 5. Optional: substitute an already-running local model
The numerical runtime stays unchanged. This optional cell uses a local OpenAI-compatible server on loopback, such as an existing LM Studio session. It is **off by default** and is not part of the saved verification. Set the model identifier to one already loaded in your server. Requests stay on this computer. Free-form text is parsed by the same strict schema; malformed responses are failures.

This is an adapter example, not a claim that a pretrained model has been evaluated here.'''),
code('''USE_LOCAL_LM=False
LOCAL_MODEL_ID=''  # Fill in the exact identifier of an already-loaded local model.
if USE_LOCAL_LM:
    if not LOCAL_MODEL_ID: raise ValueError('Set LOCAL_MODEL_ID to your loaded local model')
    import urllib.request
    from lm_testbed.language import TOOLS
    def local_policy(case):
        payload={'model':LOCAL_MODEL_ID,'temperature':0,'max_tokens':80,'messages':[
            {'role':'system','content':'Return only JSON with one key tool. Allowed tools: '+', '.join(TOOLS)+'. Choose needed evidence before inference. Growth requires image. Mass requires current and motion. Charge requires current. Unknown clock requires align.'},
            {'role':'user','content':json.dumps(case)}]}
        request=urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions',
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=60) as response: data=json.load(response)
        return data['choices'][0]['message']['content']
    display(run_agent(local_policy,records[0]))
else:
    print('Optional local server disabled; executed results above use the trained tiny LM.')''')],
['Expand evaluation beyond four toy cases: include wrong calibration, contradictory channels, malformed calls and unsupported questions.','Add a real second observation-acquisition turn while keeping the new measurement separate from evaluator truth.','Connect Factor’s provider abstraction to this runtime and compare models using identical observations, budgets and numerical tools.'])

write('14','closed_loop_experiment_world','a complete synthetic inference loop',
'''Bring the pieces together: configure a machine, acquire partial diagnostic evidence, infer uncertain parameters, choose an additional synthetic measurement, update the posterior, predict trajectories and give an LM access to the numerical result. Save the configuration, evidence and decisions as one run.

This is the first complete teaching slice of the eventual framework. It is not a full simulation of Z or a facility experiment optimizer.''',
'Bayesian experimental design, expected information gain, sequential inference, posterior predictive checks, integrated evidence records',[
md(r'''## 1. Freeze the experiment-design question
We infer charge and mass; resistance is fixed at .22 and gains/timing are assumed known. Begin with current summaries only. Candidate actions add one family of reconstructed summaries: speed, radius or thermal proxy. Costs are fictional teaching units. The objective is expected reduction in **joint charge–mass grid entropy per cost**.

The posterior is exact for this discrete grid and assumed likelihood. It is not exact inference for real physics. Acquisition uses the synthetic diagnostic-summary operator from lesson 08, not raw-instrument reconstructions. Closing the raw-to-inference path with calibrated reconstruction uncertainty is a future extension.'''),
code('''seed_all(1401)
true_theta=torch.tensor([[1.07,.83,.22,.6]])
# Only the acquisition function may see the synthetic world. Inference gets returned measurements.
with torch.no_grad(): synthetic_signal=summaries(true_theta)[0]
generator=torch.Generator().manual_seed(1402)
def acquire(indices,sigma=SUMMARY_SIGMA):
    idx=torch.tensor(indices); return synthetic_signal[idx]+torch.randn(len(indices),generator=generator)*sigma[idx]
observed=torch.full((12,),float('nan')); observed[:3]=acquire([0,1,2])
initial=grid_posterior(observed,channels=[0,1,2],n=41)
weights=initial['weights']; predictions=initial['predictions']; parameters=initial['theta']
print('Initial mean charge/mass:',initial['mean'][:2].tolist())
print('Initial 90% mass interval:',weighted_interval(parameters[:,1],weights).tolist())'''),
md(r'''## 2. Estimate the value of an additional diagnostic
Draw hypothetical truths from the current posterior, simulate possible measurements, and update the posterior for each. Average the entropy reduction. A large score means an instrument is expected to distinguish currently plausible parameter combinations, under the current model. It does not rank real hardware feasibility or safety.'''),
code('''candidates={'speed':{'indices':[3,4,5],'cost':1.0},
            'radius':{'indices':[6,7,8],'cost':1.2},
            'thermal_proxy':{'indices':[9,10,11],'cost':.8}}
def expected_gain(prior,indices,seed,draws=48):
    gen=torch.Generator().manual_seed(seed); idx=torch.tensor(indices)
    choice=torch.multinomial(prior,draws,replacement=True,generator=gen)
    potential=predictions[choice][:,idx]+torch.randn(draws,len(idx),generator=gen)*SUMMARY_SIGMA[idx]
    residual=(potential[:,None,:]-predictions[None,:,idx])/SUMMARY_SIGMA[idx]
    posterior=torch.softmax(torch.log(prior.clamp_min(1e-30))[None]-.5*residual.square().sum(-1),dim=1)
    entropies=-(posterior*torch.log(posterior.clamp_min(1e-30))).sum(1)
    gains=entropy(prior)-entropies
    return float(gains.mean()),float(gains.std()/np.sqrt(draws))
design=[]
for j,(name,candidate) in enumerate(candidates.items()):
    gain,se=expected_gain(weights,candidate['indices'],1410+j)
    design.append({'diagnostic':name,'expected_gain_nats':gain,'MC_standard_error':se,'cost':candidate['cost'],'gain_per_cost':gain/candidate['cost']})
chosen=max(design,key=lambda row:row['gain_per_cost'])['diagnostic']
display(design); print('Selected synthetic acquisition:',chosen)
plt.bar([r['diagnostic'] for r in design],[r['gain_per_cost'] for r in design]); plt.ylabel('expected information / teaching cost'); plt.show()'''),
md(r'''## 3. Acquire, update, and retain the actual outcome
Expected information gain is an average over possible outcomes. One realized observation can occasionally increase uncertainty. Preserve that result instead of forcing the observed gain to be positive.'''),
code('''indices=candidates[chosen]['indices']; measurement=acquire(indices); observed[indices]=measurement
log_likelihood=-.5*(((predictions[:,indices]-measurement)/SUMMARY_SIGMA[indices])**2).sum(1)
updated=torch.softmax(torch.log(weights.clamp_min(1e-30))+log_likelihood,0)
mean_after=(updated[:,None]*parameters).sum(0)
print('Actual entropy change:',entropy(weights)-entropy(updated))
print('Updated mass interval:',weighted_interval(parameters[:,1],updated).tolist())
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,w,title in zip(axes,[weights,updated],['Current only','After '+chosen]):
    ax.imshow(w.reshape(41,41).T,origin='lower',extent=[.75,1.25,.65,1.35],aspect='auto')
    ax.set(title=title,xlabel='charge',ylabel='mass')
plt.tight_layout(); plt.show()'''),
md(r'''## 4. Push parameter uncertainty through the world model
For this small problem, the numerical simulator is fast enough to propagate every grid point. The learned transition model from lesson 10 could replace it after its rollout error is incorporated. Sampling parameter uncertainty alone does not include uncertainty in equations, diagnostics, or surrogate approximation.'''),
code('''with torch.no_grad(): tt,paths=simulate(parameters)
mean_path=(paths*updated[:,None,None]).sum(0)
# Weighted pointwise quantiles, not a simultaneous trajectory band.
bands=torch.stack([weighted_interval(paths[:,j,2],updated) for j in range(paths.shape[1])])
plt.plot(tt,mean_path[:,2],label='posterior mean radius'); plt.fill_between(tt,bands[:,0],bands[:,1],alpha=.2,label='pointwise 90% conditional band')
plt.xlabel('time'); plt.ylabel('radius'); plt.legend(); plt.show()'''),
md(r'''## 5. Check predictions against a channel held back from fitting
Acquire the unselected candidate families after freezing the selection. These measurements are checks, not inputs to the posterior shown above. A posterior predictive check uses posterior predictive variance plus measurement variance. A large residual suggests some combination of wrong model, calibration, or reconstruction assumptions.'''),
code('''check_indices=[i for name,c in candidates.items() if name!=chosen for i in c['indices']]
check_values=acquire(check_indices)
pmean=(updated[:,None]*predictions).sum(0)
pvar=(updated[:,None]*(predictions-pmean).square()).sum(0)
check_z=(check_values-pmean[check_indices])/torch.sqrt(pvar[check_indices]+SUMMARY_SIGMA[check_indices]**2)
print('Held-back standardized residuals:',check_z.tolist())
print('Predictive check flagged:',bool((check_z.abs()>4).any()))
# Negative control: changed force coupling, same assumed inference operator.
with torch.no_grad(): wrong_signal=summaries(true_theta,replace(MachineConfig(),coupling=.2))[0]
wrong_z=(wrong_signal[check_indices]-pmean[check_indices])/torch.sqrt(pvar[check_indices]+SUMMARY_SIGMA[check_indices]**2)
print('Misspecified-world check residuals:',wrong_z.tolist())'''),
md(r'''## 6. Evaluate uncertainty across independent synthetic shots
One attractive posterior plot proves little. Freeze the selected measurement family and test current-only versus augmented inference on 24 independent shots. This is a conditional evaluation for the chosen design; it does not evaluate the full adaptive selection policy across all worlds. Counts are small and public.'''),
code('''evaluation_theta=draw_parameters(24,1450); evaluation_theta[:,2]=.22
with torch.no_grad(): evaluation_signal=summaries(evaluation_theta)
evaluation_noise=torch.randn(evaluation_signal.shape,generator=torch.Generator().manual_seed(1451))*SUMMARY_SIGMA
evaluation_obs=evaluation_signal+evaluation_noise
study={}
for name,idx in [('current',[0,1,2]),('augmented',[0,1,2]+indices)]:
    covered=[]; widths=[]; errors=[]
    for obs_row,truth_row in zip(evaluation_obs,evaluation_theta):
        ll=-.5*(((predictions[:,idx]-obs_row[idx])/SUMMARY_SIGMA[idx])**2).sum(1)
        w=torch.softmax(ll,0); interval=weighted_interval(parameters[:,1],w)
        covered.append(float(interval[0]<=truth_row[1]<=interval[1])); widths.append(float(interval[1]-interval[0]))
        errors.append(float(((w*parameters[:,1]).sum()-truth_row[1]).abs()))
    study[name]={'mass_coverage90':float(np.mean(covered)),'mean_mass_interval_width':float(np.mean(widths)),'mass_MAE':float(np.mean(errors)),'shots':24}
display(study)'''),
md(r'''## 7. Hand a complete summary record to the LM runtime
The LM does not receive simulator parameters or trajectories. It chooses a tool from metadata; the numerical tool calculates an estimate and records its assumptions. To satisfy the runtime's current “motion available” contract, acquire both speed and radius summaries here. This is a **separate final request with more information** than the design comparison above.'''),
code('''from lm_testbed.language import train_lm, constrained_call
from lm_testbed.runtime import run_agent
from lm_testbed.language import expected_tool
final_observed=observed.clone()
for family in ([3,4,5],[6,7,8]):
    missing=[i for i in family if not torch.isfinite(final_observed[i])]
    if missing: final_observed[missing]=acquire(missing)
final_observed[9:]=float('nan')
public_record={'case':{'question':'mass','current':True,'motion':True,'clock':True},'summaries':final_observed,'shot_id':'capstone-public'}
seed_all(12); lm,_=train_lm()
agent_result=run_agent(lambda case:constrained_call(lm,case),public_record,max_calls=2)
rule_result=run_agent(lambda case:{'tool':expected_tool(case)},public_record,max_calls=2)
print('LM episode:'); display(agent_result)
print('Conventional episode:'); display(rule_result)'''),
md(r'''## 8. Save the experiment as a reproducible application record
The record contains assumed machine configuration, acquisition design, posterior summaries, checks, evaluation and agent trace. This is the contract Factor can consume through an adapter. Existing Factor APIs are left unchanged; `lm_testbed.runtime` is a standalone teaching interface.'''),
code('''metrics={'selected_diagnostic':chosen,'design_scores':design,'actual_entropy_reduction':entropy(weights)-entropy(updated),
         'posterior_mean':mean_after.tolist(),'predictive_check_z':check_z.tolist(),'mismatch_check_z':wrong_z.tolist(),
         'study':study,'LM_result':agent_result,'rule_result':rule_result}
config={'machine':asdict(MachineConfig()),'diagnostics':asdict(DiagnosticConfig()),
        'inference':{'parameters':['charge','mass'],'resistance_fixed':.22,'grid':41,'likelihood':'Gaussian reconstructed summaries'},
        'design':{'objective':'joint entropy reduction per synthetic cost','candidates':candidates,'MC_draws':48}}
run=save_run(ROOT,'14',metrics,config)
public_json={**public_record,'summaries':[float(v) if torch.isfinite(v) else None for v in final_observed]}
(run/'public_observations.json').write_text(json.dumps(public_json,indent=2,allow_nan=False))
(run/'experiment_config.json').write_text(json.dumps(config,indent=2))
print('Complete experiment record:',run)'''),
md(r'''## What comes next in the framework
The layers are now explicit: configuration → evolving state → diagnostic response → reconstruction → posterior → decision → evidence. Each can be replaced independently.

Next substantial increments are a calibrated raw-signal likelihood, a circuit with time-dependent inputs, spatial material dynamics, uncertainty over model families, a learned observation/state-space model, natural-language tool-use training, and a Factor provider adapter. Higher-fidelity Z work would need physical units, validated constitutive laws, facility documentation, diagnostic calibration and independent real-data validation. This course establishes the software and learning patterns, not that validation.''')],
['Re-run design under a broad current-gain prior. Does another diagnostic become more valuable?','Evaluate the entire adaptive selection policy across new shots, including acquisition cost and failed predictive checks.','Connect the learned rollout surrogate while adding measured surrogate error to predictive uncertainty.'])

if __name__=='__main__':
    print('Created',len(list((ROOT/'notebooks').glob('*.ipynb'))),'notebooks')
