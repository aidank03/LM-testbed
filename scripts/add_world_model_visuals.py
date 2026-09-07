"""Insert tagged visual lessons without replacing existing notebook cells or outputs."""
from pathlib import Path
import textwrap
import nbformat as nb
PACK='world-visuals-v1'
ADDITIONS={}

def add(lesson,before,title,explanation,source):
    cells=[nb.v4.new_markdown_cell('### Visual lab — '+title+'\n\n'+textwrap.dedent(explanation).strip()),
           nb.v4.new_code_cell(textwrap.dedent(source).strip())]
    for cell in cells: cell.metadata['world_visual_pack']=PACK
    ADDITIONS.setdefault(lesson,[]).append((before,cells))

add('05','## 3. Make invalid','Who owns each setting?',
'''Follow one question across the four layers. **Blue** settings change the synthetic machine; **orange** settings change what is recorded; **purple** settings change the analysis; **green** settings change how tools are used. A PDV window belongs to processing even when a software configuration stores it alongside acquisition fields.

**Predict:** if you change gain, should the hidden physical trajectory move?''',
r'''
v_fig,v_ax=plt.subplots(figsize=(12,5),layout='constrained'); v_ax.axis('off')
v_columns=[('MACHINE','#dceaf7',['Charge / capacitance','Mass / resistance','Inductance / coupling','Hidden state trajectory']),
           ('ACQUISITION','#fde7ce',['Gain / clock offset','Noise / missing channels','Image exposure / PSF','Recorded signals']),
           ('INFERENCE','#eae0f5',['Reconstruction window','Likelihood / covariance','Priors / parameter set','Conditional posterior']),
           ('LM + TOOLS','#ddefdf',['Model / vocabulary','Permitted tool schema','Call budget / evidence','Decision or data request'])]
for v_j,(v_title,v_color,v_items) in enumerate(v_columns):
    v_x=v_j*3
    v_ax.text(v_x,4.25,v_title,ha='center',weight='bold',fontsize=11)
    for v_i,v_item in enumerate(v_items):
        v_ax.text(v_x,3.45-v_i*.88,v_item,ha='center',va='center',fontsize=10,
                  bbox=dict(boxstyle='round,pad=.6',fc=v_color,ec='white'))
    if v_j<3: v_ax.annotate('',xy=(v_x+1.95,.15),xytext=(v_x+1.15,.15),arrowprops=dict(arrowstyle='->',color='#40566b',lw=2))
v_ax.set(xlim=(-1.6,10.65),ylim=(-.35,4.8)); plt.show()
''')
add('05','## 6. Record','What does a batch of shots look like?',
'''Each row is an independent shot. Each column is a time sample. Color shows how a state changes through the batch. Every panel has its own color scale because current, radius and perturbation have different ranges.

**Notice:** frames from one row are related. Randomly distributing those frames across training and evaluation leaks information about the same shot.''',
r'''
with torch.no_grad():
    v_parameters=draw_parameters(8,50501); v_time,v_batch=simulate(v_parameters)
v_fig,v_axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for v_ax,v_col in zip(v_axes,[1,2,5]):
    v_im=v_ax.imshow(v_batch[:,:,v_col].numpy(),aspect='auto',origin='lower',extent=[0,float(v_time[-1]),-.5,7.5],cmap='viridis')
    v_ax.set(xlabel='time within shot',ylabel='independent shot',title=STATE_NAMES[v_col].replace('_',' '))
    v_fig.colorbar(v_im,ax=v_ax,shrink=.85,label='dimensionless value')
plt.show()
''')

add('06','## 2. Integrate','The circuit pushes the liner—and the liner changes the circuit',
'''The top path drives motion. The return arrow is feedback: changing radius changes inductance, which changes the current equation. The heat and perturbation variables are passive outputs of this particular model.

**Notice:** removing the motional-inductance term breaks the energy accounting, even if a trajectory still looks smooth.''',
r'''
v_fig,v_ax=plt.subplots(figsize=(12,4),layout='constrained'); v_ax.axis('off')
for v_x,v_y,v_label,v_color in [(0,1,'Capacitor V\nenergy store','#dceaf7'),(3,1,'Current I\nthrough L(r) and R','#dceaf7'),(6,1,'Magnetic force\nk I² / (2r)','#fde7ce'),(9,1,'Radius r / speed u\nmoving load','#ddefdf'),(3,-1,'Joule heating\nthermal proxy T','#fde7ce'),(6,-1,'Generic growth\nperturbation a','#eae0f5')]:
    v_ax.text(v_x,v_y,v_label,ha='center',va='center',fontsize=11,bbox=dict(boxstyle='round,pad=.7',fc=v_color,ec='#7c8b98'))
for v_x in [0,3,6]: v_ax.annotate('',xy=(v_x+1.85,1),xytext=(v_x+1,1),arrowprops=dict(arrowstyle='->',lw=1.8))
v_ax.annotate('',xy=(3,.25),xytext=(3,-.3),arrowprops=dict(arrowstyle='<-',lw=1.5))
v_ax.annotate('',xy=(5.1,-.6),xytext=(3.9,.45),arrowprops=dict(arrowstyle='->',lw=1.5))
v_ax.annotate('geometry → inductance feedback',xy=(3,1.8),xytext=(7.6,2.4),ha='center',fontsize=10,
              arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=15',lw=1.7,color='#48698a'))
v_ax.set(xlim=(-1.6,10.7),ylim=(-1.9,3)); plt.show()
''')
add('06','## 3. Differentiate','Play a synthetic shot',
'''Use the play/pause and frame controls under the animation. The same clock advances the current, radius and end-on liner sketch. The sketch uses the actual radius; the small radius change is easier to see in the middle panel. It is schematic geometry, not a mesh-based liner calculation.

**Predict:** does the minimum capacitor voltage coincide with the maximum inward speed? The existing state plots can help you check.''',
r'''
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from IPython.display import HTML
v_t=t.detach().numpy(); v_shot=state[1].detach().numpy()
v_fig,v_axes=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
v_axes[0].plot(v_t,v_shot[:,1],color='#2873a6'); v_dot_i,=v_axes[0].plot([],[],'o',color='#d66b24',ms=8)
v_axes[0].set(xlabel='time',ylabel='current',title='Circuit response')
v_axes[1].plot(v_t,v_shot[:,2],color='#2873a6'); v_dot_r,=v_axes[1].plot([],[],'o',color='#d66b24',ms=8)
v_axes[1].set(xlabel='time',ylabel='radius',title='Small inward displacement')
v_ring=Circle((0,0),1,fill=False,lw=5,color='#2873a6'); v_axes[2].add_patch(v_ring)
v_axes[2].add_patch(Circle((0,0),1,fill=False,lw=1,ls='--',color='gray'))
v_axes[2].set(xlim=(-1.2,1.2),ylim=(-1.2,1.2),aspect='equal',title='End-on radius sketch'); v_axes[2].axis('off')
v_clock=v_axes[2].text(0,-1.12,'',ha='center',fontsize=10)
def v_update_shot(v_frame):
    v_dot_i.set_data([v_t[v_frame]],[v_shot[v_frame,1]])
    v_dot_r.set_data([v_t[v_frame]],[v_shot[v_frame,2]])
    v_ring.set_radius(v_shot[v_frame,2]); v_clock.set_text(f't = {v_t[v_frame]:.2f}   r = {v_shot[v_frame,2]:.4f}')
    return v_dot_i,v_dot_r,v_ring,v_clock
v_update_shot(len(v_t)-1); display(v_fig)  # Static fallback for viewers without JavaScript.
v_movie=FuncAnimation(v_fig,v_update_shot,frames=np.linspace(0,len(v_t)-1,24,dtype=int),interval=140,blit=False)
v_html=v_movie.to_jshtml(default_mode='once'); plt.close(v_fig); display(HTML(v_html))
''')
add('06','## 5. Check time-step','Read a gradient as a local response',
'''These bars answer a specific question: how much would final radius change for a **1% increase** in one parameter, holding the others fixed? They are local derivatives multiplied by the parameter perturbation, not large finite interventions.

**Notice:** the growth parameter has no effect on mean radius because the perturbation does not feed back in this teaching world.''',
r'''
v_effect=.01*theta0.detach().numpy()[0]*grad.detach().numpy()[0]
v_fig,v_ax=plt.subplots(figsize=(9,3.4),layout='constrained')
v_ax.barh(PARAM_NAMES,v_effect,color=['#2873a6' if v_e>=0 else '#d66b24' for v_e in v_effect])
v_ax.axvline(0,color='black',lw=1); v_ax.set(xlabel='linearized change in final radius for +1% parameter',title='Direction and magnitude of local sensitivity')
plt.show()
''')

add('07','## 5. Radiation','Blur can hide structure that is still present',
'''All three columns observe the **same hidden shot**. Only the image point-spread width changes, with the same noise seed. The lower panels show the edge extracted by the existing threshold baseline.

**Notice:** a weak reconstructed ripple can mean poor visibility, not a stable material surface. Pixel quantization is visible in the stepped edge estimate.''',
r'''
v_fig,v_axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
for v_j,v_blur in enumerate([.25,3.,6.]):
    v_record=observe(t.numpy(),s[0].numpy(),replace(cfg,image_psf_pixels=v_blur),seed=70707)['diagnostics']['image']
    v_radius,v_amplitude,v_edge=edge_summary(v_record)
    v_axes[0,v_j].imshow(v_record['intensity'][-1],extent=[-1.2,1.2,1,0],aspect='auto',cmap='gray',vmin=.25,vmax=1.05)
    v_axes[0,v_j].set(xlim=(.88,1.07),title=f'PSF width = {v_blur:g} pixels',xlabel='x, zoomed positive edge',ylabel='z')
    v_axes[1,v_j].plot(v_record['z'],v_edge,label='threshold edge')
    v_frame_t=v_record['time'][-1]; v_true_r=np.interp(v_frame_t,t,s[0,:,2]); v_true_a=np.interp(v_frame_t,t,s[0,:,5])
    v_axes[1,v_j].plot(v_record['z'],v_true_r+v_true_a*np.cos(8*np.pi*v_record['z']),'k--',label='instantaneous hidden edge')
    v_axes[1,v_j].set(xlabel='z',ylabel='radius',ylim=(.95,1.01),title=f'Extracted mode amplitude = {v_amplitude:.4f}')
v_axes[1,0].legend(fontsize=8); plt.show()
''')
add('07','## Your next experiments','Distinguish a sample from an exposure',
'''Dots indicate waveform samples, while orange bands indicate finite camera exposures. A frame summarizes an interval rather than an instant. The second row uses the shifted-clock convention from this lesson: the state is sampled at recorded time plus the clock offset.

**Predict:** which fast features could disappear when an exposure is widened?''',
r'''
v_fig,v_ax=plt.subplots(figsize=(11,3.5),layout='constrained')
v_frame_times=raw['image']['time']; v_exposure=cfg.image_exposure
for v_row,v_shift,v_label in [(2,0,'Image exposure: aligned'),(1,.08,'Image exposure: clock +0.08')]:
    for v_ft in v_frame_times:
        v_ax.broken_barh([(v_ft+v_shift-v_exposure/2,v_exposure)],(v_row-.18,.36),facecolors='#d89042')
        v_ax.plot(v_ft+v_shift,v_row,'|',color='black',ms=12)
v_ax.plot(raw['bdot']['time'][::4],np.zeros_like(raw['bdot']['time'][::4]),'.',color='#2873a6',ms=5)
v_ax.set(yticks=[0,1,2],yticklabels=['Subset of B-dot samples','Image exposure: clock +0.08','Image exposure: aligned'],
         xlabel='time of sampled physical state',xlim=(0,float(t[-1])+.12),ylim=(-.55,2.55),title='A sampling interval, an exposure, and a clock shift are different things')
plt.show()
''')

add('08','## 2. Inspect','Watch a mass distribution become informative',
'''The shaded areas are marginal probability masses on the discrete mass grid, obtained by summing over charge. The dashed line is evaluator truth. The bands show each model's central 90% interval.

**Notice:** current alone constrains charge much more strongly than mass. A narrow posterior requires measurements that distinguish the remaining alternatives.''',
r'''
v_fig,v_axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
for v_ax,v_p,v_title in zip(v_axes,[current_only,joint],['Current only','Joint diagnostic summaries']):
    v_m=v_p['m'].numpy(); v_marginal=v_p['weights'].reshape(len(v_p['q']),len(v_m)).sum(0).numpy()
    v_band=weighted_interval(v_p['theta'][:,1],v_p['weights']).numpy()
    v_ax.fill_between(v_m,v_marginal,alpha=.25,color='#2873a6'); v_ax.plot(v_m,v_marginal,color='#2873a6')
    v_ax.axvspan(*v_band,color='#d89042',alpha=.15,label='central 90% interval')
    v_ax.axvline(float(truth[0,1]),color='black',ls='--',label='synthetic truth')
    v_ax.set(title=v_title,xlabel='mass',ylabel='marginal probability per grid point'); v_ax.legend(fontsize=8)
plt.show()
''')
add('08','## 3. Fit','Two hidden worlds can produce the same evidence',
'''Only the growth parameter differs between these two worlds. Their current, speed and radius agree exactly under this model, but their perturbations differ.

**Ask:** could a larger LM identify growth from those identical mean-state measurements? The missing information must come from another observable or a justified constraint.''',
r'''
v_twins=truth.repeat(2,1); v_twins[:,3]=torch.tensor([.3,.9])
with torch.no_grad(): v_tt,v_ss=simulate(v_twins)
v_fig,v_axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
for v_j,v_g in enumerate([.3,.9]):
    for v_ax,v_col in zip(v_axes,[1,2,5]): v_ax.plot(v_tt,v_ss[v_j,:,v_col],ls='-' if v_j==0 else '--',label=f'growth = {v_g}')
for v_ax,v_title in zip(v_axes,['Current: identical','Radius: identical','Perturbation: different']):
    v_ax.set(title=v_title,xlabel='time'); v_ax.legend(fontsize=9)
plt.show()
''')
add('08','## 4. A timing','When calibration error looks like physics',
'''Compare the true parameter values with both optimization results. In the fixed-gain fit, charge and mass can move to compensate for the wrong gain. Allowing gain to vary with a prior reduces that confounding for this example.

**Notice:** a successful optimizer can return biased machine parameters if its instrument model is wrong.''',
r'''
v_truths=[float(truth[0,0]),float(truth[0,1]),GAIN_TRUE]
v_fig,v_axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for v_j,(v_ax,v_name) in enumerate(zip(v_axes,['charge','mass','gain'])):
    v_ax.bar(['fixed gain','gain fitted'],[fixed[v_name],free[v_name]],color=['#d89042','#2873a6'])
    v_ax.axhline(v_truths[v_j],color='black',ls='--',label='synthetic truth')
    v_ax.set(title=v_name,ylabel='dimensionless value',ylim=(0,max(v_truths[v_j],fixed[v_name],free[v_name])*1.2)); v_ax.legend(fontsize=8)
plt.show()
''')

add('09','## 4. Check','How calibration chooses an interval multiplier',
'''The calibration scores are absolute errors divided by the network's predicted scale. The vertical line marks the chosen multiplier; the horizontal line marks 90% of calibration examples. Each parameter gets its own multiplier.

**Notice:** calibration uses separate shots. Adjusting the multiplier until evaluation coverage looks good would use the evaluation set for fitting.''',
r'''
v_fig,v_axes=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
for v_j,v_ax in enumerate(v_axes):
    v_scores=np.sort(score[:,v_j].numpy()); v_fraction=np.arange(1,len(v_scores)+1)/len(v_scores)
    v_ax.step(v_scores,v_fraction,where='post',color='#2873a6')
    v_ax.axhline(.9,color='gray',ls='--'); v_ax.axvline(float(multiplier[v_j]),color='#d66b24',ls='--')
    v_ax.set(title=PARAM_NAMES[v_j],xlabel='absolute error / predicted scale',ylabel='fraction of calibration shots',ylim=(0,1.02))
plt.show()
''')
add('09','## Your next experiments','See individual uncertainty failures',
'''Each horizontal line is a calibrated mass interval for one evaluation shot, plotted as an error relative to that shot's true mass. **Blue** intervals cross zero; **orange** intervals miss. All 128 shots contribute to the reported coverage; the figure shows the first 24 without selecting favorable cases.

**Notice:** narrow intervals under changed physics can be confidently wrong. Width alone is not a measure of usefulness.''',
r'''
v_fig,v_axes=plt.subplots(1,2,figsize=(11,6),sharex=True,sharey=True,layout='constrained')
for v_ax,v_mu,v_width,v_truth,v_title in zip(v_axes,[mu,mus],[width,ws],[ye,yshift],['Matched simulator','Changed coupling + thermal loss']):
    v_errors=(v_mu[:24,1]-v_truth[:24,1]).numpy(); v_half=v_width[:24,1].numpy()
    for v_i,(v_error,v_h) in enumerate(zip(v_errors,v_half)):
        v_color='#2873a6' if abs(v_error)<=v_h else '#d66b24'
        v_ax.plot([v_error-v_h,v_error+v_h],[v_i,v_i],color=v_color,lw=2)
        v_ax.plot(v_error,v_i,'o',color=v_color,ms=3)
    v_ax.axvline(0,color='black',ls='--'); v_ax.set(title=v_title,xlabel='estimated mass minus true mass',ylabel='evaluation shot',ylim=(24,-1))
plt.show()
''')

add('10','## 3. Compare','Teacher forcing versus free rollout',
'''During one-step training, every prediction starts from a true simulated state. During rollout, the next input is the model's own prediction. Errors can therefore feed back and accumulate.

**Predict:** can low one-step loss coexist with a poor long rollout? Compare this diagram with the horizon-error plot below.''',
r'''
v_fig,v_ax=plt.subplots(figsize=(12,5.4),layout='constrained'); v_ax.axis('off')
v_ax.text(-1.35,3.4,'Teacher forcing: each column starts from a supplied true state',weight='bold',fontsize=11)
for v_i,v_x in enumerate([0,4.5,9]):
    v_ax.text(v_x,2.65,f'true s{v_i}',ha='center',va='center',bbox=dict(boxstyle='round,pad=.55',fc='#dceaf7',ec='white'))
    v_ax.text(v_x,1.2,f'predicted s{v_i+1}',ha='center',va='center',bbox=dict(boxstyle='round,pad=.55',fc='#fde7ce',ec='white'))
    v_ax.annotate('',xy=(v_x,1.6),xytext=(v_x,2.25),arrowprops=dict(arrowstyle='->',lw=1.5))
    v_ax.text(v_x+.2,1.92,'model',fontsize=9)
v_ax.text(4.5,.5,'Predictions are scored against next-step truth; they are not fed into the next training example.',ha='center',fontsize=10,color='#40566b')
v_ax.text(-1.35,-.2,'Free rollout: each prediction becomes the next input',weight='bold',fontsize=11)
for v_i in range(4):
    v_label='true s0' if v_i==0 else f'predicted s{v_i}'
    v_ax.text(v_i*3,-.95,v_label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#dceaf7' if v_i==0 else '#fde7ce',ec='white'))
    if v_i<3:
        v_ax.annotate('',xy=(v_i*3+2.05,-.95),xytext=(v_i*3+.85,-.95),arrowprops=dict(arrowstyle='->',lw=1.5))
        v_ax.text(v_i*3+1.5,-.7,'model',ha='center',fontsize=9)
v_ax.set(xlim=(-1.65,10.45),ylim=(-1.6,3.9)); plt.show()
''')
add('10','## Your next experiments','Which shots drift as the forecast gets longer?',
'''Color is root-mean-square state error after dividing each state by its training-set scale. Every row is a separate held-out shot, and the two panels use the same color scale.

**Notice:** the mean horizon curve can hide a subset of difficult trajectories. The shifted panel changes the parameter distribution rather than supplying more training data.''',
r'''
v_matched_error=((pred-eval_s)/scale).square().mean(-1).sqrt().numpy()
v_shift_error=((shifted_pred-shift_s)/scale).square().mean(-1).sqrt().numpy()
v_max=max(v_matched_error.max(),v_shift_error.max())
v_fig,v_axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for v_ax,v_image,v_title in zip(v_axes,[v_matched_error,v_shift_error],['Matched shots','Shifted shots']):
    v_im=v_ax.imshow(v_image,origin='lower',aspect='auto',extent=[0,float(t[-1]),-.5,len(v_image)-.5],cmap='magma',vmin=0,vmax=v_max)
    v_ax.set(title=v_title,xlabel='forecast horizon',ylabel='independent shot')
v_fig.colorbar(v_im,ax=v_axes,label='normalized state RMSE',shrink=.9); plt.show()
''')

add('11','## 2. Build','Turn diagnostic values into tokens—and keep missingness visible',
'''The left panel shows standardized values for one shot, grouped by diagnostic and time. The middle panel represents missing tokens as grey cells; the right panel records the mask explicitly. A missing speed channel is not evidence of zero speed.

**Notice:** token identity tells the encoder which diagnostic produced the values. This is a summary-token example, not raw waveform or image encoding.''',
r'''
from matplotlib.colors import ListedColormap
v_token_values=et[0].numpy(); v_token_mask=np.array([False,True,False,False])
v_masked=np.ma.array(v_token_values,mask=np.repeat(v_token_mask[:,None],3,axis=1))
v_cmap=plt.get_cmap('coolwarm').copy(); v_cmap.set_bad('#b6bcc4')
v_bound=max(abs(v_token_values).max(),1)
v_fig,v_axes=plt.subplots(1,3,figsize=(11,3.8),gridspec_kw={'width_ratios':[3,3,1.7]},layout='constrained')
for v_ax,v_data,v_title in zip(v_axes[:2],[v_token_values,v_masked],['Standardized summaries','Speed token withheld']):
    v_im=v_ax.imshow(v_data,aspect='auto',cmap=v_cmap,vmin=-v_bound,vmax=v_bound)
    v_ax.set(xticks=[0,1,2],xticklabels=['early','middle','late'],yticks=range(4),yticklabels=['current','speed','radius','thermal'],title=v_title)
v_axes[2].imshow(v_token_mask[:,None],aspect='auto',cmap=ListedColormap(['#dceaf7','#b6bcc4']),vmin=0,vmax=1)
for v_i,v_missing in enumerate(v_token_mask): v_axes[2].text(0,v_i,'missing' if v_missing else 'present',ha='center',va='center',fontsize=10)
v_axes[2].set(xticks=[],yticks=[],title='Availability')
v_fig.colorbar(v_im,ax=v_axes[:2],shrink=.75,label='training-standardized value'); plt.show()
''')
add('11','## 3. Give','Inspect the attention weights actually used by the encoder',
'''These are attention probabilities from the trained first layer for one evaluation shot, averaged over heads. Rows are querying tokens; columns are tokens being read. In the right panel, a missing speed token cannot act as a key/value, so its column receives zero attention.

**Caution for interpretation:** weights show a model computation, not which diagnostic is physically causal. A missing token can still form a query internally, but its output is excluded by the model's masked pooling.''',
r'''
model.eval(); v_attention=[]
with torch.no_grad():
    for v_missing_speed in [False,True]:
        v_mask=torch.tensor([[False,v_missing_speed,False,False]])
        v_values=et[:1].masked_fill(v_mask[:,:,None],0)
        v_tokens=model.project(v_values)+model.modality(torch.arange(4))[None]
        _,v_weights=model.encoder.layers[0].self_attn(v_tokens,v_tokens,v_tokens,key_padding_mask=v_mask,need_weights=True,average_attn_weights=True)
        v_attention.append(v_weights[0].numpy())
v_fig,v_axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for v_ax,v_weights,v_title in zip(v_axes,v_attention,['All diagnostics available','Speed key/value masked']):
    v_im=v_ax.imshow(v_weights,vmin=0,vmax=1,cmap='Blues')
    for v_row in range(4):
        for v_col in range(4): v_ax.text(v_col,v_row,f'{v_weights[v_row,v_col]:.2f}',ha='center',va='center',color='white' if v_weights[v_row,v_col]>.55 else 'black')
    v_ax.set(xticks=range(4),xticklabels=['current','speed','radius','thermal'],yticks=range(4),yticklabels=['current','speed','radius','thermal'],title=v_title,xlabel='key/value being read',ylabel='query token')
v_fig.colorbar(v_im,ax=v_axes,label='attention probability',shrink=.8); plt.show()
''')
add('11','## Your next experiments','Which missing channel hurts which parameter?',
'''Each number is parameter MAE divided by that parameter's prior range. This lets charge, mass and resistance share a color scale. The left panel is the Transformer; the right is ridge regression on the same permitted inputs.

**Notice:** one overall score can conceal a parameter-specific weakness. Compare the mass column when speed is missing.''',
r'''
v_names=list(results); v_matrices=[np.array([results[v_name][v_method] for v_name in v_names])/span.numpy() for v_method in ['attention_MAE','ridge_MAE']]
v_max=max(v_matrix.max() for v_matrix in v_matrices)
v_fig,v_axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for v_ax,v_matrix,v_title in zip(v_axes,v_matrices,['Attention encoder','Ridge baseline']):
    v_im=v_ax.imshow(v_matrix,aspect='auto',vmin=0,vmax=v_max,cmap='YlOrRd')
    for v_i in range(len(v_names)):
        for v_j in range(3): v_ax.text(v_j,v_i,f'{v_matrix[v_i,v_j]:.2f}',ha='center',va='center',color='white' if v_matrix[v_i,v_j]>.7*v_max else 'black')
    v_ax.set(xticks=range(3),xticklabels=PARAM_NAMES[:3],yticks=range(len(v_names)),yticklabels=[v_name.replace('_',' ') for v_name in v_names],title=v_title)
v_fig.colorbar(v_im,ax=v_axes,label='MAE / prior range',shrink=.8); plt.show()
''')

add('12','## 2. Causal','The model predicts the next token, not the token it just read',
'''The upper strip is the input sequence; the lower strip is its one-position-shifted target. Pale targets are prompt positions ignored by the loss. Orange targets are response tokens that contribute to training, including the end token.

**Notice:** the prompt is still visible as context even though its reconstruction is not scored.''',
r'''
from matplotlib.colors import ListedColormap
v_input=x[0].numpy(); v_target=y[0].numpy(); v_length=len(v_input)
v_fig,v_axes=plt.subplots(2,1,figsize=(13,4.8),layout='constrained')
v_axes[0].imshow(np.zeros((1,v_length)),aspect='auto',cmap=ListedColormap(['#dceaf7']))
v_axes[1].imshow((v_target!=-100)[None,:],aspect='auto',cmap=ListedColormap(['#eef0f2','#f5c48b']),vmin=0,vmax=1)
for v_ax,v_tokens,v_title in zip(v_axes,[v_input,v_target],['Input at each position','Target for the next token; ignored prompt positions are grey']):
    v_labels=[VOCAB[int(v_token)] if v_token!=-100 else 'ignored' for v_token in v_tokens]
    v_ax.set(xticks=range(v_length),xticklabels=v_labels,yticks=[],title=v_title)
    v_ax.tick_params(axis='x',labelrotation=40,labelsize=9)
    for v_i in range(v_length): v_ax.text(v_i,0,str(v_i),ha='center',va='center',fontsize=9)
plt.show()
''')
add('12','## 3. The language','A causal mask prevents peeking ahead',
'''A blue square means a query at the row position may read the column position. The upper triangle is blocked because those tokens are in the future. This is the **permission mask**, not a learned attention-weight matrix.

**Notice:** causal masking and response-loss masking solve different problems. One limits available information; the other decides which errors are scored.''',
r'''
from matplotlib.colors import ListedColormap
v_n=x.shape[1]; v_allowed=np.tril(np.ones((v_n,v_n)))
v_fig,v_ax=plt.subplots(figsize=(7,6),layout='constrained')
v_ax.imshow(v_allowed,cmap=ListedColormap(['#eef0f2','#76add3']),vmin=0,vmax=1)
v_ax.set(xlabel='key position: token being read',ylabel='query position: token making prediction',title='Causal attention permission mask',xticks=np.arange(0,v_n,2),yticks=np.arange(0,v_n,2))
v_ax.text(v_n*.7,v_n*.2,'future\nblocked',ha='center',fontsize=12,color='#5f6874')
v_ax.text(v_n*.25,v_n*.7,'past + present\nallowed',ha='center',fontsize=12,color='#203e56')
plt.show()
''')
add('12','## 5. Save','Watch autoregressive generation one token at a time',
'''At each step the LM assigns probabilities to the next token, chooses the largest one, appends it, and repeats. The bars show the top five tokens at that step for the first held-out request. Playback is saved with the notebook; a static view of the tool-name step appears first.

**Notice:** these are next-token probabilities inside a tiny command language. They are not calibrated probabilities that a scientific decision is correct.''',
r'''
from lm_testbed.language import encode
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import textwrap as v_textwrap
v_case=evaluation[0]; v_prefix=encode(prompt(v_case)); v_ids=v_prefix.copy(); v_generation=[]
with torch.no_grad():
    for v_step in range(8):
        v_prob=model(torch.tensor([v_ids]))[0,-1].softmax(-1)
        v_probabilities,v_tokens=torch.topk(v_prob,5)
        v_next=int(v_prob.argmax())
        v_generation.append({'before':v_ids[len(v_prefix):].copy(),'tokens':v_tokens.tolist(),'probabilities':v_probabilities.tolist(),'chosen':v_next})
        v_ids.append(v_next)
        if v_next==TOKEN['<eos>']: break
v_fig,v_ax=plt.subplots(figsize=(10,4.8),layout='constrained')
def v_update_tokens(v_index):
    v_ax.clear(); v_frame=v_generation[v_index]
    v_names=[VOCAB[v_token] for v_token in v_frame['tokens']]
    v_colors=['#d66b24' if v_token==v_frame['chosen'] else '#2873a6' for v_token in v_frame['tokens']]
    v_ax.barh(np.arange(5),v_frame['probabilities'],color=v_colors)
    v_ax.set(yticks=np.arange(5),yticklabels=v_names,xlim=(0,1),ylim=(4.6,-.6),xlabel='next-token probability')
    v_so_far=' '.join(VOCAB[v_token] for v_token in v_frame['before']) or '(nothing yet)'
    v_ax.set_title(f'Step {v_index+1}: append {VOCAB[v_frame["chosen"]]}\nGenerated so far: {v_so_far}',fontsize=11)
    v_ax.text(.99,.03,'Orange = greedily selected token',transform=v_ax.transAxes,ha='right',fontsize=9)
    return []
v_update_tokens(min(3,len(v_generation)-1)); display(v_fig)
v_token_movie=FuncAnimation(v_fig,v_update_tokens,frames=len(v_generation),interval=900,blit=False)
v_token_html=v_token_movie.to_jshtml(default_mode='once'); plt.close(v_fig); display(HTML(v_token_html))
''')

add('13','## 2. Inspect','Keep the language model and numerical evidence separate',
'''The LM selects an action. The runtime checks the call and information requirements. Only then does a numerical tool calculate a conditional estimate. Unsupported requests or missing observations lead to a request, rejection or unresolved result.

**Notice:** evaluator truth is outside this path. A returned evidence identifier points back to observations; it does not certify the physical interpretation.''',
r'''
v_fig,v_ax=plt.subplots(figsize=(12,5),layout='constrained'); v_ax.axis('off')
v_nodes=[(0,1,'Permitted observations\n+ request metadata','#dceaf7'),(3,1,'LM proposes\na JSON tool call','#eae0f5'),(6,1,'Runtime validates\nschema + required data','#fde7ce'),(9,1,'Numerical tool\nconditional inference','#ddefdf'),(6,-1,'Reject / unresolved /\nrequest measurement','#f7dfd8'),(9,-1,'Estimate + assumptions\n+ evidence reference','#ddefdf')]
for v_x,v_y,v_text,v_color in v_nodes:
    v_ax.text(v_x,v_y,v_text,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.7',fc=v_color,ec='#8293a2'))
for v_x in [0,3,6]:v_ax.annotate('',xy=(v_x+1.9,1),xytext=(v_x+1.1,1),arrowprops=dict(arrowstyle='->',lw=1.6))
for v_x in [6,9]:v_ax.annotate('',xy=(v_x,-.4),xytext=(v_x,.4),arrowprops=dict(arrowstyle='->',lw=1.6))
v_ax.text(1.2,-1.15,'Hidden simulator truth is supplied only to the evaluator.',ha='center',fontsize=10,color='#5c6771',wrap=True)
v_ax.set(xlim=(-1.7,10.7),ylim=(-2,2)); plt.show()
''')
add('13','## 4. Test','Separate choosing the right tool from producing an estimate',
'''The matrix marks correct action choices for each policy and held-out case. The right panel counts conditional estimates returned by the runtime. An estimate count alone is not an accuracy score: sometimes the correct action is to request missing information.

**Notice:** the rule baseline uses the authored policy. The small evaluation has only four cases; inspect the individual failures.''',
r'''
from matplotlib.colors import ListedColormap
v_policies=list(policies); v_shots=[v_record['shot_id'] for v_record in records]
v_correct=np.array([[next(v_row['tool_correct'] for v_row in rows if v_row['policy']==v_name and v_row['shot']==v_shot) for v_shot in v_shots] for v_name in v_policies])
v_fig,v_axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.6,1]},layout='constrained')
v_axes[0].imshow(v_correct,aspect='auto',cmap=ListedColormap(['#f0be9b','#b6d8be']),vmin=0,vmax=1)
for v_i,v_name in enumerate(v_policies):
    for v_j,v_shot in enumerate(v_shots):
        v_row=next(v_r for v_r in rows if v_r['policy']==v_name and v_r['shot']==v_shot)
        v_axes[0].text(v_j,v_i,v_row['tool'] or 'invalid',ha='center',va='center',fontsize=8)
v_axes[0].set(yticks=range(len(v_policies)),yticklabels=v_policies,xticks=range(len(v_shots)),
              xticklabels=[f'case {v_i+1}\nneed {expected_tool(v_record["case"])}' for v_i,v_record in enumerate(records)],title='Action correctness: green yes / orange no')
v_axes[0].tick_params(axis='x',labelsize=8)
v_estimates=[metrics[v_name]['estimated_cases'] for v_name in v_policies]
v_axes[1].barh(v_policies,v_estimates,color='#2873a6'); v_axes[1].set(xlim=(0,len(records)),xlabel='conditional estimates returned',title='Execution outcome ≠ action accuracy')
plt.show()
''')
add('13','## 5. Optional','Read one actual tool trace',
'''This panel follows a recorded episode from the evaluation above. It prefers an episode that returned an estimate, if one exists; the policy and shot are named explicitly. The original trace remains available as structured data.

**Ask:** which parts were proposed by the language policy, and which were computed or enforced by the runtime?''',
r'''
v_trace=next((v_item for v_item in traces if v_item['episode']['final']['status']=='conditional_estimate'),traces[0])
v_episode=v_trace['episode']; v_final=v_episode['final']; v_call=v_episode['trace'][0]['call'] if v_episode['trace'] else {'tool':'invalid'}
v_details=[f'Policy: {v_trace["policy"]}\nShot: {v_trace["shot"]}',f'Proposed tool\n{v_call["tool"]}',f'Runtime result\n{v_final["status"]}']
if 'interval90' in v_final: v_details.append(f'{v_final["parameter"]} = {v_final["mean"]:.3f}\n90% interval [{v_final["interval90"][0]:.3f}, {v_final["interval90"][1]:.3f}]')
else: v_details.append(v_final.get('reason','Additional information required'))
v_fig,v_ax=plt.subplots(figsize=(12,3.1),layout='constrained'); v_ax.axis('off')
for v_i,v_detail in enumerate(v_details):
    v_ax.text(v_i*3,0,v_detail,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.75',fc=['#dceaf7','#eae0f5','#fde7ce','#ddefdf'][v_i],ec='#8293a2'))
    if v_i<3:v_ax.annotate('',xy=(v_i*3+1.9,0),xytext=(v_i*3+1.1,0),arrowprops=dict(arrowstyle='->',lw=1.5))
v_ax.text(4.5,-.85,'Evidence: '+', '.join(v_final.get('evidence',[])),ha='center',fontsize=9)
v_ax.set(xlim=(-1.7,10.8),ylim=(-1.2,1)); plt.show()
''')

add('14','## 3. Acquire','Compare information, uncertainty in the estimate, and cost',
'''The left bars show expected information gain with one Monte Carlo standard error. The right bars divide the estimated gain by each candidate's fictional acquisition cost. Orange marks the selected candidate.

**Notice:** the selection is conditional on this posterior, noise model and utility. Monte Carlo error is only numerical sampling uncertainty; it does not include errors in the physical model.''',
r'''
v_labels=[v_r['diagnostic'].replace('_',' ') for v_r in design]
v_colors=['#d66b24' if v_r['diagnostic']==chosen else '#2873a6' for v_r in design]
v_fig,v_axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
v_axes[0].bar(v_labels,[v_r['expected_gain_nats'] for v_r in design],yerr=[v_r['MC_standard_error'] for v_r in design],capsize=5,color=v_colors)
v_axes[0].set(ylabel='expected information gain (nats)',title='Benefit under the current posterior')
v_axes[1].bar(v_labels,[v_r['gain_per_cost'] for v_r in design],color=v_colors)
v_axes[1].set(ylabel='expected information / teaching cost',title='Objective used to choose a diagnostic')
for v_ax in v_axes: v_ax.tick_params(axis='x',labelrotation=15)
plt.show()
''')
add('14','## 4. Push','Assimilate the newly acquired samples one at a time',
'''The orange family contains three summary measurements. These panels sequentially add each actual acquired sample to the current-only posterior. All use the same discrete grid and a shared color scale; colors show posterior mass relative to the largest value across the panels.

**Notice:** an individual sample need not reduce uncertainty by the same amount as the next one. The entire sequence is conditional on the specified likelihood.''',
r'''
v_stage_weights=[weights]; v_logw=torch.log(weights.clamp_min(1e-30)).clone()
for v_j,v_index in enumerate(indices):
    v_logw=v_logw-.5*((predictions[:,v_index]-measurement[v_j])/SUMMARY_SIGMA[v_index]).square()
    v_stage_weights.append(torch.softmax(v_logw,0))
v_top=max(float(v_w.max()) for v_w in v_stage_weights)
v_fig,v_axes=plt.subplots(1,4,figsize=(13,3.5),sharex=True,sharey=True,layout='constrained')
for v_j,(v_ax,v_w) in enumerate(zip(v_axes,v_stage_weights)):
    v_im=v_ax.imshow(v_w.reshape(41,41).T.numpy()/v_top,origin='lower',extent=[.75,1.25,.65,1.35],aspect='auto',cmap='viridis',vmin=0,vmax=1)
    v_ax.set(title='Current only' if v_j==0 else f'+ {v_j} {chosen} sample(s)',xlabel='charge')
    v_ax.text(.02,.98,f'H = {entropy(v_w):.2f}',transform=v_ax.transAxes,va='top',color='white',fontsize=9)
v_axes[0].set_ylabel('mass'); v_fig.colorbar(v_im,ax=v_axes,label='posterior mass / shared peak',shrink=.8); plt.show()
''')
add('14','## 6. Evaluate','A held-back diagnostic can expose a wrong world model',
'''Bars show standardized predictive residuals for the held-back channels. Blue uses the new noisy measurements from the matched simulator. Orange is the existing changed-coupling negative control, evaluated without added measurement noise. Both use the same posterior predictive denominator. These are diagnostic examples, not paired estimates of false-alarm rates.

**Notice:** a poor prediction flags inconsistency. It does not uniquely tell us whether the problem lies in the equations, calibration, or noise assumptions.''',
r'''
v_fig,v_ax=plt.subplots(figsize=(11,4.3),layout='constrained'); v_positions=np.arange(len(check_indices))
v_ax.bar(v_positions-.18,check_z.numpy(),width=.36,color='#2873a6',label='matched held-back measurement')
v_ax.bar(v_positions+.18,wrong_z.numpy(),width=.36,color='#d66b24',label='changed-coupling noiseless control')
v_ax.axhspan(-4,4,color='#ddefdf',alpha=.4,zorder=0,label='teaching ±4 threshold')
v_ax.axhline(0,color='black',lw=.8); v_ax.set(xticks=v_positions,xticklabels=[SUMMARY_NAMES[v_i] for v_i in check_indices],ylabel='predictive residual / predictive standard deviation',title='Predictive check on evidence not used in the posterior')
v_ax.tick_params(axis='x',labelrotation=25); v_ax.legend(fontsize=8); plt.show()
''')
add('14','## 8. Save','Follow the full inference cycle',
'''This map marks the stages completed in the capstone. The current-only posterior chooses a synthetic measurement; the new observation updates inference; held-back diagnostics check predictions. The final LM request uses a separately completed current-and-motion record, as described above.

**Notice:** a measurement request must produce new evidence before inference can use it. A language-model statement is not itself a new physical observation.''',
r'''
v_fig,v_ax=plt.subplots(figsize=(12,5.3),layout='constrained'); v_ax.axis('off')
v_nodes=[(0,2,'1  Configure world\nassumptions + priors','#dceaf7'),(4.3,2,'2  Acquire current\npartial evidence','#fde7ce'),(8.6,2,'3  Infer charge / mass\ncurrent-only posterior','#eae0f5'),
         (8.6,0,'4  Select + acquire\n'+chosen,'#fde7ce'),(4.3,0,'5  Update + predict\nposterior trajectories','#eae0f5'),(0,0,'6  Check / explain\nheld-back data + tool trace','#ddefdf')]
for v_x,v_y,v_label,v_color in v_nodes:v_ax.text(v_x,v_y,v_label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.8',fc=v_color,ec='#8293a2'))
for v_a,v_b in [((1.45,2),(2.85,2)),((5.75,2),(7.15,2)),((8.6,1.35),(8.6,.65)),((7.15,0),(5.75,0)),((2.85,0),(1.45,0))]:
    v_ax.annotate('',xy=v_b,xytext=v_a,arrowprops=dict(arrowstyle='->',lw=1.8))
v_ax.annotate('Next learning experiment: revise an assumption, then gather new evidence',xy=(0,2.7),xytext=(4.3,3.35),ha='center',fontsize=10,
              arrowprops=dict(arrowstyle='->',connectionstyle='angle,angleA=0,angleB=90,rad=12',lw=1.4))
v_ax.set(xlim=(-1.85,10.45),ylim=(-.9,3.8)); plt.show()
''')

def enhance(notebook,lesson):
    """Idempotent insertion with exact heading anchors; retain every untagged cell."""
    original=[c for c in notebook.cells if c.metadata.get('world_visual_pack')!=PACK]
    additions=ADDITIONS.get(lesson,[])
    for anchor,_ in additions:
        matches=[c for c in original if c.cell_type=='markdown' and c.source.startswith(anchor)]
        if len(matches)!=1: raise ValueError(f'Lesson {lesson}: expected one heading for {anchor!r}; found {len(matches)}')
    result=[]
    import copy
    for cell in original:
        for anchor,new_cells in additions:
            if cell.cell_type=='markdown' and cell.source.startswith(anchor): result.extend(copy.deepcopy(new_cells))
        result.append(cell)
    notebook.cells=result; notebook.metadata['lm_testbed']['visual_pack']=PACK
    return notebook

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    count=0
    for p in sorted((root/'notebooks').glob('*.ipynb')):
        lesson=p.name[:2]
        if lesson not in ADDITIONS: continue
        n=nb.read(p,as_version=4)
        if n.metadata.get('lm_testbed',{}).get('lesson')!=lesson: continue
        enhance(n,lesson); nb.validate(n)
        for c in n.cells:
            if c.cell_type=='code' and c.metadata.get('world_visual_pack')==PACK: compile(c.source,p.name,'exec')
        nb.write(n,p); count+=1
    print(f'Enhanced {count} notebooks with {sum(len(v) for v in ADDITIONS.values())} visual labs.')
