"""Dimensionless teaching world. No facility parameters or validated material model."""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, gaussian_filter1d

STATE_NAMES = ('capacitor_voltage', 'current', 'radius', 'inward_speed', 'thermal_proxy', 'perturbation')
PARAM_NAMES = ('charge', 'mass', 'resistance', 'growth')
LOW = np.array([.8, .7, .12, .3], dtype=np.float32)
HIGH = np.array([1.2, 1.3, .32, .9], dtype=np.float32)

@dataclass(frozen=True)
class MachineConfig:
    charge: float = 1.0
    mass: float = 1.0
    resistance: float = .22
    growth: float = .6
    capacitance: float = 1.0
    inductance: float = .7
    coupling: float = .12
    damping: float = .08
    thermal_loss: float = .15
    end_time: float = 1.4
    samples: int = 96

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if type(self.samples) is not int or self.samples < 16:
            raise ValueError('samples must be an integer >= 16')
        if self.end_time > 1.6:
            raise ValueError('teaching solver is restricted to the pre-stagnation interval <= 1.6')

def fingerprint(config):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()

def rhs(y, theta, cfg):
    """Batched differentiable circuit, motion, heat proxy and generic growth ODE."""
    v, current, radius, speed, heat, amp = y.unbind(-1)
    _, mass, resistance, growth = theta.unbind(-1)
    r = radius.clamp_min(.2)  # guard only; simulate() rejects trajectories reaching it
    inductance = cfg.inductance + cfg.coupling * torch.log(1 / r)
    ldot = cfg.coupling * speed / r
    # F = (I^2 / 2) dL/dx. The same coupling appears in the circuit and mechanics.
    accel = (.5 * cfg.coupling * current.square() / r - cfg.damping * speed) / mass
    return torch.stack((-current / cfg.capacitance,
        (v - resistance * current - current * ldot) / inductance,
        -speed, accel,
        resistance * current.square() - cfg.thermal_loss * (heat - .1),
        (growth * current.square() - .2) * amp), -1)

def simulate(theta=None, cfg=None):
    """Return time [T] and state [B,T,6]. theta is [B,4]; gradients are retained."""
    cfg = cfg or MachineConfig()
    if theta is None:
        theta = torch.tensor([[cfg.charge, cfg.mass, cfg.resistance, cfg.growth]], dtype=torch.float32)
    if theta.ndim != 2 or theta.shape[1] != 4 or not torch.isfinite(theta).all() or (theta <= 0).any():
        raise ValueError('theta must have shape [batch, 4] with finite positive entries')
    t = torch.linspace(0, cfg.end_time, cfg.samples, device=theta.device, dtype=theta.dtype)
    zero = torch.zeros_like(theta[:, 0]); one = zero + 1
    y = torch.stack((theta[:, 0], zero, one, zero, one * .1, one * .01), -1)
    path = [y]; dt = t[1] - t[0]
    for _ in range(cfg.samples - 1):
        k1 = rhs(y, theta, cfg); k2 = rhs(y + dt*k1/2, theta, cfg)
        k3 = rhs(y + dt*k2/2, theta, cfg); k4 = rhs(y + dt*k3, theta, cfg)
        y = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        path.append(y)
    result = torch.stack(path, 1)
    if not torch.isfinite(result).all() or (result[..., 2] <= .2).any():
        raise ValueError('trajectory outside supported pre-stagnation domain')
    return t, result

def draw_parameters(n, seed, shifted=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(LOW, HIGH, (n, 4)).astype('float32')
    if shifted:
        x[:, 0] = rng.uniform(1.3, 1.5, n)
        x[:, 2] = rng.uniform(.35, .45, n)
    return torch.from_numpy(x)

@dataclass(frozen=True)
class DiagnosticConfig:
    gain: float = 1.0
    clock_shift: float = 0.0
    blur_samples: float = .6
    noise: float = .015
    pdv_wavelength: float = .01  # dimensionless length, not a real laser wavelength
    pdv_window: int = 64
    image_psf_pixels: float = 1.0
    image_exposure: float = .04
    radiation_response: float = 1.0
    missing: tuple = ()

    def __post_init__(self):
        for key in ('gain','pdv_wavelength','radiation_response'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key) <= 0: raise ValueError(key)
        for key in ('noise','blur_samples','image_psf_pixels','image_exposure'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key) < 0: raise ValueError(key)
        if not np.isfinite(self.clock_shift): raise ValueError('clock_shift')
        if type(self.pdv_window) is not int or not 8 <= self.pdv_window <= 512: raise ValueError('pdv_window')
        if set(self.missing) - {'bdot','pdv','image','radiation','spectrum'}: raise ValueError('unknown diagnostic')

def observe(t, states, cfg=None, seed=0):
    """Raw-like diagnostic records; separate from simulator truth and reconstructions."""
    cfg = cfg or DiagnosticConfig(); rng = np.random.default_rng(seed)
    t = np.asarray(t); s = np.asarray(states)
    if s.shape != (len(t),6): raise ValueError('one shot [time,6] required')
    def sample(values, clock=t):
        a = np.interp(clock + cfg.clock_shift, t, values)
        return gaussian_filter1d(a, cfg.blur_samples) if cfg.blur_samples > 0 else a
    def noisy(x): return x + rng.normal(0,cfg.noise,np.shape(x))
    raw = {}
    if 'bdot' not in cfg.missing:
        raw['bdot'] = {'time':t.copy(), 'voltage':noisy(cfg.gain*sample(np.gradient(s[:,1], t)))}
    if 'pdv' not in cfg.missing:
        clock = np.linspace(t[0],t[-1],1024); speed = sample(s[:,3],clock)
        phase = 2*np.pi*np.cumsum(2*speed/cfg.pdv_wavelength)*(clock[1]-clock[0])
        raw['pdv'] = {'time':clock, 'voltage':noisy(cfg.gain*np.cos(phase+.3))}
    if 'radiation' not in cfg.missing:
        raw['radiation'] = {'time':t.copy(), 'voltage':noisy(cfg.radiation_response*sample(s[:,4]**2))}
    if 'spectrum' not in cfg.missing:
        energy = np.linspace(.1,3,80); heat = s[-1,4]
        expected = 1000*cfg.radiation_response*np.exp(-energy/(heat+.1))
        raw['spectrum'] = {'energy':energy,'counts':rng.poisson(expected)}
    if 'image' not in cfg.missing:
        # A blurred projected edge proxy, not radiation transport or an Abel projection.
        x = np.linspace(-1.2,1.2,96); z = np.linspace(0,1,64)
        frames=[]; frame_times=np.linspace(.3*t[-1],.95*t[-1],4)
        for ft in frame_times:
            gate=[]
            for gt in np.linspace(ft-cfg.image_exposure/2,ft+cfg.image_exposure/2,5):
                radius=np.interp(gt+cfg.clock_shift,t,s[:,2]); amp=np.interp(gt+cfg.clock_shift,t,s[:,5])
                edge=radius+amp*np.cos(2*np.pi*4*z)
                gate.append(1-.65/(1+np.exp(np.clip((np.abs(x[None,:])-edge[:,None])/.015,-60,60))))
            frames.append(noisy(gaussian_filter(np.mean(gate,axis=0),cfg.image_psf_pixels)))
        raw['image']={'time':frame_times,'x':x,'z':z,'intensity':np.stack(frames)}
    return {'schema_version':'z-teaching-v1','units':'dimensionless','diagnostics':raw,
            'acquisition':asdict(cfg),'seed':seed}

def summaries(theta, cfg=None):
    """Differentiable reconstructed-summary operator [B,12]; used in inverse lessons.
    This deliberately simplified likelihood does NOT consume observe() raw waveforms.
    """
    _, s = simulate(theta, cfg)
    idx = [s.shape[1]//3,2*s.shape[1]//3,s.shape[1]-1]
    return torch.cat([s[:,idx,j] for j in (1,3,2,4)],dim=1)

SUMMARY_NAMES = tuple(f'{name}_{i}' for name in ('current','speed','radius','thermal_proxy') for i in range(3))
SUMMARY_SIGMA = torch.tensor([.012]*3+[.002]*3+[.003]*3+[.008]*3)

def summary_dataset(n, seed, shifted=False, cfg=None):
    theta = draw_parameters(n, seed, shifted)
    with torch.no_grad(): clean = summaries(theta,cfg)
    g = torch.Generator().manual_seed(seed+800000)
    obs = clean + torch.randn(clean.shape,generator=g)*SUMMARY_SIGMA
    return obs, theta
