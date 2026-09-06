"""Small, inspectable PyTorch lessons with separate evaluation data."""
from __future__ import annotations

import math
import random
import time
from typing import Callable

import torch
from torch import nn


EventSink = Callable[[dict], None]


def choose_device(requested: str = "auto") -> str:
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if requested not in available:
        raise ValueError("device must be auto, cpu, mps, or cuda")
    if not available[requested]:
        raise ValueError(f"requested device is unavailable: {requested}")
    return requested


def _finite_number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def _generator(seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _emit(emit, kind, **data):
    emit({"kind": kind, "time": time.time(), **data})


def run_line_fit(config: dict, emit: EventSink) -> dict:
    """Teach tensors/autograd with y = wx+b and locked development/test sets."""
    seed = _integer(config.get("seed", 7), "seed", 0, 2**31 - 1)
    epochs = _integer(config.get("epochs", 80), "epochs", 5, 300)
    learning_rate = _finite_number(config.get("learning_rate", 0.08), "learning_rate", 1e-4, 1.0)
    noise = _finite_number(config.get("noise", 0.15), "noise", 0.0, 1.0)
    delay = _finite_number(config.get("delay", 0.025), "delay", 0.0, 0.2)
    device = choose_device(config.get("device", "auto"))
    torch.manual_seed(seed)
    truth_weight, truth_bias = 2.5, -0.4

    def make_split(n, split_seed):
        generator = _generator(split_seed)
        x = torch.rand((n, 1), generator=generator) * 4 - 2
        epsilon = torch.randn((n, 1), generator=generator) * noise
        return x, truth_weight * x + truth_bias + epsilon

    train_x, train_y = make_split(80, seed + 11)
    dev_x, dev_y = make_split(40, seed + 29)
    test_x, test_y = make_split(40, seed + 47)
    model = nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    _emit(emit, "started", lesson="line_fit", device=device, seed=seed,
          explanation="The model starts with a random slope and intercept.")

    for epoch in range(1, epochs + 1):
        model.train()
        x, y = train_x.to(device), train_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        gradient_norm = math.sqrt(sum(float(p.grad.detach().square().sum())
                                      for p in model.parameters() if p.grad is not None))
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            dev_loss = float(loss_fn(model(dev_x.to(device)), dev_y.to(device)))
        weight = float(model.weight.detach().cpu().squeeze())
        bias = float(model.bias.detach().cpu().squeeze())
        _emit(emit, "epoch", epoch=epoch, epochs=epochs,
              train_loss=float(loss.detach().cpu()), dev_loss=dev_loss,
              weight=weight, bias=bias, gradient_norm=gradient_norm)
        if delay:
            time.sleep(delay)

    model.eval()
    with torch.inference_mode():
        prediction = model(test_x.to(device)).cpu()
        test_loss = float(loss_fn(prediction, test_y))
    learned_weight = float(model.weight.detach().cpu().squeeze())
    learned_bias = float(model.bias.detach().cpu().squeeze())
    points = sorted(zip(test_x.squeeze().tolist(), test_y.squeeze().tolist(),
                        prediction.squeeze().tolist()))
    accepted = test_loss <= 0.04 and abs(learned_weight - truth_weight) <= 0.15
    result = {
        "lesson": "line_fit",
        "device": device,
        "seed": seed,
        "data": {"train": 80, "development": 40, "test": 40,
                 "test_used_during_training": False},
        "truth": {"weight": truth_weight, "bias": truth_bias},
        "learned": {"weight": learned_weight, "bias": learned_bias},
        "metrics": {"test_mse": test_loss,
                    "weight_absolute_error": abs(learned_weight - truth_weight),
                    "bias_absolute_error": abs(learned_bias - truth_bias)},
        "gate": {"accepted": accepted, "criteria": "test MSE <= 0.04 and slope error <= 0.15"},
        "plot": [{"x": x, "observed": y, "predicted": p} for x, y, p in points],
        "claim": "Numerical teaching example only; passing does not validate a scientific model.",
    }
    _emit(emit, "completed", result=result)
    return result


def _motion_data(n, seed, *, drift, reference):
    generator = _generator(seed)
    truth = torch.rand((n, 1), generator=generator) * 80 - 60
    nuisance = ((torch.rand((n, 1), generator=generator) * 100 - 50)
                if drift else torch.zeros((n, 1)))
    apparent = truth + nuisance + torch.randn((n, 1), generator=generator) * 2.0
    witness = (nuisance + torch.randn((n, 1), generator=generator) * 2.0
               if reference else torch.zeros((n, 1)))
    features = torch.cat([apparent, witness], dim=1) if reference else apparent
    return features, truth, nuisance


def _fit_motion(train, dev, *, inputs, seed, device, epochs, emit, stage, delay):
    torch.manual_seed(seed)
    model = nn.Linear(inputs, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()
    train_x, train_y = (part.to(device) for part in train[:2])
    dev_x, dev_y = (part.to(device) for part in dev[:2])
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.inference_mode():
                dev_mae = float((model(dev_x) - dev_y).abs().mean())
            _emit(emit, "stage_epoch", stage=stage, epoch=epoch, epochs=epochs,
                  train_loss=float(loss.detach()), development_mae_nm=dev_mae)
        if delay:
            time.sleep(delay)
    return model


def _motion_metrics(model, data, device):
    features, truth, nuisance = data
    model.eval()
    with torch.inference_mode():
        prediction = model(features.to(device)).cpu()
    error = prediction - truth
    return {
        "mae_nm": float(error.abs().mean()),
        "bias_nm": float(error.mean()),
        "rmse_nm": float(error.square().mean().sqrt()),
        "worst_absolute_error_nm": float(error.abs().max()),
        "sample": [{"truth_nm": float(truth[i]), "apparent_nm": float(features[i, 0]),
                    "drift_nm": float(nuisance[i]), "prediction_nm": float(prediction[i])}
                   for i in range(min(12, len(truth)))],
    }


def run_motion_loop(config: dict, emit: EventSink) -> dict:
    """A bounded failure→intervention loop that preserves information-set boundaries."""
    seed = _integer(config.get("seed", 23), "seed", 0, 2**31 - 1)
    epochs = _integer(config.get("epochs", 100), "epochs", 20, 300)
    delay = _finite_number(config.get("delay", 0.005), "delay", 0.0, 0.1)
    device = choose_device(config.get("device", "auto"))
    _emit(emit, "started", lesson="motion_loop", device=device, seed=seed,
          explanation="The test set is created once and never used for weight updates.")

    clean_train = _motion_data(160, seed + 1, drift=False, reference=False)
    clean_dev = _motion_data(80, seed + 2, drift=False, reference=False)
    drift_train = _motion_data(240, seed + 3, drift=True, reference=False)
    drift_dev = _motion_data(80, seed + 4, drift=True, reference=False)
    reference_train = _motion_data(240, seed + 3, drift=True, reference=True)
    reference_dev = _motion_data(80, seed + 4, drift=True, reference=True)
    locked_test_no_reference = _motion_data(160, seed + 1001, drift=True, reference=False)
    locked_test_reference = _motion_data(160, seed + 1001, drift=True, reference=True)

    _emit(emit, "stage", stage="baseline", title="Train on clean apparent motion",
          information="apparent displacement only")
    baseline = _fit_motion(clean_train, clean_dev, inputs=1, seed=seed + 10,
                           device=device, epochs=epochs, emit=emit, stage="baseline", delay=delay)
    baseline_metrics = _motion_metrics(baseline, locked_test_no_reference, device)
    _emit(emit, "diagnosis", stage="baseline", metrics=baseline_metrics,
          finding="Hidden optical drift dominates the held-out error.")

    _emit(emit, "stage", stage="more_data", title="Add varied drift to training",
          information="same apparent displacement only")
    more_data = _fit_motion(drift_train, drift_dev, inputs=1, seed=seed + 20,
                            device=device, epochs=epochs, emit=emit, stage="more_data", delay=delay)
    more_data_metrics = _motion_metrics(more_data, locked_test_no_reference, device)
    _emit(emit, "diagnosis", stage="more_data", metrics=more_data_metrics,
          finding="More examples cannot identify motion and drift from their sum in an individual case.")

    _emit(emit, "stage", stage="reference", title="Add a noisy drift-reference measurement",
          information="apparent displacement plus reference channel")
    referenced = _fit_motion(reference_train, reference_dev, inputs=2, seed=seed + 30,
                             device=device, epochs=epochs, emit=emit, stage="reference", delay=delay)
    reference_metrics = _motion_metrics(referenced, locked_test_reference, device)
    improvement = baseline_metrics["mae_nm"] - reference_metrics["mae_nm"]
    accepted = reference_metrics["mae_nm"] <= 5.0 and improvement >= 10.0
    result = {
        "lesson": "motion_loop",
        "device": device,
        "seed": seed,
        "data": {"locked_test_cases": 160, "test_used_during_training": False,
                 "units": "nm", "generator": "independent seeded synthetic draws"},
        "stages": [
            {"id": "baseline", "name": "Clean-trained model", "information": "apparent only",
             "metrics": baseline_metrics},
            {"id": "more_data", "name": "Drift-trained model", "information": "apparent only",
             "metrics": more_data_metrics},
            {"id": "reference", "name": "Reference-assisted model",
             "information": "apparent + noisy drift reference", "metrics": reference_metrics},
        ],
        "gate": {"accepted": accepted,
                 "criteria": "reference-track MAE <= 5 nm and >= 10 nm better than clean baseline",
                 "mae_improvement_nm": improvement},
        "interpretation": {
            "failure": "Apparent displacement alone combines material motion and optical drift.",
            "same_information_change": "Adding drifted training examples tests robustness but does not restore per-case identifiability.",
            "new_measurement_change": "The reference track supplies additional information and is a different task.",
            "prohibited_claim": "This toy result does not validate a real PDV reconstruction, melting inference, ETI, or MRTI.",
        },
    }
    _emit(emit, "completed", result=result)
    return result


RUNNERS = {"line_fit": run_line_fit, "motion_loop": run_motion_loop}


def run_experiment(kind: str, config: dict, emit: EventSink = lambda event: None) -> dict:
    if kind not in RUNNERS:
        raise ValueError(f"unknown lesson: {kind}")
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    return RUNNERS[kind](config, emit)
