"""Explicit-unit analytic checks. These are not an MHD solver or material model."""
import math

MU0 = 4 * math.pi * 1e-7  # specified approximation in H/m


def positive(name, value, allow_zero=False):
    if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def magnetic_pressure(current_A, radius_m):
    current_A = positive("current_A", current_A, allow_zero=True)
    radius_m = positive("radius_m", radius_m)
    field = MU0 * current_A / (2 * math.pi * radius_m)
    return {"value": field**2 / (2 * MU0), "unit": "Pa", "field_T": field,
            "equation": "B=mu0 I/(2 pi R); p=B^2/(2 mu0)",
            "assumptions": ["long cylindrical conductor", "exterior azimuthal field only", "specified mu0 approximation"]}


def diffusion_length(resistivity_ohm_m, time_s):
    eta = positive("resistivity_ohm_m", resistivity_ohm_m)
    t = positive("time_s", time_s, allow_zero=True)
    return {"value": math.sqrt(eta * t / MU0), "unit": "m", "equation": "ell=sqrt(eta t/mu0)",
            "assumptions": ["constant electrical resistivity", "transient diffusion scale convention, not sinusoidal skin depth"]}


def latent_heat_duration(density_kg_m3, latent_heat_J_kg, resistivity_ohm_m, current_density_A_m2):
    rho = positive("density_kg_m3", density_kg_m3)
    latent = positive("latent_heat_J_kg", latent_heat_J_kg)
    eta = positive("resistivity_ohm_m", resistivity_ohm_m)
    j = positive("current_density_A_m2", current_density_A_m2)
    return {"value": rho * latent / (eta*j*j), "unit": "s", "equation": "dt=rho L/(eta J^2)",
            "assumptions": ["already at melting temperature", "uniform constant density, resistivity and current density",
                            "latent heat only; no work, conduction or radiation", "not a rod melt prediction"]}


def event_delay(structure_time_ns, melt_time_ns, structure_sigma_ns, melt_sigma_ns, correlation):
    for name, value in (("structure_time_ns", structure_time_ns), ("melt_time_ns", melt_time_ns), ("correlation", correlation)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not -1 <= correlation <= 1:
        raise ValueError("correlation must be between -1 and 1")
    s1 = positive("structure_sigma_ns", structure_sigma_ns, allow_zero=True)
    s2 = positive("melt_sigma_ns", melt_sigma_ns, allow_zero=True)
    variance = s1*s1 + s2*s2 - 2*correlation*s1*s2
    return {"value": structure_time_ns-melt_time_ns, "unit": "ns", "standard_deviation_ns": math.sqrt(max(0, variance)),
            "equation": "Var(t1-t2)=Var(t1)+Var(t2)-2 Cov(t1,t2)",
            "assumptions": ["supplied times refer to the intended common physical patch and clock", "phase-label validity is not established by this calculation"]}


CALCULATIONS = {"magnetic_pressure": magnetic_pressure, "diffusion_length": diffusion_length,
                "latent_heat_duration": latent_heat_duration, "event_delay": event_delay}


def calculate(request):
    if set(request) != {"calculation", "parameters"}:
        raise ValueError("Calculation request requires exactly calculation and parameters")
    name = request["calculation"]
    if name not in CALCULATIONS or not isinstance(request["parameters"], dict):
        raise ValueError("Unknown calculation or invalid parameters")
    try:
        result = CALCULATIONS[name](**request["parameters"])
    except TypeError as exc:
        raise ValueError(f"Wrong parameters for {name}: {exc}") from None
    if not math.isfinite(result["value"]):
        raise ValueError("Calculation overflowed its numerical range")
    return {"calculation": name, "inputs": request["parameters"], "result": result,
            "status": "analytic_check_under_stated_assumptions"}
