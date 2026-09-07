"""Factor Learning Lab: observable local ML experiments."""


def run_experiment(*args, **kwargs):
    """Load the optional PyTorch experiment implementation on first use."""
    from .experiments import run_experiment as _run_experiment

    return _run_experiment(*args, **kwargs)

__all__ = ["run_experiment"]
