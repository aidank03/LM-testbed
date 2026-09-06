"""Factor: bounded scientific workflows and explicit evaluation contracts."""

__version__ = "0.3.0"


def run(request, *, runtime):
    """Run a scientific question with an explicitly configured runtime."""
    return runtime.run(request)

