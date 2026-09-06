"""Portable run metadata without guessing CPU model, physical cores, or RAM."""
import os
import platform


def _detected(probe):
    try:
        value = probe()
    except OSError:
        return "unknown"
    return value if value not in (None, "") else "unknown"


def hardware_metadata(device):
    """Record available host facts and the device actually selected by training."""
    return {
        "os": _detected(platform.system),
        "os_release": _detected(platform.release),
        "architecture": _detected(platform.machine),
        "selected_torch_device": str(device),
        "logical_cpu_count": _detected(os.cpu_count),
        "cpu_model": "unknown",
        "physical_cpu_cores": "unknown",
        "memory_bytes": "unknown",
    }
