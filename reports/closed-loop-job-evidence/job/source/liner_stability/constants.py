from scipy.special import ndtri
from . import __version__

VERSION = __version__
SCENARIOS = ("nominal", "blurred", "pdv_dropout", "timing_jitter",
             "omitted_optical_drift", "no_structure")
UNITS = {"compression_nm": "nm", "turnaround_ns": "ns", "mode_amplitude_um": "um"}
Z90 = float(ndtri(0.95))
