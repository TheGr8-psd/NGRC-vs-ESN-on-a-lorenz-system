"""
Reservoir Computing & Nonlinear Dynamics Benchmark Package.
"""

from .systems import DynamicalSystem, Lorenz63, Rossler
from .esn import EchoStateNetwork
from .ngrc import NGRC
from .metrics import (
    compute_normalized_error,
    compute_vpt,
    compute_psd,
    compute_psd_error,
    compute_climate_metrics,
    compute_summary_benchmark,
)
from .plots import (
    plot_3d_attractor_comparison,
    plot_timeseries_divergence,
    plot_error_growth,
    plot_psd_comparison,
    plot_climate_distributions,
)

__all__ = [
    "DynamicalSystem",
    "Lorenz63",
    "Rossler",
    "EchoStateNetwork",
    "NGRC",
    "compute_normalized_error",
    "compute_vpt",
    "compute_psd",
    "compute_psd_error",
    "compute_climate_metrics",
    "compute_summary_benchmark",
    "plot_3d_attractor_comparison",
    "plot_timeseries_divergence",
    "plot_error_growth",
    "plot_psd_comparison",
    "plot_climate_distributions",
]
