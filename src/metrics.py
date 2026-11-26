"""
Dynamical Metrics Module.

Provides quantitative validation metrics for chaotic dynamical systems and reservoir computing:
1. Valid Prediction Time (VPT): Measured in Lyapunov time units based on normalized error thresholds.
2. Power Spectral Density (PSD) Analysis: Frequency-domain matching via Welch's periodogram.
3. Climate / Ergodic Attractor Metrics: Long-term statistical convergence, including:
   - Component-wise and vector mean / standard deviation
   - Relative covariance Frobenius norm error
   - 1D marginal Wasserstein-1 distance (Earth Mover's Distance)
   - Skewness and kurtosis matching
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal, stats


def compute_normalized_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attractor_std: Optional[float] = None,
) -> np.ndarray:
    """
    Compute normalized Euclidean error over time:
    E(t) = ||y_true(t) - y_pred(t)||_2 / sigma_attractor

    where sigma_attractor = sqrt(mean_t(||y_true(t) - mean(y_true)||^2)).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    diff = y_true - y_pred
    euclidean_error = np.linalg.norm(diff, axis=-1)  # shape (N,)

    if attractor_std is None:
        mean_state = np.mean(y_true, axis=0, keepdims=True)
        fluctuations = np.linalg.norm(y_true - mean_state, axis=-1)
        attractor_std = float(np.sqrt(np.mean(fluctuations**2)))

    if attractor_std <= 0:
        attractor_std = 1.0

    return euclidean_error / attractor_std


def compute_vpt(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt: float,
    mle: float,
    threshold: float = 0.4,
    attractor_std: Optional[float] = None,
) -> Tuple[float, float, int]:
    """
    Compute Valid Prediction Time (VPT).

    VPT is defined as the earliest time at which the normalized Euclidean error
    crosses a specified threshold (typically epsilon in [0.3, 0.5]):
        t_valid = min { t | E(t) > threshold }

    Parameters
    ----------
    y_true : np.ndarray of shape (N, dim)
        Ground truth trajectory.
    y_pred : np.ndarray of shape (N, dim)
        Predicted trajectory.
    dt : float
        Time step between successive samples.
    mle : float
        Maximum Lyapunov Exponent (natural time units^-1).
    threshold : float
        Normalized error threshold (default 0.4).
    attractor_std : Optional[float]
        Standard deviation of attractor. If None, computed from y_true.

    Returns
    -------
    vpt_lyap : float
        Valid Prediction Time expressed in Lyapunov Times (VPT * lambda_max).
    vpt_time : float
        Valid Prediction Time in natural time units.
    vpt_step : int
        Index of the time step where divergence occurred (or N if it never diverged).
    """
    norm_error = compute_normalized_error(y_true, y_pred, attractor_std=attractor_std)
    diverged_indices = np.where(norm_error > threshold)[0]

    N = len(y_true)
    if len(diverged_indices) > 0:
        vpt_step = int(diverged_indices[0])
    else:
        vpt_step = N

    vpt_time = vpt_step * dt
    vpt_lyap = vpt_time * mle

    return float(vpt_lyap), float(vpt_time), vpt_step


def compute_psd(
    trajectory: np.ndarray,
    dt: float,
    nperseg: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Power Spectral Density (PSD) using Welch's averaged periodogram.

    Parameters
    ----------
    trajectory : np.ndarray of shape (N, dim)
        Continuous trajectory.
    dt : float
        Sampling interval.
    nperseg : int
        Length of each segment for FFT.

    Returns
    -------
    freqs : np.ndarray of shape (F,)
        Frequency array (Hz).
    psd : np.ndarray of shape (F, dim)
        Power spectral density for each coordinate.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    fs = 1.0 / dt
    n_samples, dim = trajectory.shape

    # Constrain nperseg if trajectory is short
    nperseg = min(nperseg, n_samples)

    psd_channels = []
    freqs = None
    for d in range(dim):
        f, Pxx = signal.welch(trajectory[:, d], fs=fs, nperseg=nperseg, scaling="density")
        if freqs is None:
            freqs = f
        psd_channels.append(Pxx)

    return freqs, np.column_stack(psd_channels)


def compute_psd_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt: float,
    nperseg: int = 256,
) -> float:
    """
    Compute Log-Spectral Mean Squared Error (L-SMSE) in decibels across all coordinates:
    Error = mean_d,f ( |10*log10(P_pred(f)) - 10*log10(P_true(f))|^2 )
    """
    _, psd_true = compute_psd(y_true, dt, nperseg=nperseg)
    _, psd_pred = compute_psd(y_pred, dt, nperseg=nperseg)

    eps = 1e-12
    db_true = 10.0 * np.log10(np.maximum(psd_true, eps))
    db_pred = 10.0 * np.log10(np.maximum(psd_pred, eps))

    return float(np.mean((db_pred - db_true) ** 2))


def compute_climate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute long-term ergodic attractor metrics (Climate properties).

    Parameters
    ----------
    y_true : np.ndarray of shape (N, dim)
    y_pred : np.ndarray of shape (N, dim)

    Returns
    -------
    metrics : Dict[str, Any]
        Dictionary of statistical comparisons.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    dim = y_true.shape[1]

    # Means
    mean_true = np.mean(y_true, axis=0)
    mean_pred = np.mean(y_pred, axis=0)
    mean_error = float(np.linalg.norm(mean_true - mean_pred))

    # Standard Deviations
    std_true = np.std(y_true, axis=0)
    std_pred = np.std(y_pred, axis=0)
    std_error = float(np.linalg.norm(std_true - std_pred))

    # Covariance Matrix Relative Frobenius Error
    cov_true = np.cov(y_true, rowvar=False)
    cov_pred = np.cov(y_pred, rowvar=False)
    cov_norm_true = np.linalg.norm(cov_true, "fro")
    if cov_norm_true > 0:
        cov_rel_error = float(np.linalg.norm(cov_true - cov_pred, "fro") / cov_norm_true)
    else:
        cov_rel_error = float(np.linalg.norm(cov_true - cov_pred, "fro"))

    # 1D Marginal Wasserstein-1 Distances (Earth Mover's Distance)
    wasserstein_per_coord = [
        float(stats.wasserstein_distance(y_true[:, d], y_pred[:, d])) for d in range(dim)
    ]
    mean_wasserstein = float(np.mean(wasserstein_per_coord))

    # Higher order moments: Skewness & Kurtosis (guarded against constant signals)
    if np.all(std_true > 1e-12) and np.all(std_pred > 1e-12):
        skew_true = stats.skew(y_true, axis=0)
        skew_pred = stats.skew(y_pred, axis=0)
        skew_error = float(np.linalg.norm(skew_true - skew_pred))

        kurt_true = stats.kurtosis(y_true, axis=0)
        kurt_pred = stats.kurtosis(y_pred, axis=0)
        kurt_error = float(np.linalg.norm(kurt_true - kurt_pred))
    else:
        skew_error = 0.0
        kurt_error = 0.0

    return {
        "mean_error": mean_error,
        "mean_true": mean_true.tolist(),
        "mean_pred": mean_pred.tolist(),
        "std_error": std_error,
        "std_true": std_true.tolist(),
        "std_pred": std_pred.tolist(),
        "cov_rel_error": cov_rel_error,
        "wasserstein_per_dim": wasserstein_per_coord,
        "mean_wasserstein": mean_wasserstein,
        "skew_error": skew_error,
        "kurt_error": kurt_error,
    }


def compute_summary_benchmark(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt: float,
    mle: float,
    threshold: float = 0.4,
    psd_nperseg: int = 256,
) -> Dict[str, Any]:
    """
    Compute comprehensive dynamical benchmark metrics between true and predicted trajectories.
    """
    vpt_lyap, vpt_time, vpt_step = compute_vpt(y_true, y_pred, dt=dt, mle=mle, threshold=threshold)
    climate = compute_climate_metrics(y_true, y_pred)
    psd_err = compute_psd_error(y_true, y_pred, dt=dt, nperseg=psd_nperseg)

    # Initial short-term RMSE before divergence (e.g. up to 1 Lyapunov time or up to vpt_step)
    steps_1_lyap = max(1, int(1.0 / (dt * mle)))
    short_term_slice = min(len(y_true), steps_1_lyap)
    rmse_1_lyap = float(np.sqrt(np.mean((y_true[:short_term_slice] - y_pred[:short_term_slice]) ** 2)))

    # Overall RMSE across entire trajectory
    rmse_overall = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "vpt_lyap": vpt_lyap,
        "vpt_time": vpt_time,
        "vpt_step": vpt_step,
        "rmse_1_lyap": rmse_1_lyap,
        "rmse_overall": rmse_overall,
        "psd_error_db2": psd_err,
        "climate_mean_error": climate["mean_error"],
        "climate_std_error": climate["std_error"],
        "climate_cov_rel_error": climate["cov_rel_error"],
        "climate_mean_wasserstein": climate["mean_wasserstein"],
        "climate_skew_error": climate["skew_error"],
        "climate_kurt_error": climate["kurt_error"],
    }
