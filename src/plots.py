"""
Publication-Quality Visualization Module for Chaotic Dynamics & Reservoir Computing.

Generates high-resolution publication-grade figures:
1. 3D Phase Space Attractor Comparison (Side-by-side or overlaid butterfly wings)
2. Time-Series Divergence & Tracking Horizon (Lyapunov time units with VPT markers)
3. Power Spectral Density (PSD) Spectral Cascade
4. Climate / Ergodic Marginal Probability Distributions (PDFs)
5. Comprehensive Benchmark Summary Dashboard
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Ensure headless rendering without Qt/GUI backend conflicts
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

# Set standard publication styling
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.antialiased": True,
    }
)

COLORS = {
    "truth": "#1f2937",  # Dark slate
    "esn": "#dc2626",    # Crimson red
    "ngrc": "#2563eb",   # Royal blue
    "vpt_esn": "#ef4444",
    "vpt_ngrc": "#3b82f6",
}


def plot_3d_attractor_comparison(
    trajectories: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    elev: float = 22.0,
    azim: float = 48.0,
    title_suffix: str = "Lorenz-63 Attractor",
    dpi: int = 300,
) -> plt.Figure:
    """
    Side-by-side 3D phase space plots comparing Ground Truth against Reservoir models.

    Parameters
    ----------
    trajectories : Dict[str, np.ndarray]
        Dictionary mapping label ("Ground Truth", "Classical ESN", "NGRC") to (N, 3) trajectories.
    save_path : Optional[str]
        File path to save the high-res figure.
    elev, azim : float
        3D camera view angles.
    """
    n_models = len(trajectories)
    fig = plt.figure(figsize=(5.0 * n_models, 4.8), dpi=dpi)

    plot_colors = [COLORS["truth"], COLORS["esn"], COLORS["ngrc"], "#10b981"]

    for idx, (label, traj) in enumerate(trajectories.items()):
        ax = fig.add_subplot(1, n_models, idx + 1, projection="3d")
        c = plot_colors[idx % len(plot_colors)]

        # Trajectory line
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            lw=0.6,
            color=c,
            alpha=0.85,
            label=label,
        )

        # Mark starting point
        ax.scatter(
            [traj[0, 0]],
            [traj[0, 1]],
            [traj[0, 2]],
            color="#000000",
            s=25,
            edgecolors="white",
            linewidths=0.8,
            zorder=10,
        )

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{label}\n({title_suffix})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(r"$x$", labelpad=2)
        ax.set_ylabel(r"$y$", labelpad=2)
        ax.set_zlabel(r"$z$", labelpad=2)
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return fig


def plot_timeseries_divergence(
    t_lyap: np.ndarray,
    y_true: np.ndarray,
    y_esn: np.ndarray,
    y_ngrc: np.ndarray,
    vpt_esn: Optional[float] = None,
    vpt_ngrc: Optional[float] = None,
    max_lyap: float = 12.0,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Multi-panel time series divergence plot showing state tracking over Lyapunov times.
    """
    mask = t_lyap <= max_lyap
    t_crop = t_lyap[mask]
    y_true_c = y_true[mask]
    y_esn_c = y_esn[mask]
    y_ngrc_c = y_ngrc[mask]

    var_names = [r"$x(t)$", r"$y(t)$", r"$z(t)$"]
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 6.8), sharex=True, dpi=dpi)

    for i in range(3):
        ax = axes[i]
        ax.plot(t_crop, y_true_c[:, i], color=COLORS["truth"], lw=1.6, label="Ground Truth")
        ax.plot(
            t_crop,
            y_esn_c[:, i],
            color=COLORS["esn"],
            lw=1.2,
            linestyle="--",
            alpha=0.9,
            label="Classical ESN",
        )
        ax.plot(
            t_crop,
            y_ngrc_c[:, i],
            color=COLORS["ngrc"],
            lw=1.2,
            linestyle="-.",
            alpha=0.9,
            label="NGRC",
        )

        # Plot VPT vertical lines
        if vpt_esn is not None and vpt_esn <= max_lyap:
            ax.axvline(
                vpt_esn,
                color=COLORS["vpt_esn"],
                linestyle=":",
                lw=1.2,
                label=f"ESN VPT ({vpt_esn:.1f} LT)" if i == 0 else None,
            )
        if vpt_ngrc is not None and vpt_ngrc <= max_lyap:
            ax.axvline(
                vpt_ngrc,
                color=COLORS["vpt_ngrc"],
                linestyle=":",
                lw=1.2,
                label=f"NGRC VPT ({vpt_ngrc:.1f} LT)" if i == 0 else None,
            )

        ax.set_ylabel(var_names[i], fontsize=11, fontweight="medium")
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[-1].set_xlabel(r"Lyapunov Time Scale $[\lambda_{\mathrm{max}} \cdot t]$", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True, ncol=5, fontsize=8.5)
    fig.suptitle("Chaotic Trajectory Divergence: Ground Truth vs ESN vs NGRC", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return fig


def plot_error_growth(
    t_lyap: np.ndarray,
    err_esn: np.ndarray,
    err_ngrc: np.ndarray,
    threshold: float = 0.4,
    vpt_esn: Optional[float] = None,
    vpt_ngrc: Optional[float] = None,
    max_lyap: float = 12.0,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Logarithmic error growth over Lyapunov times, illustrating exponential error divergence.
    """
    mask = t_lyap <= max_lyap
    t_c = t_lyap[mask]
    e_esn = err_esn[mask]
    e_ngrc = err_ngrc[mask]

    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=dpi)

    ax.semilogy(t_c, np.maximum(e_esn, 1e-6), color=COLORS["esn"], lw=1.5, label="Classical ESN Error")
    ax.semilogy(t_c, np.maximum(e_ngrc, 1e-6), color=COLORS["ngrc"], lw=1.5, label="NGRC Error")
    ax.axhline(threshold, color="#6b7280", linestyle="--", lw=1.2, label=f"VPT Threshold ({threshold})")

    if vpt_esn is not None and vpt_esn <= max_lyap:
        ax.axvline(vpt_esn, color=COLORS["vpt_esn"], linestyle=":", lw=1.2, label=f"ESN Horizon: {vpt_esn:.2f} LT")
    if vpt_ngrc is not None and vpt_ngrc <= max_lyap:
        ax.axvline(vpt_ngrc, color=COLORS["vpt_ngrc"], linestyle=":", lw=1.2, label=f"NGRC Horizon: {vpt_ngrc:.2f} LT")

    ax.set_xlabel(r"Lyapunov Time $[\lambda_{\mathrm{max}} \cdot t]$", fontsize=11)
    ax.set_ylabel(r"Normalized Error $E(t) / \sigma_{\mathrm{attractor}}$", fontsize=11)
    ax.set_title("Error Growth & Divergence Horizon", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return fig


def plot_psd_comparison(
    freqs: np.ndarray,
    psd_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Power Spectral Density (PSD) comparison across coordinates on semilogy scale.
    """
    dim = next(iter(psd_dict.values())).shape[1]
    coord_labels = [r"$x$", r"$y$", r"$z$"]
    fig, axes = plt.subplots(1, dim, figsize=(4.4 * dim, 3.8), dpi=dpi)
    if dim == 1:
        axes = [axes]

    colors = [COLORS["truth"], COLORS["esn"], COLORS["ngrc"], "#10b981"]

    for d in range(dim):
        ax = axes[d]
        for idx, (label, psd) in enumerate(psd_dict.items()):
            c = colors[idx % len(colors)]
            lw = 1.4 if "Truth" in label else 1.1
            ls = "-" if "Truth" in label else ("--" if "ESN" in label else "-.")
            ax.semilogy(freqs, psd[:, d], label=label, color=c, lw=lw, linestyle=ls, alpha=0.85)

        ax.set_title(f"PSD - Coordinate {coord_labels[d]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Frequency [Hz]", fontsize=10)
        if d == 0:
            ax.set_ylabel(r"Power Spectral Density $[V^2 / \mathrm{Hz}]$", fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.set_xlim(left=0.0, right=min(10.0, float(np.max(freqs))))

    axes[0].legend(loc="upper right", frameon=True, fontsize=8.5)
    fig.suptitle("Broadband Chaotic Frequency Spectra", fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return fig


def plot_climate_distributions(
    trajectories: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    bins: int = 50,
    dpi: int = 300,
) -> plt.Figure:
    """
    Marginal empirical probability density functions (PDFs) of attractor coordinates.
    """
    dim = next(iter(trajectories.values())).shape[1]
    coord_labels = [r"$x$", r"$y$", r"$z$"]
    fig, axes = plt.subplots(1, dim, figsize=(4.4 * dim, 3.8), dpi=dpi)
    if dim == 1:
        axes = [axes]

    colors = [COLORS["truth"], COLORS["esn"], COLORS["ngrc"], "#10b981"]

    for d in range(dim):
        ax = axes[d]
        for idx, (label, traj) in enumerate(trajectories.items()):
            c = colors[idx % len(colors)]
            # Filter finite values if any
            finite_vals = traj[:, d][np.isfinite(traj[:, d])]
            if len(finite_vals) > 0:
                counts, bin_edges = np.histogram(finite_vals, bins=bins, density=True)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                lw = 1.6 if "Truth" in label else 1.2
                ls = "-" if "Truth" in label else ("--" if "ESN" in label else "-.")
                ax.plot(bin_centers, counts, label=label, color=c, lw=lw, linestyle=ls)

        ax.set_title(f"Marginal PDF: {coord_labels[d]}", fontsize=11, fontweight="bold")
        ax.set_xlabel(coord_labels[d], fontsize=10)
        if d == 0:
            ax.set_ylabel("Probability Density", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].legend(loc="upper right", frameon=True, fontsize=8.5)
    fig.suptitle("Attractor Climate Invariant Measure Reconstruction", fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return fig
