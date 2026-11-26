"""
Benchmark Experiment: Classical Echo State Network (ESN) vs. Next-Generation Reservoir Computing (NGRC).

Target System: Lorenz-63 Chaotic Attractor (sigma=10.0, rho=28.0, beta=8/3, MLE approx 0.9056).

Workflow:
1. High-precision numerical integration (RK45) of Lorenz-63 system.
2. Train Classical ESN (with spectral radius scaling and quadratic readout).
3. Train Next-Generation Reservoir Computer (NGRC with delay taps and quadratic monomials).
4. Autonomous (closed-loop) forecasting over extended Lyapunov time horizon.
5. Compute quantitative dynamical metrics (VPT in Lyapunov units, PSD error, and Climate invariants).
6. Export publication-quality figures and JSON/CSV metric reports.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

# Ensure src module is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import numpy as np

from src.systems import Lorenz63
from src.esn import EchoStateNetwork
from src.ngrc import NGRC
from src.metrics import (
    compute_normalized_error,
    compute_psd,
    compute_summary_benchmark,
    compute_vpt,
)
from src.plots import (
    plot_3d_attractor_comparison,
    plot_climate_distributions,
    plot_error_growth,
    plot_psd_comparison,
    plot_timeseries_divergence,
)


def run_benchmark(
    dt: float = 0.02,
    train_time: float = 100.0,
    test_time: float = 40.0,
    discard_transient: float = 20.0,
    n_reservoir: int = 500,
    spectral_radius: float = 0.9,
    input_scaling: float = 0.25,
    leak_rate: float = 0.3,
    esn_washout: int = 200,
    quadratic_readout: bool = True,
    esn_reg: float = 1e-3,
    normalize_inputs: bool = True,
    k_delays: int = 2,
    delay_stride: int = 1,
    poly_degree: int = 2,
    ngrc_reg: float = 1e-6,
    seed: int = 42,
    output_dir: str = "results",
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end benchmark between ESN and NGRC."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("RESERVOIR COMPUTING BENCHMARK: CLASSICAL ESN vs. NGRC")
    print("Dynamical System: Lorenz-63 Chaotic Attractor")
    print(f"Time step dt: {dt} | Random Seed: {seed}")
    print("=" * 80)

    # 1. Integrate Lorenz-63 system
    system = Lorenz63()
    total_time = train_time + test_time
    print(f"\n[1/5] Integrating Lorenz-63 (total useful time: {total_time:.1f} s, discard: {discard_transient:.1f} s)...")
    t_full, u_full = system.integrate(
        t_span=total_time,
        dt=dt,
        discard_transient=discard_transient,
    )

    n_train = int(train_time / dt)
    n_test = int(test_time / dt)

    u_train = u_full[:n_train]
    u_test = u_full[n_train : n_train + n_test]
    t_test = t_full[n_train : n_train + n_test] - t_full[n_train]
    t_test_lyap = system.to_lyapunov_time(t_test)

    print(f"  -> Total trajectory points: {len(u_full)}")
    print(f"  -> Training points: {len(u_train)} ({train_time:.1f} time units)")
    print(f"  -> Test horizon: {len(u_test)} points ({test_time:.1f} time units = {t_test_lyap[-1]:.1f} Lyapunov times)")

    # 2. Classical ESN
    print(f"\n[2/5] Training Classical ESN (Nodes: {n_reservoir}, Spectral Radius: {spectral_radius}, Leak: {leak_rate}, Quad: {quadratic_readout}, Norm: {normalize_inputs})...")
    esn = EchoStateNetwork(
        n_inputs=system.dim,
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        sparsity=0.1,
        input_scaling=input_scaling,
        bias_scaling=0.0,
        leak_rate=leak_rate,
        quadratic_readout=quadratic_readout,
        regularization=esn_reg,
        normalize_inputs=normalize_inputs,
        seed=seed,
    )

    start_train_esn = time.perf_counter()
    esn.fit(u_train, washout=esn_washout)
    time_train_esn = time.perf_counter() - start_train_esn

    # Autonomous forecasting with ESN
    u_warmup_esn = u_train[-esn_washout:]
    start_pred_esn = time.perf_counter()
    pred_esn, _ = esn.predict_autonomous(u_warmup=u_warmup_esn, n_steps=len(u_test))
    time_pred_esn = time.perf_counter() - start_pred_esn
    print(f"  -> ESN fit completed in {time_train_esn * 1000:.2f} ms | Readout params: {esn.total_trainable_parameters}")
    print(f"  -> ESN autonomous roll-out completed in {time_pred_esn * 1000:.2f} ms")

    # 3. Next-Generation Reservoir Computing (NGRC)
    print(f"\n[3/5] Training NGRC (Delays k: {k_delays}, Stride s: {delay_stride}, Degree: {poly_degree})...")
    ngrc = NGRC(
        n_inputs=system.dim,
        n_delays=k_delays,
        delay_stride=delay_stride,
        poly_degree=poly_degree,
        include_constant=True,
        regularization=ngrc_reg,
        column_normalized=True,
    )

    start_train_ngrc = time.perf_counter()
    ngrc.fit(u_train)
    time_train_ngrc = time.perf_counter() - start_train_ngrc

    u_warmup_ngrc = u_train[-ngrc.warmup_steps :]
    start_pred_ngrc = time.perf_counter()
    pred_ngrc = ngrc.predict_autonomous(u_warmup=u_warmup_ngrc, n_steps=len(u_test))
    time_pred_ngrc = time.perf_counter() - start_pred_ngrc
    print(f"  -> NGRC fit completed in {time_train_ngrc * 1000:.2f} ms | Feature dimension: {ngrc.n_features} | Readout params: {ngrc.total_trainable_parameters}")
    print(f"  -> NGRC autonomous roll-out completed in {time_pred_ngrc * 1000:.2f} ms")

    # 4. Quantitative Metrics Evaluation
    print("\n[4/5] Computing dynamical & ergodic attractor metrics...")
    metrics_esn = compute_summary_benchmark(
        y_true=u_test,
        y_pred=pred_esn,
        dt=dt,
        mle=system.mle,
        threshold=0.4,
    )
    metrics_ngrc = compute_summary_benchmark(
        y_true=u_test,
        y_pred=pred_ngrc,
        dt=dt,
        mle=system.mle,
        threshold=0.4,
    )

    # Attach performance timings & model specs
    metrics_esn["train_time_sec"] = time_train_esn
    metrics_esn["pred_time_sec"] = time_pred_esn
    metrics_esn["trainable_params"] = esn.total_trainable_parameters
    metrics_esn["feature_dim"] = esn.feature_dim

    metrics_ngrc["train_time_sec"] = time_train_ngrc
    metrics_ngrc["pred_time_sec"] = time_pred_ngrc
    metrics_ngrc["trainable_params"] = ngrc.total_trainable_parameters
    metrics_ngrc["feature_dim"] = ngrc.n_features

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Metric':<35} | {'Classical ESN':<22} | {'NGRC':<22}")
    print("-" * 90)
    print(f"{'Feature Dimension':<35} | {esn.feature_dim:<22} | {ngrc.n_features:<22}")
    print(f"{'Trainable Parameters':<35} | {esn.total_trainable_parameters:<22} | {ngrc.total_trainable_parameters:<22}")
    print(f"{'Training Time':<35} | {time_train_esn*1000:.2f} ms{'':<14} | {time_train_ngrc*1000:.2f} ms{'':<14}")
    print(f"{'Autonomous Rollout Time':<35} | {time_pred_esn*1000:.2f} ms{'':<14} | {time_pred_ngrc*1000:.2f} ms{'':<14}")
    print(f"{'Valid Prediction Time (VPT)':<35} | {metrics_esn['vpt_lyap']:.2f} Lyapunov times    | {metrics_ngrc['vpt_lyap']:.2f} Lyapunov times")
    print(f"{'Short-term RMSE (1 Lyap Time)':<35} | {metrics_esn['rmse_1_lyap']:.4e}{'':<10} | {metrics_ngrc['rmse_1_lyap']:.4e}")
    print(f"{'Overall RMSE (Full Horizon)':<35} | {metrics_esn['rmse_overall']:.4e}{'':<10} | {metrics_ngrc['rmse_overall']:.4e}")
    print(f"{'PSD Error (dB^2)':<35} | {metrics_esn['psd_error_db2']:.4f}{'':<14} | {metrics_ngrc['psd_error_db2']:.4f}")
    print(f"{'Attractor Covariance Rel. Error':<35} | {metrics_esn['climate_cov_rel_error']:.4f}{'':<14} | {metrics_ngrc['climate_cov_rel_error']:.4f}")
    print(f"{'Marginal Wasserstein Distance':<35} | {metrics_esn['climate_mean_wasserstein']:.4f}{'':<14} | {metrics_ngrc['climate_mean_wasserstein']:.4f}")
    print("=" * 90)

    # 5. Export JSON & CSV Reports
    results_dict = {
        "metadata": {
            "system": "Lorenz-63",
            "dt": dt,
            "train_time": train_time,
            "test_time": test_time,
            "mle": system.mle,
            "seed": seed,
        },
        "Classical_ESN": metrics_esn,
        "NGRC": metrics_ngrc,
    }

    json_path = os.path.join(output_dir, "benchmark_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"\n[5/5] Saved JSON report to: {json_path}")

    csv_path = os.path.join(output_dir, "benchmark_metrics.csv")
    csv_rows = [
        "Model,FeatureDim,TrainableParams,TrainTimeMs,InferenceTimeMs,VPT_Lyapunov,RMSE_1LT,RMSE_Overall,PSD_Error_dB2,CovRelError,Wasserstein",
        f"Classical_ESN,{esn.feature_dim},{esn.total_trainable_parameters},{time_train_esn*1000:.2f},{time_pred_esn*1000:.2f},{metrics_esn['vpt_lyap']:.3f},{metrics_esn['rmse_1_lyap']:.5e},{metrics_esn['rmse_overall']:.5e},{metrics_esn['psd_error_db2']:.4f},{metrics_esn['climate_cov_rel_error']:.4f},{metrics_esn['climate_mean_wasserstein']:.4f}",
        f"NGRC,{ngrc.n_features},{ngrc.total_trainable_parameters},{time_train_ngrc*1000:.2f},{time_pred_ngrc*1000:.2f},{metrics_ngrc['vpt_lyap']:.3f},{metrics_ngrc['rmse_1_lyap']:.5e},{metrics_ngrc['rmse_overall']:.5e},{metrics_ngrc['psd_error_db2']:.4f},{metrics_ngrc['climate_cov_rel_error']:.4f},{metrics_ngrc['climate_mean_wasserstein']:.4f}",
    ]
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"  -> Saved CSV report to: {csv_path}")

    # 6. Generate Publication-Quality Figures
    if save_plots:
        print("  -> Generating high-resolution publication plots in results/ ...")

        # 1. 3D Attractor Comparison
        plot_3d_attractor_comparison(
            trajectories={
                "Ground Truth": u_test,
                "Classical ESN": pred_esn,
                "NGRC": pred_ngrc,
            },
            save_path=os.path.join(output_dir, "attractor_3d_comparison.png"),
        )

        # 2. Time-series Divergence
        plot_timeseries_divergence(
            t_lyap=t_test_lyap,
            y_true=u_test,
            y_esn=pred_esn,
            y_ngrc=pred_ngrc,
            vpt_esn=metrics_esn["vpt_lyap"],
            vpt_ngrc=metrics_ngrc["vpt_lyap"],
            max_lyap=12.0,
            save_path=os.path.join(output_dir, "timeseries_divergence.png"),
        )

        # 3. Error Growth Curve
        err_esn = compute_normalized_error(u_test, pred_esn)
        err_ngrc = compute_normalized_error(u_test, pred_ngrc)
        plot_error_growth(
            t_lyap=t_test_lyap,
            err_esn=err_esn,
            err_ngrc=err_ngrc,
            threshold=0.4,
            vpt_esn=metrics_esn["vpt_lyap"],
            vpt_ngrc=metrics_ngrc["vpt_lyap"],
            max_lyap=12.0,
            save_path=os.path.join(output_dir, "error_growth_vpt.png"),
        )

        # 4. Power Spectral Density (PSD)
        freqs, psd_truth = compute_psd(u_test, dt=dt)
        _, psd_esn = compute_psd(pred_esn, dt=dt)
        _, psd_ngrc = compute_psd(pred_ngrc, dt=dt)
        plot_psd_comparison(
            freqs=freqs,
            psd_dict={
                "Ground Truth": psd_truth,
                "Classical ESN": psd_esn,
                "NGRC": psd_ngrc,
            },
            save_path=os.path.join(output_dir, "power_spectral_density.png"),
        )

        # 5. Marginal Climate Invariant Distributions
        plot_climate_distributions(
            trajectories={
                "Ground Truth": u_test,
                "Classical ESN": pred_esn,
                "NGRC": pred_ngrc,
            },
            save_path=os.path.join(output_dir, "climate_distributions.png"),
        )
        print("  -> All 5 publication figures successfully exported!")

    print("\nBenchmark experiment finished successfully.")
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Classical ESN vs. NGRC on Chaotic Dynamics"
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Integration time step")
    parser.add_argument("--train-time", type=float, default=100.0, help="Training time units")
    parser.add_argument("--test-time", type=float, default=40.0, help="Test horizon time units")
    parser.add_argument("--n-res", type=int, default=500, help="Number of ESN reservoir nodes")
    parser.add_argument("--spectral-radius", type=float, default=0.9, help="ESN spectral radius")
    parser.add_argument("--input-scaling", type=float, default=0.25, help="ESN input scaling")
    parser.add_argument("--leak-rate", type=float, default=0.3, help="ESN leak rate")
    parser.add_argument("--esn-reg", type=float, default=1e-3, help="ESN ridge regularization")
    parser.add_argument("--no-esn-norm", action="store_true", help="Disable ESN input standardization")
    parser.add_argument("--k-delays", type=int, default=2, help="Number of NGRC delay taps")
    parser.add_argument("--stride", type=int, default=1, help="NGRC delay stride")
    parser.add_argument("--ngrc-reg", type=float, default=1e-6, help="NGRC ridge regularization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output files")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(parent_dir, out_dir)

    run_benchmark(
        dt=args.dt,
        train_time=args.train_time,
        test_time=args.test_time,
        n_reservoir=args.n_res,
        spectral_radius=args.spectral_radius,
        input_scaling=args.input_scaling,
        leak_rate=args.leak_rate,
        esn_reg=args.esn_reg,
        normalize_inputs=not args.no_esn_norm,
        k_delays=args.k_delays,
        delay_stride=args.stride,
        ngrc_reg=args.ngrc_reg,
        seed=args.seed,
        output_dir=out_dir,
        save_plots=not args.no_plots,
    )
