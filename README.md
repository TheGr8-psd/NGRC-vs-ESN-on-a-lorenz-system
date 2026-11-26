# Reservoir Computing Benchmark: Classical ESN vs. Next-Generation RC (NGRC)

A modular, reproducible computational physics and machine learning research codebase benchmarking **Classical Echo State Networks (ESN)** against **Next-Generation Reservoir Computing (NGRC)** on chaotic dynamical systems (specifically the canonical **Lorenz-63** attractor).

---

## 1. Overview & Theoretical Background

### Dynamical System: Lorenz-63 Attractor
The Lorenz-63 system is defined by the coupled nonlinear ODEs:
$$\frac{dx}{dt} = \sigma (y - x)$$
$$\frac{dy}{dt} = x (\rho - z) - y$$
$$\frac{dz}{dt} = x y - \beta z$$

With canonical chaotic parameters $\sigma = 10.0$, $\rho = 28.0$, $\beta = 8/3$, the system possesses a maximum Lyapunov exponent $\lambda_{\max} \approx 0.9056$, defining a characteristic Lyapunov timescale $`T_{\lambda} = 1 / \lambda_{\max} \approx 1.104`$ seconds.

### Classical Echo State Networks (ESN)
- Input standardization (zero mean, unit variance computed on training data) to prevent `tanh` saturation.
- Leaky recurrent reservoir: $`\mathbf{r}(t+1) = (1 - \alpha) \mathbf{r}(t) + \alpha \tanh(\mathbf{W}_{\text{res}} \mathbf{r}(t) + \mathbf{W}_{\text{in}} \mathbf{u}(t) + \mathbf{b})`$ ($\alpha = 0.3$, `input_scaling = 0.25`).
- Echo State Property governed by spectral radius scaling $\rho(\mathbf{W}_{\text{res}}) = 0.90 < 1.0$.
- Feature representation: Includes a quadratic state readout $[\mathbf{1}, \mathbf{r}(t), \mathbf{r}^2(t)]$ to capture nonlinear quadratic dependencies.
- Readout optimization: Regularized Tikhonov Ridge Regression ($\beta = 10^{-3}$).

### Next-Generation Reservoir Computing (NGRC / NVAR)
As introduced by Gauthier et al. (*Nature Communications* 12, 5564, 2021), NGRC replaces the high-dimensional random recurrent neural network with an explicit, deterministic feature library:
1. **Linear delay coordinates**: $[\mathbf{u}(t), \mathbf{u}(t-s), \dots, \mathbf{u}(t-(k-1)s)]$ with $k$ taps and stride $s$.
2. **Constant bias**: $c = 1$.
3. **Polynomial interaction features**: Unique degree-2 monomials $\{u_i u_j\}$.
For Lorenz-63 with $d=3, k=2, s=1$, the entire feature space requires only **28 dimensions** (compared to $1001$ for a 500-node ESN).

---

## 2. Directory Layout

```text
D:\BITS\4-1\SOP\
├── src\
│   ├── __init__.py
│   ├── systems.py       # Numerical integration (RK45) of Lorenz-63 & Rössler systems
│   ├── esn.py           # Pure NumPy ESN (spectral radius scaling, washout, quadratic readout)
│   ├── ngrc.py          # NGRC (Gauthier et al. 2021 polynomial delay features)
│   ├── metrics.py       # Dynamical metrics: VPT (Lyapunov units), PSD, & Climate metrics
│   └── plots.py         # Publication-quality 3D attractor, time-series divergence, & PSD plots
├── experiments\
│   └── run_benchmark.py # Main execution pipeline comparing ESN vs. NGRC
├── tests\
│   └── test_benchmark.py# Comprehensive unit and integration test suite
├── results\             # Benchmark figures and JSON/CSV metric reports
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 3. Benchmark Experimental Results

Evaluated on autonomous closed-loop forecasting for $40.0$ natural time units ($36.2$ Lyapunov times) after training on $100.0$ time units ($5,000$ points at $dt = 0.02$):

| Metric | Classical ESN | Next-Generation RC (NGRC) | Advantage / Note |
| :--- | :---: | :---: | :--- |
| **Internal Feature Dimension** | $1,001$ | **$28$** | **$35.75\times$ more compact** |
| **Trainable Readout Parameters** | $3,003$ | **$84$** | **$35.75\times$ fewer parameters** |
| **Training Time** | $282.03\text{ ms}$ | **$258.52\text{ ms}$** | Faster & fully deterministic |
| **Autonomous Rollout Time** | **$95.74\text{ ms}$** | $104.72\text{ ms}$ | Real-time generation |
| **Valid Prediction Time (VPT)** | **$5.29\text{ LT}$** | $5.22\text{ LT}$ | Comparable tracking horizon |
| **Short-term RMSE (1 Lyap Time)**| **$8.9706\times 10^{-3}$** | $3.6738\times 10^{-2}$ | Both track closely in short horizon |
| **Broadband PSD Error** | $5.50\text{ dB}^2$ | **$3.31\text{ dB}^2$** | NGRC closer across high frequencies |
| **Attractor Covariance Rel. Error**| **$0.0599$** | $0.0876$ | Both preserve $<9\%$ climate error |
| **Marginal Wasserstein Distance** | **$0.8728$** | $1.5987$ | Both capture invariant measure |

---

## 4. Key Takeaways & Physics Insights

1. **Parameter Efficiency**: NGRC matches a well-tuned, input-standardized 500-node Classical ESN ($5.22\text{ LT}$ vs. $5.29\text{ LT}$ VPT) using only **28 features** and **84 parameters** ($35.75\times$ smaller), eliminating random weight generation and hyperparameter tuning (`input_scaling`, `spectral_radius`, `leak_rate`, `sparsity`).
2. **Climate / Ergodic Invariance**: After point-wise chaotic divergence ($t > 5.2\text{ LT}$), both models remain stably bounded on the Lorenz-63 strange attractor, with NGRC achieving a lower broadband PSD error ($3.31\text{ dB}^2$ vs. $5.50\text{ dB}^2$) and both achieving $<9\%$ relative covariance error.
3. **Reproducibility**: NGRC is 100% deterministic with zero stochastic weight initialization seeds required.

---

## 5. Execution Instructions

### Running the Benchmark
```bash
python experiments/run_benchmark.py --dt 0.02 --train-time 100.0 --test-time 40.0
```

### Running Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Generated Artifacts in `results/`:
- `attractor_3d_comparison.png`: 3D phase space reconstruction of butterfly wings.
- `timeseries_divergence.png`: State tracking over Lyapunov times with VPT markers.
- `error_growth_vpt.png`: Semi-log error growth curve showing exponential divergence.
- `power_spectral_density.png`: Log-scale Welch periodogram comparing broadband spectra.
- `climate_distributions.png`: Marginal invariant probability density functions (PDFs).
- `benchmark_metrics.json` & `benchmark_metrics.csv`: Machine-readable benchmark logs.
