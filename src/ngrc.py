"""
Next-Generation Reservoir Computing (NGRC) Module.

Pure NumPy implementation based on Gauthier et al. (2021),
"Next generation reservoir computing", Nature Communications 12, 5564.

NGRC replaces the high-dimensional random recurrent network of classical ESNs
with a compact, fully deterministic feature library composed of:
1. Linear delay taps: u(t), u(t - s), ..., u(t - (k-1)s)
2. Constant bias term: 1
3. Unique polynomial interaction features (e.g., degree-2 quadratic monomials)

This module includes column-norm preconditioned Tikhonov ridge regression to guarantee
numerical stability across disparate feature scales (from O(1) constants to O(10^3) polynomials).
"""

from typing import List, Optional, Tuple
import itertools
import numpy as np
from scipy import linalg


class NGRC:
    """
    Next-Generation Reservoir Computing (NVAR / Polynomial Delay Network).

    Parameters
    ----------
    n_inputs : int
        Dimension of the input state vector u(t).
    n_delays : int
        Number of time delay taps k >= 1.
    delay_stride : int
        Sampling interval multiplier s >= 1 between successive delay taps.
    poly_degree : int
        Maximum polynomial degree for nonlinear interactions (typically 2 for quadratic vector fields).
    include_constant : bool
        Whether to include a constant bias term (= 1.0) in the feature library.
    regularization : float
        L2 Tikhonov ridge regression parameter beta >= 0.
    column_normalized : bool
        Whether to precondition the feature matrix with column-norm scaling before solving
        the normal equations, ensuring numerical stability across widely differing feature magnitudes.
    """

    def __init__(
        self,
        n_inputs: int,
        n_delays: int = 2,
        delay_stride: int = 1,
        poly_degree: int = 2,
        include_constant: bool = True,
        regularization: float = 1e-6,
        column_normalized: bool = True,
    ):
        if n_delays < 1:
            raise ValueError("n_delays must be at least 1.")
        if delay_stride < 1:
            raise ValueError("delay_stride must be at least 1.")
        if poly_degree < 1:
            raise ValueError("poly_degree must be at least 1.")

        self.n_inputs = int(n_inputs)
        self.n_delays = int(n_delays)
        self.delay_stride = int(delay_stride)
        self.poly_degree = int(poly_degree)
        self.include_constant = bool(include_constant)
        self.regularization = float(regularization)
        self.column_normalized = bool(column_normalized)

        # Minimum history required to construct one feature vector
        self.warmup_steps = (self.n_delays - 1) * self.delay_stride + 1

        # Total linear features = n_inputs * n_delays
        self.n_linear = self.n_inputs * self.n_delays

        # Precompute polynomial monomial indices
        self._mono_indices: List[Tuple[int, ...]] = []
        for deg in range(2, self.poly_degree + 1):
            combos = list(itertools.combinations_with_replacement(range(self.n_linear), deg))
            self._mono_indices.extend(combos)

        self.n_nonlinear = len(self._mono_indices)
        self.n_features = (
            (1 if self.include_constant else 0) + self.n_linear + self.n_nonlinear
        )

        # Readout weights: shape (n_inputs, n_features)
        self.W_out: Optional[np.ndarray] = None

    def _extract_linear_delays(self, window: np.ndarray) -> np.ndarray:
        """
        Extract flattened linear delay vector from a time window of length self.warmup_steps.

        window: shape (warmup_steps, n_inputs) ordered chronologically [t - (k-1)s, ..., t].
        Returns: 1D array of shape (n_linear,) with delays [u(t), u(t-s), ..., u(t - (k-1)s)].
        """
        delays = []
        for i in range(self.n_delays):
            idx = -1 - (i * self.delay_stride)
            delays.append(window[idx])
        return np.concatenate(delays)

    def _construct_features_from_linear(self, lin_vec: np.ndarray) -> np.ndarray:
        """Construct full feature vector [1, O_lin, O_poly] from linear delay vector."""
        parts = []
        if self.include_constant:
            parts.append(np.array([1.0], dtype=np.float64))

        parts.append(lin_vec)

        if self.n_nonlinear > 0:
            poly_vals = np.array(
                [np.prod(lin_vec[list(idx)]) for idx in self._mono_indices],
                dtype=np.float64,
            )
            parts.append(poly_vals)

        return np.concatenate(parts)

    def compute_feature_matrix(self, u_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct feature matrix X and corresponding one-step-ahead targets Y.

        Parameters
        ----------
        u_series : np.ndarray of shape (N, n_inputs)

        Returns
        -------
        X : np.ndarray of shape (N - warmup_steps, n_features)
        Y : np.ndarray of shape (N - warmup_steps, n_inputs)
        """
        N = len(u_series)
        n_samples = N - self.warmup_steps
        if n_samples <= 0:
            raise ValueError(
                f"Trajectory length ({N}) must be strictly greater than warmup requirement ({self.warmup_steps})."
            )

        X = np.zeros((n_samples, self.n_features), dtype=np.float64)
        Y = np.zeros((n_samples, self.n_inputs), dtype=np.float64)

        for i in range(n_samples):
            t_idx = self.warmup_steps - 1 + i
            window = u_series[i : t_idx + 1]
            lin_feat = self._extract_linear_delays(window)
            X[i] = self._construct_features_from_linear(lin_feat)
            Y[i] = u_series[t_idx + 1]  # Target is u(t + 1)

        return X, Y

    def fit(self, u_train: np.ndarray) -> "NGRC":
        """
        Fit the NGRC readout weights using Ridge Regression.

        Parameters
        ----------
        u_train : np.ndarray of shape (N, n_inputs)
            Training trajectory.

        Returns
        -------
        self : NGRC
        """
        u_train = np.asarray(u_train, dtype=np.float64)
        X, Y = self.compute_feature_matrix(u_train)

        if self.column_normalized:
            # Column-norm preconditioning
            col_norms = np.linalg.norm(X, axis=0)
            col_norms[col_norms == 0.0] = 1.0
            X_scaled = X / col_norms

            reg_matrix = X_scaled.T @ X_scaled + self.regularization * np.eye(self.n_features)
            rhs = X_scaled.T @ Y

            try:
                w_scaled_T = linalg.solve(reg_matrix, rhs, assume_a="pos")
            except (linalg.LinAlgError, ValueError):
                w_scaled_T, _, _, _ = linalg.lstsq(reg_matrix, rhs)

            self.W_out = (w_scaled_T / col_norms[:, None]).T
        else:
            reg_matrix = X.T @ X + self.regularization * np.eye(self.n_features)
            rhs = X.T @ Y

            try:
                w_out_T = linalg.solve(reg_matrix, rhs, assume_a="pos")
            except (linalg.LinAlgError, ValueError):
                w_out_T, _, _, _ = linalg.lstsq(reg_matrix, rhs)

            self.W_out = w_out_T.T

        return self

    def predict_autonomous(
        self,
        u_warmup: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Autonomous (closed-loop) iterative prediction.

        Parameters
        ----------
        u_warmup : np.ndarray of shape (L_warm, n_inputs)
            Historical trajectory segment of length >= self.warmup_steps.
        n_steps : int
            Number of future time steps to generate.

        Returns
        -------
        predictions : np.ndarray of shape (n_steps, n_inputs)
            Autonomously predicted trajectory.
        """
        if self.W_out is None:
            raise RuntimeError("Model is not fitted. Call fit() before predicting.")

        u_warmup = np.asarray(u_warmup, dtype=np.float64)
        if len(u_warmup) < self.warmup_steps:
            raise ValueError(
                f"u_warmup length ({len(u_warmup)}) must be >= required warmup steps ({self.warmup_steps})."
            )

        # Initialize rolling buffer with the most recent warmup observations
        buffer = list(u_warmup[-self.warmup_steps :])
        predictions = np.zeros((n_steps, self.n_inputs), dtype=np.float64)

        for step in range(n_steps):
            window = np.array(buffer, dtype=np.float64)
            lin_feat = self._extract_linear_delays(window)
            features = self._construct_features_from_linear(lin_feat)

            # Predict next state u(t+1)
            pred_next = self.W_out @ features
            predictions[step] = pred_next

            # Slide window: drop oldest, append newest prediction
            buffer.pop(0)
            buffer.append(pred_next)

        return predictions

    def predict_open_loop(self, u_series: np.ndarray) -> np.ndarray:
        """
        One-step-ahead teacher-forced prediction across an observed trajectory.

        Parameters
        ----------
        u_series : np.ndarray of shape (N, n_inputs)

        Returns
        -------
        predictions : np.ndarray of shape (N - warmup_steps, n_inputs)
        """
        if self.W_out is None:
            raise RuntimeError("Model is not fitted.")

        X, _ = self.compute_feature_matrix(u_series)
        return (self.W_out @ X.T).T

    def feature_names(self, var_names: Optional[List[str]] = None) -> List[str]:
        """Return human-readable names for each feature in the regression library."""
        if var_names is None:
            var_names = [f"u{i}" for i in range(self.n_inputs)]

        # Linear delay names
        lin_names = []
        for d in range(self.n_delays):
            delay_t = d * self.delay_stride
            for v in var_names:
                suffix = "(t)" if delay_t == 0 else f"(t-{delay_t})"
                lin_names.append(f"{v}{suffix}")

        names = []
        if self.include_constant:
            names.append("1")

        names.extend(lin_names)

        # Monomial names
        for combo in self._mono_indices:
            term_str = " * ".join([lin_names[i] for i in combo])
            names.append(term_str)

        return names

    @property
    def total_trainable_parameters(self) -> int:
        """Total trainable parameters in the readout matrix."""
        if self.W_out is None:
            return self.n_inputs * self.n_features
        return self.W_out.size
