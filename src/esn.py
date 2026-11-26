"""
Classical Echo State Network (ESN) Module.

Pure NumPy implementation of the Echo State Network architecture (Jaeger 2001, Lukosevicius 2012)
with:
- Strict spectral radius scaling of sparse random internal recurrent weights.
- Leaky integrator reservoir dynamics.
- Configurable washout transient removal.
- Quadratic readout feature augmentation (Pathak et al. 2018 / Herteux & Räth 2020)
  for breaking symmetries and capturing nonlinear dynamics in chaotic systems.
- Autonomous (closed-loop / free-running) iterative forecasting.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import linalg


class EchoStateNetwork:
    """
    Classical Echo State Network for chaotic time series prediction.

    Parameters
    ----------
    n_inputs : int
        Dimension of the input state vector.
    n_reservoir : int
        Number of recurrent reservoir neurons (internal states).
    spectral_radius : float
        Target spectral radius rho(W_res) = max(|lambda_i|). Controls the Echo State Property.
    sparsity : float
        Fraction of non-zero connections in W_res (connectivity density, in (0, 1]).
    input_scaling : float
        Uniform scale magnitude for input matrix W_in in [-input_scaling, input_scaling].
    bias_scaling : float
        Uniform scale magnitude for reservoir bias vector b.
    leak_rate : float
        Leaking rate alpha in (0, 1]. alpha = 1.0 yields standard non-leaky reservoir updates.
    quadratic_readout : bool
        If True, augments the feature vector with squared reservoir states [1, r, r^2],
        enabling the linear readout to capture nonlinear quadratic dependencies.
    regularization : float
        L2 Tikhonov ridge regression parameter beta >= 0.
    normalize_inputs : bool
        If True, standardize inputs (zero mean, unit variance) using training statistics
        to prevent tanh saturation and keep autonomous feedback well-conditioned.
    seed : Optional[int]
        Random seed for reproducible weight initialization.
    """

    def __init__(
        self,
        n_inputs: int,
        n_reservoir: int = 500,
        spectral_radius: float = 0.9,
        sparsity: float = 0.1,
        input_scaling: float = 0.25,
        bias_scaling: float = 0.0,
        leak_rate: float = 0.3,
        quadratic_readout: bool = True,
        regularization: float = 1e-3,
        normalize_inputs: bool = True,
        seed: Optional[int] = 42,
    ):
        self.n_inputs = int(n_inputs)
        self.n_reservoir = int(n_reservoir)
        self.spectral_radius = float(spectral_radius)
        self.sparsity = float(sparsity)
        self.input_scaling = float(input_scaling)
        self.bias_scaling = float(bias_scaling)
        self.leak_rate = float(leak_rate)
        self.quadratic_readout = bool(quadratic_readout)
        self.regularization = float(regularization)
        self.normalize_inputs = bool(normalize_inputs)
        self.seed = seed

        self.rng = np.random.default_rng(self.seed)

        # Internal reservoir weights
        self.W_in: np.ndarray = np.empty(0)
        self.W_res: np.ndarray = np.empty(0)
        self.bias: np.ndarray = np.empty(0)

        # Input standardization statistics (fit on training data only)
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None

        # Readout weights: shape (n_inputs, n_features)
        self.W_out: Optional[np.ndarray] = None

        # Feature dimension: [1, r] (1 + N) or [1, r, r^2] (1 + 2*N)
        self.feature_dim = 1 + (2 * self.n_reservoir if self.quadratic_readout else self.n_reservoir)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize W_in, W_res (with spectral radius scaling), and bias."""
        self.W_in = self.rng.uniform(
            -self.input_scaling, self.input_scaling, size=(self.n_reservoir, self.n_inputs)
        )

        if self.bias_scaling > 0.0:
            self.bias = self.rng.uniform(
                -self.bias_scaling, self.bias_scaling, size=self.n_reservoir
            )
        else:
            self.bias = np.zeros(self.n_reservoir, dtype=np.float64)

        # Sparse W_res
        raw_W = self.rng.standard_normal((self.n_reservoir, self.n_reservoir))
        mask = self.rng.uniform(0.0, 1.0, size=(self.n_reservoir, self.n_reservoir)) < self.sparsity
        W_res = raw_W * mask

        # Spectral radius computation & scaling
        eigenvalues = linalg.eigvals(W_res)
        current_sr = np.max(np.abs(eigenvalues))

        if current_sr == 0.0:
            raise ValueError(
                "Generated reservoir matrix has zero spectral radius. Try increasing sparsity or reservoir size."
            )

        self.W_res = W_res * (self.spectral_radius / current_sr)

    def _normalize(self, u: np.ndarray) -> np.ndarray:
        """Apply fitted input standardization if normalize_inputs is enabled."""
        if not self.normalize_inputs or self._input_mean is None or self._input_std is None:
            return u
        return (u - self._input_mean) / self._input_std

    def _denormalize(self, u_norm: np.ndarray) -> np.ndarray:
        """Invert fitted input standardization if normalize_inputs is enabled."""
        if not self.normalize_inputs or self._input_mean is None or self._input_std is None:
            return u_norm
        return u_norm * self._input_std + self._input_mean

    def _feature_vector(self, r: np.ndarray) -> np.ndarray:
        """
        Form augmented readout feature vector from reservoir state r.

        If quadratic_readout is True: [1, r, r^2]
        Otherwise: [1, r]
        """
        if self.quadratic_readout:
            return np.concatenate(([1.0], r, r**2))
        return np.concatenate(([1.0], r))

    def _update_state(self, r: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Single reservoir state step: r(t+1) = (1 - alpha)*r(t) + alpha * tanh(W_res*r + W_in*u + b)."""
        pre_activation = self.W_res @ r + self.W_in @ u + self.bias
        r_new = (1.0 - self.leak_rate) * r + self.leak_rate * np.tanh(pre_activation)
        return r_new

    def fit(self, u_train: np.ndarray, washout: int = 100) -> "EchoStateNetwork":
        """
        Train the readout matrix W_out using Ridge Regression on next-step prediction.

        Parameters
        ----------
        u_train : np.ndarray of shape (N, n_inputs)
            Input trajectory for training.
        washout : int
            Number of initial steps to discard to eliminate transient state effects.

        Returns
        -------
        self : EchoStateNetwork
        """
        u_train = np.asarray(u_train, dtype=np.float64)
        n_samples = len(u_train)
        if n_samples <= washout + 1:
            raise ValueError(
                f"Training sequence length ({n_samples}) must exceed washout ({washout}) + 1."
            )

        if self.normalize_inputs:
            self._input_mean = np.mean(u_train, axis=0)
            std = np.std(u_train, axis=0)
            self._input_std = np.where(std > 0.0, std, 1.0)
        else:
            self._input_mean = None
            self._input_std = None

        u_proc = self._normalize(u_train)
        n_effective = n_samples - 1 - washout

        # Preallocate feature matrix: shape (n_effective, feature_dim)
        features = np.zeros((n_effective, self.feature_dim), dtype=np.float64)

        r = np.zeros(self.n_reservoir, dtype=np.float64)

        # Drive reservoir forward
        for t in range(n_samples - 1):
            r = self._update_state(r, u_proc[t])
            if t >= washout:
                idx = t - washout
                features[idx] = self._feature_vector(r)

        targets = u_proc[washout + 1 :]  # shape (n_effective, n_inputs)

        # Ridge regression: (S^T * S + beta * I) * W_out^T = S^T * targets
        reg_matrix = features.T @ features + self.regularization * np.eye(self.feature_dim)
        rhs = features.T @ targets

        try:
            w_out_T = linalg.solve(reg_matrix, rhs, assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            w_out_T, _, _, _ = linalg.lstsq(reg_matrix, rhs)

        self.W_out = w_out_T.T  # shape (n_inputs, feature_dim)

        return self

    def predict_autonomous(
        self,
        u_warmup: np.ndarray,
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autonomous (closed-loop) iterative prediction.

        Parameters
        ----------
        u_warmup : np.ndarray of shape (L_warm, n_inputs)
            History sequence to synchronize the reservoir state prior to free-running prediction.
        n_steps : int
            Number of future time steps to generate autonomously.

        Returns
        -------
        predictions : np.ndarray of shape (n_steps, n_inputs)
            Autonomously predicted trajectory.
        reservoir_states : np.ndarray of shape (n_steps, n_reservoir)
            Internal reservoir state trajectories during autonomous phase.
        """
        if self.W_out is None:
            raise RuntimeError("Model is not fitted. Call fit() before predicting.")

        u_warmup = self._normalize(np.asarray(u_warmup, dtype=np.float64))

        # Warm up the reservoir
        r = np.zeros(self.n_reservoir, dtype=np.float64)
        for t in range(len(u_warmup)):
            r = self._update_state(r, u_warmup[t])

        # Autonomous loop in normalized input space
        predictions = np.zeros((n_steps, self.n_inputs), dtype=np.float64)
        states = np.zeros((n_steps, self.n_reservoir), dtype=np.float64)

        feat = self._feature_vector(r)
        current_u = self.W_out @ feat

        for step in range(n_steps):
            predictions[step] = self._denormalize(current_u)
            states[step] = r
            r = self._update_state(r, current_u)
            feat = self._feature_vector(r)
            current_u = self.W_out @ feat

        return predictions, states

    def predict_open_loop(
        self,
        u_sequence: np.ndarray,
        initial_reservoir_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """One-step ahead teacher-forced prediction on an observed trajectory."""
        if self.W_out is None:
            raise RuntimeError("Model is not fitted.")

        u_proc = self._normalize(np.asarray(u_sequence, dtype=np.float64))
        N = len(u_proc)
        predictions = np.zeros((N, self.n_inputs), dtype=np.float64)
        r = (
            np.zeros(self.n_reservoir, dtype=np.float64)
            if initial_reservoir_state is None
            else initial_reservoir_state.copy()
        )

        for t in range(N):
            r = self._update_state(r, u_proc[t])
            feat = self._feature_vector(r)
            predictions[t] = self._denormalize(self.W_out @ feat)

        return predictions

    @property
    def total_trainable_parameters(self) -> int:
        """Total trainable parameters in the readout matrix."""
        if self.W_out is None:
            return self.n_inputs * self.feature_dim
        return self.W_out.size
