"""
Dynamical Systems Module.

Provides numerical integration of canonical chaotic systems:
- Lorenz-63 system (Lorenz, 1963)
- Rössler system (Rössler, 1976)

Numerical integration is carried out using adaptive Runge-Kutta 4(5) (Dormand-Prince)
via SciPy's solve_ivp with strict tolerance control to ensure high fidelity of chaotic trajectories.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp


class DynamicalSystem(ABC):
    """Abstract base class for continuous-time dynamical systems."""

    def __init__(self, name: str, dim: int, mle: float):
        self.name = name
        self.dim = dim
        self.mle = mle  # Maximum Lyapunov Exponent (natural time units^-1)

    @abstractmethod
    def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
        """Evaluate the vector field dx/dt = f(t, x)."""
        pass

    @property
    def lyapunov_time(self) -> float:
        """Characteristic Lyapunov time scale T_lambda = 1 / lambda_max."""
        if self.mle <= 0:
            raise ValueError("MLE must be positive to define a Lyapunov time.")
        return 1.0 / self.mle

    def to_lyapunov_time(self, t: np.ndarray) -> np.ndarray:
        """Convert natural time coordinates into Lyapunov times (t * lambda_max)."""
        return np.asarray(t) * self.mle

    def integrate(
        self,
        t_span: float,
        dt: float,
        initial_state: Optional[np.ndarray] = None,
        discard_transient: float = 20.0,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numerically integrate the dynamical system using RK45.

        Parameters
        ----------
        t_span : float
            Length of useful time series after discarding initial transient.
        dt : float
            Sampling time interval.
        initial_state : np.ndarray, optional
            Initial state vector (length dim). If None, defaults to system canonical seed.
        discard_transient : float
            Initial time period to discard so the trajectory settles onto the chaotic attractor.
        rtol : float
            Relative solver tolerance.
        atol : float
            Absolute solver tolerance.

        Returns
        -------
        t : np.ndarray of shape (N,)
            Timestamps starting at 0.0.
        trajectory : np.ndarray of shape (N, dim)
            State trajectory samples along the attractor.
        """
        if initial_state is None:
            initial_state = self.default_initial_state()
        else:
            initial_state = np.asarray(initial_state, dtype=np.float64)

        t_total = t_span + discard_transient
        t_eval = np.arange(0.0, t_total + 0.5 * dt, dt)

        sol = solve_ivp(
            fun=self.vector_field,
            t_span=(0.0, t_total),
            y0=initial_state,
            method="RK45",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        # Discard transient
        mask = sol.t >= discard_transient
        t_post = sol.t[mask] - discard_transient
        traj_post = sol.y.T[mask]

        return t_post, traj_post

    @abstractmethod
    def default_initial_state(self) -> np.ndarray:
        """Return a canonical initial state for the system."""
        pass


class Lorenz63(DynamicalSystem):
    """
    Lorenz-63 Chaotic System.

    Equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Standard chaotic parameters:
        sigma = 10.0, rho = 28.0, beta = 8/3.
    Maximum Lyapunov Exponent (MLE) lambda_max approx 0.9056.
    Characteristic Lyapunov time T_lambda approx 1.104.
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        mle: float = 0.9056,
    ):
        super().__init__(name="Lorenz-63", dim=3, mle=mle)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.beta = float(beta)

    def default_initial_state(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)

    def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz], dtype=np.float64)

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix df/dx evaluated at state."""
        x, y, z = state
        return np.array(
            [
                [-self.sigma, self.sigma, 0.0],
                [self.rho - z, -1.0, -x],
                [y, x, -self.beta],
            ],
            dtype=np.float64,
        )


class Rossler(DynamicalSystem):
    """
    Rössler Chaotic System.

    Equations:
        dx/dt = -y - z
        dy/dt = x + a * y
        dz/dt = b + z * (x - c)

    Standard chaotic parameters:
        a = 0.2, b = 0.2, c = 5.7.
    Maximum Lyapunov Exponent (MLE) lambda_max approx 0.0714.
    Characteristic Lyapunov time T_lambda approx 14.0.
    """

    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        mle: float = 0.0714,
    ):
        super().__init__(name="Rössler", dim=3, mle=mle)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def default_initial_state(self) -> np.ndarray:
        return np.array([0.1, 0.0, 0.1], dtype=np.float64)

    def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz], dtype=np.float64)

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian matrix df/dx evaluated at state."""
        x, y, z = state
        return np.array(
            [
                [0.0, -1.0, -1.0],
                [1.0, self.a, 0.0],
                [z, 0.0, x - self.c],
            ],
            dtype=np.float64,
        )
