"""
Unit & Integration Tests for the Reservoir Computing Benchmark Codebase.
Runs with standard library unittest.
"""

import os
import sys
import unittest
import numpy as np

# Ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.systems import Lorenz63, Rossler
from src.esn import EchoStateNetwork
from src.ngrc import NGRC
from src.metrics import (
    compute_normalized_error,
    compute_vpt,
    compute_psd,
    compute_climate_metrics,
    compute_summary_benchmark,
)


class TestBenchmarkSuite(unittest.TestCase):

    def test_lorenz63_integration(self):
        sys_l63 = Lorenz63()
        t, traj = sys_l63.integrate(t_span=5.0, dt=0.02, discard_transient=5.0)
        self.assertEqual(len(t), len(traj))
        self.assertEqual(traj.shape[1], 3)
        self.assertGreater(sys_l63.mle, 0.9)
        self.assertGreater(sys_l63.lyapunov_time, 1.0)

    def test_rossler_integration(self):
        sys_ros = Rossler()
        t, traj = sys_ros.integrate(t_span=5.0, dt=0.05, discard_transient=5.0)
        self.assertEqual(len(t), len(traj))
        self.assertEqual(traj.shape[1], 3)
        self.assertGreater(sys_ros.mle, 0.05)

    def test_esn_fit_and_predict(self):
        sys_l63 = Lorenz63()
        _, traj = sys_l63.integrate(t_span=20.0, dt=0.02, discard_transient=5.0)
        u_train = traj[:700]

        esn = EchoStateNetwork(
            n_inputs=3,
            n_reservoir=50,
            spectral_radius=0.9,
            quadratic_readout=True,
            regularization=1e-3,
            normalize_inputs=True,
            seed=123,
        )
        self.assertEqual(esn.feature_dim, 1 + 2 * 50)
        esn.fit(u_train, washout=50)
        self.assertIsNotNone(esn._input_mean)
        self.assertIsNotNone(esn._input_std)

        pred, states = esn.predict_autonomous(u_warmup=u_train[-50:], n_steps=20)
        self.assertEqual(pred.shape, (20, 3))
        self.assertEqual(states.shape, (20, 50))
        self.assertFalse(np.isnan(pred).any())

        open_pred = esn.predict_open_loop(u_train[:30])
        self.assertEqual(open_pred.shape, (30, 3))
        self.assertFalse(np.isnan(open_pred).any())

    def test_ngrc_fit_and_predict(self):
        sys_l63 = Lorenz63()
        _, traj = sys_l63.integrate(t_span=20.0, dt=0.02, discard_transient=5.0)
        u_train = traj[:700]

        ngrc = NGRC(n_inputs=3, n_delays=2, delay_stride=1, poly_degree=2, regularization=1e-6)
        self.assertEqual(ngrc.n_features, 28)
        ngrc.fit(u_train)

        pred = ngrc.predict_autonomous(u_warmup=u_train[-ngrc.warmup_steps:], n_steps=30)
        self.assertEqual(pred.shape, (30, 3))
        self.assertFalse(np.isnan(pred).any())

    def test_metrics_computation(self):
        y_true = np.ones((100, 3))
        y_pred = np.ones((100, 3))

        err = compute_normalized_error(y_true, y_pred)
        self.assertTrue(np.allclose(err, 0.0))

        vpt_lyap, vpt_time, vpt_step = compute_vpt(y_true, y_pred, dt=0.02, mle=0.9, threshold=0.4)
        self.assertEqual(vpt_step, 100)

        clim = compute_climate_metrics(y_true, y_pred)
        self.assertTrue(np.isclose(clim["mean_error"], 0.0))

        f, psd = compute_psd(y_true, dt=0.02)
        self.assertEqual(len(f), len(psd))
        self.assertEqual(psd.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
