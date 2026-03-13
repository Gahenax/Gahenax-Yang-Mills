"""
Tests for next-level features:
- statistics.py: bootstrap, jackknife, autocorrelation
- lattice_gauge.py: input validation, overrelaxation, mixed_sweep, adaptive_thermalize
- wilson_loop.py: Polyakov loop, topological charge
- visualization.py: no-crash smoke tests
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statistics import (
    bootstrap_error,
    jackknife_error,
    integrated_autocorrelation_time,
    effective_sample_size,
)
from lattice_gauge import LatticeGauge
from wilson_loop import (
    polyakov_loop,
    average_polyakov_loop,
    topological_charge,
)


# ── statistics.py ──────────────────────────────────────────────────────────────

class TestBootstrap:
    def test_constant_series_zero_error(self):
        """Bootstrap error of a constant series must be 0."""
        samples = np.ones(100)
        mean, err = bootstrap_error(samples, n_bootstrap=200,
                                    rng=np.random.default_rng(0))
        assert mean == pytest.approx(1.0)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_mean_consistent(self):
        """Bootstrap mean should match numpy mean."""
        rng = np.random.default_rng(42)
        samples = rng.normal(size=200)
        mean, _ = bootstrap_error(samples, n_bootstrap=500,
                                   rng=np.random.default_rng(0))
        assert mean == pytest.approx(np.mean(samples), abs=1e-10)

    def test_error_positive(self):
        """Bootstrap std must be non-negative."""
        rng = np.random.default_rng(7)
        samples = rng.normal(size=100)
        _, err = bootstrap_error(samples, n_bootstrap=200,
                                  rng=np.random.default_rng(0))
        assert err >= 0.0


class TestJackknife:
    def test_constant_series_zero_error(self):
        samples = np.ones(50)
        mean, err = jackknife_error(samples)
        assert mean == pytest.approx(1.0)
        assert err == pytest.approx(0.0, abs=1e-10)

    def test_mean_consistent(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(loc=3.0, size=100)
        mean, _ = jackknife_error(samples)
        assert mean == pytest.approx(np.mean(samples), abs=1e-3)


class TestAutocorrelation:
    def test_white_noise_tau_near_half(self):
        """τ_int for iid noise should be ≈ 0.5."""
        rng = np.random.default_rng(0)
        series = rng.normal(size=2000)
        tau = integrated_autocorrelation_time(series)
        assert 0.3 <= tau <= 1.5, f"Expected tau ≈ 0.5 for white noise, got {tau}"

    def test_short_series_returns_half(self):
        """Short series (< 4 samples) returns 0.5 by convention."""
        tau = integrated_autocorrelation_time(np.array([1.0, 2.0]))
        assert tau == pytest.approx(0.5)

    def test_effective_sample_size_le_n(self):
        """N_eff must not exceed N."""
        rng = np.random.default_rng(3)
        series = rng.normal(size=500)
        n_eff = effective_sample_size(series)
        assert 1 <= n_eff <= len(series)


# ── lattice_gauge.py — input validation ───────────────────────────────────────

class TestInputValidation:
    def test_invalid_N_raises(self):
        with pytest.raises(ValueError, match="N must be"):
            LatticeGauge(N=1, dim=2, beta=2.3)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="dim must be"):
            LatticeGauge(N=4, dim=1, beta=2.3)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta must be"):
            LatticeGauge(N=4, dim=2, beta=0.0)

    def test_negative_beta_raises(self):
        with pytest.raises(ValueError, match="beta must be"):
            LatticeGauge(N=4, dim=2, beta=-1.0)

    def test_valid_args_ok(self):
        lat = LatticeGauge(N=2, dim=2, beta=0.1)
        assert lat.n_sites == 4


# ── lattice_gauge.py — overrelaxation ─────────────────────────────────────────

class TestOverrelaxation:
    def test_preserves_action_cold_start(self):
        """Overrelaxation on cold start (all identity) should leave action = 0."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        action_before = lat.plaquette_action()
        lat.overrelaxation_sweep()
        action_after = lat.plaquette_action()
        assert action_before == pytest.approx(action_after, abs=1e-10)

    def test_preserves_action_after_thermalization(self):
        """Overrelaxation after Metropolis thermalization preserves action."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=42)
        lat.thermalize(n_sweeps=10)
        action_before = lat.plaquette_action()
        lat.overrelaxation_sweep()
        action_after = lat.plaquette_action()
        assert action_before == pytest.approx(action_after, rel=1e-6, abs=1e-6)

    def test_mixed_sweep_returns_acceptance(self):
        """mixed_sweep should return a valid acceptance rate in (0, 1]."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=7)
        lat.thermalize(n_sweeps=5)
        acc = lat.mixed_sweep(n_over=2)
        assert 0.0 < acc <= 1.0

    def test_mixed_sweep_gives_valid_plaquettes(self):
        """mixed_sweep should produce physically valid plaquette values in (0, 1)."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        lat.thermalize(n_sweeps=10)
        plaqs = []
        for _ in range(30):
            lat.mixed_sweep(n_over=4)
            plaqs.append(lat.average_plaquette())
        # Plaquette ∈ [-1, 1] by definition; at beta=2.3 average should be > 0
        assert all(-1.0 - 1e-6 <= p <= 1.0 + 1e-6 for p in plaqs)
        assert np.mean(plaqs) > 0.0


class TestAdaptiveThermalize:
    def test_returns_dict_with_required_keys(self):
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        result = lat.adaptive_thermalize(n_sweeps=20)
        assert "final_epsilon" in result
        assert "final_acceptance" in result
        assert "plaquette_history" in result

    def test_history_length_matches_sweeps(self):
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        result = lat.adaptive_thermalize(n_sweeps=30)
        assert len(result["plaquette_history"]) == 30

    def test_epsilon_stays_in_bounds(self):
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        result = lat.adaptive_thermalize(
            n_sweeps=40, epsilon_min=0.05, epsilon_max=1.0)
        assert 0.05 <= result["final_epsilon"] <= 1.0


# ── wilson_loop.py — Polyakov loop ────────────────────────────────────────────

class TestPolyakovLoop:
    def test_cold_start_magnitude_one(self):
        """Cold start (all identity) -> |L(x)| = 1 for every site."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        L = polyakov_loop(lat, origin=0, direction=0)
        assert abs(abs(L) - 1.0) < 1e-10

    def test_magnitude_in_range(self):
        """|L(x)| must always be in [0, 1]."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=42)
        lat.thermalize(n_sweeps=20)
        for site in range(lat.n_sites):
            L = polyakov_loop(lat, site, direction=0)
            assert 0.0 <= abs(L) <= 1.0 + 1e-10

    def test_average_cold_start_is_one(self):
        """Average |<L>| on cold start should be 1."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        avg = average_polyakov_loop(lat, direction=0)
        assert avg == pytest.approx(1.0, abs=1e-10)

    def test_average_in_range(self):
        """After thermalization, average Polyakov loop in [0, 1]."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=1)
        lat.thermalize(n_sweeps=15)
        avg = average_polyakov_loop(lat)
        assert 0.0 <= avg <= 1.0 + 1e-10


# ── wilson_loop.py — topological charge ───────────────────────────────────────

class TestTopologicalCharge:
    def test_cold_start_4d_is_zero(self):
        """Q = 0 for cold start (all identity = trivial topology)."""
        lat = LatticeGauge(N=4, dim=4, beta=2.3, seed=0)
        Q = topological_charge(lat)
        assert Q == pytest.approx(0.0, abs=1e-10)

    def test_returns_nan_for_dim_less_than_4(self):
        """topological_charge returns nan for dim < 4."""
        lat = LatticeGauge(N=4, dim=2, beta=2.3, seed=0)
        Q = topological_charge(lat)
        assert np.isnan(Q)

    def test_finite_after_thermalization(self):
        """Q should be finite (not nan) after thermalization in 4D."""
        lat = LatticeGauge(N=4, dim=4, beta=2.3, seed=0)
        lat.thermalize(n_sweeps=5)
        Q = topological_charge(lat)
        assert np.isfinite(Q)


# ── visualization.py — smoke tests (no crash, no display needed) ──────────────

class TestVisualization:
    def test_plot_static_potential_no_crash(self):
        from visualization import plot_static_potential
        potential = [(1, 0.5), (2, 0.9), (3, 1.3)]
        fit = {"sigma": 0.4, "intercept": 0.1, "r_squared": 0.98,
               "verdict": "CONFINED"}
        plot_static_potential(potential, fit_result=fit,
                              save_path=None, show=False)

    def test_plot_plaquette_history_no_crash(self):
        from visualization import plot_plaquette_history
        history = [0.5 + 0.01 * i for i in range(50)]
        plot_plaquette_history(history, save_path=None, show=False)

    def test_plot_phase_diagram_no_crash(self):
        from visualization import plot_phase_diagram
        scan = [
            {"beta": 1.0, "plaquette": 0.3, "error": 0.02, "verdict": "COULOMB"},
            {"beta": 2.3, "plaquette": 0.65, "error": 0.01, "verdict": "TRANSITIONAL"},
            {"beta": 4.0, "plaquette": 0.92, "error": 0.005, "verdict": "CONFINED"},
        ]
        plot_phase_diagram(scan, save_path=None, show=False)

    def test_plot_empty_potential_no_crash(self):
        from visualization import plot_static_potential
        plot_static_potential([], save_path=None, show=False)
