"""
Tests for Yang-Mills v1.0 module:
- LatticeGauge (SU(2) matrices, plaquette action, thermalization)
- Wilson loops (computation, confinement detection)
- MassGapDetector (classification)
- ExperimentMemory (persistence)
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from lattice_gauge import (
    LatticeGauge, random_su2, haar_su2, su2_dagger, su2_trace
)
from wilson_loop import (
    wilson_loop, average_wilson_loop, static_potential, detect_confinement
)
from experiment_memory import ExperimentMemory


# -- SU(2) Matrix Tests --

def test_su2_unitarity():
    """SU(2) matrices should satisfy U * U^dag = I."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        U = random_su2(rng, epsilon=1.0)
        product = U @ su2_dagger(U)
        assert np.allclose(product, np.eye(2), atol=1e-10), "U * U^dag != I"


def test_su2_determinant():
    """SU(2) matrices should have det = 1."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        U = random_su2(rng, epsilon=1.0)
        det = np.linalg.det(U)
        assert abs(det - 1.0) < 1e-10, f"det(U) = {det}, expected 1"


def test_su2_trace_range():
    """Re(Tr(U))/2 should be in [-1, 1] for SU(2)."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        U = random_su2(rng, epsilon=2.0)
        t = su2_trace(U)
        assert -1.01 <= t <= 1.01, f"trace = {t}, out of range"


def test_haar_su2():
    """Haar measure SU(2) should still be unitary."""
    rng = np.random.default_rng(42)
    U = haar_su2(rng)
    product = U @ su2_dagger(U)
    assert np.allclose(product, np.eye(2), atol=1e-10)


# -- Lattice Tests --

def test_lattice_creation():
    """Lattice should initialize with identity links."""
    lat = LatticeGauge(N=4, dim=2, beta=2.0, seed=0)
    assert lat.n_sites == 16
    assert lat.n_links == 32
    # All links should be identity
    for i in range(lat.n_sites):
        for mu in range(lat.dim):
            assert np.allclose(lat.get_link(i, mu), np.eye(2))


def test_cold_start_plaquette():
    """Cold start (all identity) should give average plaquette = 1.0."""
    lat = LatticeGauge(N=4, dim=2, beta=2.0, seed=0)
    plaq = lat.average_plaquette()
    assert abs(plaq - 1.0) < 1e-10, f"Cold plaquette = {plaq}, expected 1.0"


def test_cold_start_action_zero():
    """Cold start action should be 0 (all plaquettes = identity)."""
    lat = LatticeGauge(N=4, dim=2, beta=2.0, seed=0)
    action = lat.plaquette_action()
    assert abs(action) < 1e-10, f"Cold action = {action}, expected 0"


def test_neighbor_periodic():
    """Neighbor function should wrap around periodically."""
    lat = LatticeGauge(N=4, dim=2, seed=0)
    # Site (3,0) + direction 0 should go to (0,0)
    site = lat.coords_to_site((3, 0))
    neighbor = lat.neighbor(site, 0, +1)
    coords = lat.site_to_coords(neighbor)
    assert coords == (0, 0), f"Expected (0,0), got {coords}"


def test_metropolis_sweep():
    """Metropolis sweep should have non-zero acceptance rate."""
    lat = LatticeGauge(N=4, dim=2, beta=2.0, seed=42)
    acc = lat.metropolis_sweep(n_hits=5, epsilon=0.3)
    assert 0 < acc < 1, f"Acceptance rate = {acc}"


def test_hot_start_plaquette():
    """Hot start should give plaquette significantly less than 1."""
    lat = LatticeGauge(N=4, dim=2, beta=2.0, seed=42)
    lat.hot_start()
    plaq = lat.average_plaquette()
    assert plaq < 0.95, f"Hot plaquette = {plaq}, expected < 0.95"


# -- Wilson Loop Tests --

def test_wilson_loop_identity():
    """Wilson loop on cold start (all identity) should be 1.0."""
    lat = LatticeGauge(N=4, dim=2, beta=2.0, seed=0)
    W = wilson_loop(lat, origin=0, R=2, T=2, mu=0, nu=1)
    assert abs(W - 1.0) < 1e-10, f"Wilson loop = {W}, expected 1.0"


def test_average_wilson_loop_identity():
    """Average Wilson loop on cold start should be 1.0."""
    lat = LatticeGauge(N=4, dim=2, beta=2.0, seed=0)
    W_avg = average_wilson_loop(lat, R=1, T=1)
    assert abs(W_avg - 1.0) < 1e-10


def test_confinement_detection_schema():
    """detect_confinement should return proper schema."""
    potential = [(1, 0.5), (2, 1.1), (3, 1.6)]
    result = detect_confinement(potential, threshold=0.3)
    assert "sigma" in result
    assert "confinement" in result
    assert "verdict" in result
    assert "r_squared" in result


# -- Experiment Memory Tests --

def test_memory_write_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ExperimentMemory(working_dir=tmpdir)
        assert len(mem.load_explored_betas()) == 0

        mem.save_experiment(
            beta=2.3, N=4, dim=4,
            verdict="CONFINED", sigma=0.15,
            plaquette=0.62,
        )
        betas = mem.load_explored_betas()
        assert 2.3 in betas


def test_memory_multiple_betas():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = ExperimentMemory(working_dir=tmpdir)
        mem.save_experiment(beta=1.5, N=4, dim=4, verdict="COULOMB", sigma=0.02, plaquette=0.3)
        mem.save_experiment(beta=2.3, N=4, dim=4, verdict="CONFINED", sigma=0.15, plaquette=0.6)
        mem.save_experiment(beta=4.0, N=4, dim=4, verdict="COULOMB", sigma=0.01, plaquette=0.9)

        betas = mem.load_explored_betas()
        assert len(betas) == 3
        assert betas == sorted(betas)  # Should be sorted
