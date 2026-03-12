"""
Lattice Gauge Field Engine -- SU(2) Yang-Mills on a hypercubic lattice.

Pure Python + numpy implementation for generating gauge field configurations
via Metropolis-Hastings heatbath sweeps.
"""
import numpy as np
from typing import Tuple, Optional


# -- SU(2) matrix utilities --

def random_su2(rng: np.random.Generator, epsilon: float = 0.5) -> np.ndarray:
    """
    Generate a random SU(2) matrix near identity via Cayley parametrization.
    epsilon controls how far from identity (0 = identity, 1 = full Haar).
    """
    # Random vector in the Lie algebra su(2)
    r = rng.normal(size=3) * epsilon
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-15:
        return np.eye(2, dtype=complex)

    # Cayley map: U = (I + i*r.sigma) / (I - i*r.sigma)
    # Equivalent to: U = cos(r_norm)*I + i*sin(r_norm)*(r/r_norm).sigma
    c = np.cos(r_norm)
    s = np.sin(r_norm)
    n = r / r_norm

    # Pauli matrices dot n
    sigma_n = np.array([
        [n[2], n[0] - 1j * n[1]],
        [n[0] + 1j * n[1], -n[2]]
    ], dtype=complex)

    return c * np.eye(2, dtype=complex) + 1j * s * sigma_n


def haar_su2(rng: np.random.Generator) -> np.ndarray:
    """Generate a Haar-random SU(2) matrix."""
    return random_su2(rng, epsilon=np.pi)


def su2_dagger(U: np.ndarray) -> np.ndarray:
    """Hermitian conjugate (dagger) of a matrix."""
    return U.conj().T


def su2_trace(U: np.ndarray) -> float:
    """Real part of trace, normalized: Re(Tr(U)) / 2."""
    return float(np.real(np.trace(U))) / 2.0


# -- Lattice --

class LatticeGauge:
    """
    SU(2) lattice gauge field on an N^dim hypercubic lattice.

    Links are stored as U[site_index, mu] = 2x2 complex matrix.
    Periodic boundary conditions.
    """
    def __init__(self, N: int, dim: int = 4, beta: float = 2.3,
                 seed: int = 42):
        self.N = N
        self.dim = dim
        self.beta = beta
        self.n_sites = N ** dim
        self.n_links = self.n_sites * dim
        self.rng = np.random.default_rng(seed)

        # Initialize all links to identity (cold start) -- vectorized
        self.links = np.zeros((self.n_sites, dim, 2, 2), dtype=complex)
        self.links[:, :, 0, 0] = 1.0
        self.links[:, :, 1, 1] = 1.0

        # Precompute neighbor table for O(1) lookup during sweeps
        self._neighbors = self._build_neighbor_table()

    def _build_neighbor_table(self) -> np.ndarray:
        """Precompute neighbor[site, mu, dir]: dir=0 forward (+1), dir=1 backward (-1)."""
        table = np.empty((self.n_sites, self.dim, 2), dtype=np.intp)
        for site in range(self.n_sites):
            coords = list(self.site_to_coords(site))
            for mu in range(self.dim):
                fwd = coords.copy()
                fwd[mu] = (fwd[mu] + 1) % self.N
                bwd = coords.copy()
                bwd[mu] = (bwd[mu] - 1) % self.N
                table[site, mu, 0] = self.coords_to_site(tuple(fwd))
                table[site, mu, 1] = self.coords_to_site(tuple(bwd))
        return table

    def site_to_coords(self, idx: int) -> Tuple:
        """Convert flat index to lattice coordinates."""
        coords = []
        for _ in range(self.dim):
            coords.append(idx % self.N)
            idx //= self.N
        return tuple(coords)

    def coords_to_site(self, coords: Tuple) -> int:
        """Convert lattice coordinates to flat index (periodic BC)."""
        idx = 0
        factor = 1
        for d in range(self.dim):
            idx += (coords[d] % self.N) * factor
            factor *= self.N
        return idx

    def neighbor(self, site: int, mu: int, direction: int = 1) -> int:
        """Get neighbor site in direction mu (+1 or -1), periodic BC."""
        return int(self._neighbors[site, mu, 0 if direction == 1 else 1])

    def get_link(self, site: int, mu: int) -> np.ndarray:
        """Get U_mu(site)."""
        return self.links[site, mu]

    def set_link(self, site: int, mu: int, U: np.ndarray) -> None:
        """Set U_mu(site) = U."""
        self.links[site, mu] = U

    def plaquette(self, site: int, mu: int, nu: int) -> np.ndarray:
        """
        Compute U_P = U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x).
        """
        x = site
        x_mu = self.neighbor(x, mu)
        x_nu = self.neighbor(x, nu)

        P = (self.get_link(x, mu)
             @ self.get_link(x_mu, nu)
             @ su2_dagger(self.get_link(x_nu, mu))
             @ su2_dagger(self.get_link(x, nu)))
        return P

    def plaquette_action(self) -> float:
        """
        Wilson gauge action: S = beta * sum_{P} (1 - Re Tr(U_P) / 2).
        Vectorized over all lattice sites simultaneously.
        """
        action = 0.0
        for mu in range(self.dim):
            for nu in range(mu + 1, self.dim):
                x_mu = self._neighbors[:, mu, 0]
                x_nu = self._neighbors[:, nu, 0]
                P = (self.links[:, mu]
                     @ self.links[x_mu, nu]
                     @ self.links[x_nu, mu].conj().swapaxes(-1, -2)
                     @ self.links[:, nu].conj().swapaxes(-1, -2))
                traces = np.real(P[:, 0, 0] + P[:, 1, 1]) / 2.0
                action += (1.0 - traces).sum()
        return self.beta * action

    def average_plaquette(self) -> float:
        """Average plaquette value (1 = ordered, 0 = disordered). Vectorized over all sites."""
        total = 0.0
        n_planes = 0
        for mu in range(self.dim):
            for nu in range(mu + 1, self.dim):
                x_mu = self._neighbors[:, mu, 0]
                x_nu = self._neighbors[:, nu, 0]
                P = (self.links[:, mu]
                     @ self.links[x_mu, nu]
                     @ self.links[x_nu, mu].conj().swapaxes(-1, -2)
                     @ self.links[:, nu].conj().swapaxes(-1, -2))
                traces = np.real(P[:, 0, 0] + P[:, 1, 1]) / 2.0
                total += traces.sum()
                n_planes += 1
        count = n_planes * self.n_sites
        return total / count if count > 0 else 0.0

    def staple(self, site: int, mu: int) -> np.ndarray:
        """
        Compute the sum of staples around link U_mu(site).
        Used for Metropolis update.
        """
        S = np.zeros((2, 2), dtype=complex)
        for nu in range(self.dim):
            if nu == mu:
                continue
            # Forward staple
            x_mu = self.neighbor(site, mu)
            x_nu = self.neighbor(site, nu)
            S += (self.get_link(x_mu, nu)
                  @ su2_dagger(self.get_link(x_nu, mu))
                  @ su2_dagger(self.get_link(site, nu)))
            # Backward staple
            x_nub = self.neighbor(site, nu, -1)
            x_mu_nub = self.neighbor(x_nub, mu)
            S += (su2_dagger(self.get_link(x_mu_nub, nu))
                  @ su2_dagger(self.get_link(x_nub, mu))
                  @ self.get_link(x_nub, nu))
        return S

    def metropolis_sweep(self, n_hits: int = 10, epsilon: float = 0.3) -> float:
        """
        One Metropolis-Hastings sweep over all links.
        Returns acceptance rate.
        """
        accepted = 0
        total = 0
        for site in range(self.n_sites):
            for mu in range(self.dim):
                stap = self.staple(site, mu)
                U_old = self.get_link(site, mu).copy()
                s_old = float(np.real(np.trace(U_old @ stap)))

                for _ in range(n_hits):
                    dU = random_su2(self.rng, epsilon)
                    U_new = dU @ U_old
                    s_new = float(np.real(np.trace(U_new @ stap)))
                    delta_s = self.beta * (s_new - s_old)

                    if delta_s > 0 or self.rng.random() < np.exp(delta_s):
                        U_old = U_new
                        s_old = s_new
                        accepted += 1
                    total += 1

                self.set_link(site, mu, U_old)
        return accepted / total if total > 0 else 0.0

    def thermalize(self, n_sweeps: int = 50, epsilon: float = 0.3,
                   verbose: bool = False) -> None:
        """Run n_sweeps Metropolis sweeps for thermalization."""
        for i in range(n_sweeps):
            acc = self.metropolis_sweep(n_hits=10, epsilon=epsilon)
            if verbose and (i + 1) % 10 == 0:
                plaq = self.average_plaquette()
                print(f"  Sweep {i+1}/{n_sweeps}: plaq={plaq:.4f}, acc={acc:.2%}")

    def hot_start(self) -> None:
        """Initialize all links to random Haar SU(2) matrices. Vectorized."""
        n = self.n_sites * self.dim
        r = self.rng.normal(size=(n, 3)) * np.pi
        r_norms = np.linalg.norm(r, axis=1)
        tiny = r_norms < 1e-15
        r_norms_safe = np.where(tiny, 1.0, r_norms)
        c = np.cos(r_norms_safe)
        s = np.sin(r_norms_safe)
        n_hat = r / r_norms_safe[:, np.newaxis]

        mats = np.empty((n, 2, 2), dtype=complex)
        mats[:, 0, 0] = c + 1j * s * n_hat[:, 2]
        mats[:, 0, 1] = 1j * s * (n_hat[:, 0] - 1j * n_hat[:, 1])
        mats[:, 1, 0] = 1j * s * (n_hat[:, 0] + 1j * n_hat[:, 1])
        mats[:, 1, 1] = c - 1j * s * n_hat[:, 2]
        mats[tiny] = np.eye(2, dtype=complex)
        self.links[:] = mats.reshape(self.n_sites, self.dim, 2, 2)
