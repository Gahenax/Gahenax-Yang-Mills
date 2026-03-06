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

        # Initialize all links to identity (cold start)
        self.links = np.zeros((self.n_sites, dim, 2, 2), dtype=complex)
        for i in range(self.n_sites):
            for mu in range(dim):
                self.links[i, mu] = np.eye(2, dtype=complex)

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
        coords = list(self.site_to_coords(site))
        coords[mu] = (coords[mu] + direction) % self.N
        return self.coords_to_site(tuple(coords))

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
        """
        action = 0.0
        for site in range(self.n_sites):
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    P = self.plaquette(site, mu, nu)
                    action += 1.0 - su2_trace(P)
        return self.beta * action

    def average_plaquette(self) -> float:
        """Average plaquette value (1 = ordered, 0 = disordered)."""
        total = 0.0
        count = 0
        for site in range(self.n_sites):
            for mu in range(self.dim):
                for nu in range(mu + 1, self.dim):
                    P = self.plaquette(site, mu, nu)
                    total += su2_trace(P)
                    count += 1
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
        """Initialize all links to random Haar SU(2) matrices."""
        for i in range(self.n_sites):
            for mu in range(self.dim):
                self.links[i, mu] = haar_su2(self.rng)
