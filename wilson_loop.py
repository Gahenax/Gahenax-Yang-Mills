"""
Wilson Loop Observable -- Extracts physical quantities from lattice gauge configurations.

Computes rectangular Wilson loops W(R,T), Creutz ratios, and static quark-antiquark potential V(R).
The key physics: V(R) ~ sigma*R (linear = confinement = mass gap exists).
"""
import numpy as np
from typing import List, Tuple
from lattice_gauge import LatticeGauge, su2_dagger, su2_trace


def path_ordered_product(lattice: LatticeGauge, sites: List[int],
                          directions: List[int]) -> np.ndarray:
    """
    Compute the path-ordered product of links along a sequence of (site, direction) pairs.
    directions[i] is the mu index for the link at sites[i].
    """
    U = np.eye(2, dtype=complex)
    for site, mu in zip(sites, directions):
        U = U @ lattice.get_link(site, mu)
    return U


def wilson_loop(lattice: LatticeGauge, origin: int,
                R: int, T: int, mu: int = 0, nu: int = 1) -> float:
    """
    Compute a single rectangular R x T Wilson loop in the (mu, nu) plane
    starting at origin.

    W(R,T) = Tr[ U_mu^R * U_nu^T * (U_mu^dag)^R * (U_nu^dag)^T ] / 2

    Returns Re(Tr(W))/2 (normalized for SU(2)).
    """
    # Build the path: R steps in mu, T steps in nu, R steps in -mu, T steps in -nu
    W = np.eye(2, dtype=complex)
    site = origin

    # Forward R steps in mu
    for _ in range(R):
        W = W @ lattice.get_link(site, mu)
        site = lattice.neighbor(site, mu)

    # Forward T steps in nu
    for _ in range(T):
        W = W @ lattice.get_link(site, nu)
        site = lattice.neighbor(site, nu)

    # Backward R steps in mu (use dagger)
    for _ in range(R):
        site = lattice.neighbor(site, mu, -1)
        W = W @ su2_dagger(lattice.get_link(site, mu))

    # Backward T steps in nu (use dagger)
    for _ in range(T):
        site = lattice.neighbor(site, nu, -1)
        W = W @ su2_dagger(lattice.get_link(site, nu))

    return su2_trace(W)


def average_wilson_loop(lattice: LatticeGauge, R: int, T: int,
                         mu: int = 0, nu: int = 1,
                         n_samples: int = 0) -> float:
    """
    Average Wilson loop W(R,T) over all lattice origins (or n_samples random ones).
    """
    if n_samples <= 0:
        n_samples = lattice.n_sites

    total = 0.0
    origins = range(lattice.n_sites) if n_samples >= lattice.n_sites else \
              lattice.rng.choice(lattice.n_sites, size=n_samples, replace=False)

    for origin in origins:
        total += wilson_loop(lattice, origin, R, T, mu, nu)

    return total / len(list(origins))


def creutz_ratio(W_R_T: float, W_R1_T: float,
                  W_R_T1: float, W_R1_T1: float) -> float:
    """
    Creutz ratio: chi(R,T) = -ln( W(R,T)*W(R-1,T-1) / (W(R,T-1)*W(R-1,T)) )

    Extracts the string tension sigma in the limit of large R,T.
    If chi > 0 and roughly constant, confinement (mass gap) is present.
    """
    if W_R_T1 <= 0 or W_R1_T <= 0:
        return float('nan')
    numerator = W_R_T * W_R1_T1
    denominator = W_R_T1 * W_R1_T
    if denominator <= 0 or numerator <= 0:
        return float('nan')
    return -np.log(numerator / denominator)


def static_potential(lattice: LatticeGauge, R_max: int, T: int = 4,
                      n_samples: int = 0) -> List[Tuple[int, float]]:
    """
    Compute the static quark-antiquark potential V(R) for R = 1..R_max.
    V(R) = -ln( W(R,T) / W(R,T-1) )

    If V(R) ~ sigma*R (linear), confinement is present -> mass gap.
    If V(R) ~ -alpha/R (Coulomb), no confinement -> no mass gap.

    Returns list of (R, V(R)) pairs.
    """
    if T < 2:
        T = 2

    potential = []
    for R in range(1, R_max + 1):
        W_T = average_wilson_loop(lattice, R, T, n_samples=n_samples)
        W_T1 = average_wilson_loop(lattice, R, T - 1, n_samples=n_samples)

        if W_T > 1e-15 and W_T1 > 1e-15:
            V = -np.log(W_T / W_T1)
        else:
            V = float('nan')
        potential.append((R, float(V)))

    return potential


def detect_confinement(potential: List[Tuple[int, float]],
                        threshold: float = 0.5) -> dict:
    """
    Analyze V(R) to detect confinement.
    Fits V(R) = sigma*R + c and checks if sigma > threshold.

    Returns:
        { "sigma": float, "intercept": float, "r_squared": float,
          "confinement": bool, "verdict": str }
    """
    # Filter valid points
    valid = [(R, V) for R, V in potential if np.isfinite(V)]
    if len(valid) < 2:
        return {"sigma": 0.0, "intercept": 0.0, "r_squared": 0.0,
                "confinement": False, "verdict": "INSUFFICIENT_DATA"}

    Rs = np.array([r for r, _ in valid], dtype=float)
    Vs = np.array([v for _, v in valid], dtype=float)

    # Linear fit: V = sigma * R + c
    coeffs = np.polyfit(Rs, Vs, 1)
    sigma = float(coeffs[0])
    intercept = float(coeffs[1])

    # R-squared
    V_pred = np.polyval(coeffs, Rs)
    ss_res = np.sum((Vs - V_pred) ** 2)
    ss_tot = np.sum((Vs - np.mean(Vs)) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    confinement = sigma > threshold and r_sq > 0.5

    if confinement:
        verdict = "CONFINED (mass gap detected)"
    elif sigma > 0:
        verdict = "COULOMB (weak/no mass gap)"
    else:
        verdict = "DECONFINED (no mass gap)"

    return {
        "sigma": round(sigma, 6),
        "intercept": round(intercept, 6),
        "r_squared": round(r_sq, 4),
        "confinement": confinement,
        "verdict": verdict,
    }
