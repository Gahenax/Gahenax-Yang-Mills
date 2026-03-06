"""
Mass Gap Detector -- Adapted from Chronos-Hodge verdict system.

Classifies lattice gauge configurations based on:
1. Rigidity under coupling constant (beta) deformations
2. Confinement signature from Wilson loops
3. Combined verdict: CONFINED / COULOMB / TRANSITIONAL
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MassGapVerdict:
    beta: float
    average_plaquette: float
    rigidity: float
    sigma: float
    r_squared: float
    confinement: bool
    verdict: str  # "CONFINED" | "COULOMB" | "TRANSITIONAL"
    details: str


class MassGapDetector:
    """
    Detects the mass gap in SU(2) Yang-Mills by combining:
    - Plaquette rigidity under beta variations
    - Wilson loop confinement analysis
    """
    def __init__(self, N: int = 4, dim: int = 4,
                 rigidity_threshold: float = 0.05,
                 sigma_threshold: float = 0.1,
                 seed: int = 42):
        self.N = N
        self.dim = dim
        self.rigidity_threshold = rigidity_threshold
        self.sigma_threshold = sigma_threshold
        self.seed = seed

    def measure_rigidity(self, beta: float, delta_beta: float = 0.1,
                          n_sweeps: int = 30) -> Tuple[float, float, float]:
        """
        Measure how stable the average plaquette is under small beta deformations.
        Low rigidity = observable is insensitive to deformation = structural.
        High rigidity change = phase transition region.

        Returns: (plaq_center, plaq_plus, plaq_minus)
        """
        from lattice_gauge import LatticeGauge

        results = {}
        for label, b in [("center", beta), ("plus", beta + delta_beta),
                         ("minus", beta - delta_beta)]:
            lat = LatticeGauge(self.N, self.dim, beta=b, seed=self.seed)
            lat.thermalize(n_sweeps=n_sweeps)
            results[label] = lat.average_plaquette()

        return results["center"], results["plus"], results["minus"]

    def measure_confinement(self, beta: float, R_max: int = 3,
                             T: int = 3, n_sweeps: int = 30) -> dict:
        """
        Measure the static potential V(R) and detect confinement.
        """
        from lattice_gauge import LatticeGauge
        from wilson_loop import static_potential, detect_confinement

        lat = LatticeGauge(self.N, self.dim, beta=beta, seed=self.seed)
        lat.thermalize(n_sweeps=n_sweeps)
        pot = static_potential(lat, R_max=R_max, T=T)
        return detect_confinement(pot, threshold=self.sigma_threshold)

    def classify(self, beta: float,
                  n_sweeps: int = 30,
                  delta_beta: float = 0.1,
                  R_max: int = 3, T: int = 3) -> MassGapVerdict:
        """
        Full classification of a gauge configuration at given beta.

        Verdict system (adapted from Chronos-Hodge):
        - CONFINED: rigidity low + sigma high -> mass gap exists
        - COULOMB: rigidity low + sigma low -> no mass gap
        - TRANSITIONAL: high rigidity variation -> near phase transition
        """
        # Stage 1: Rigidity
        plaq_c, plaq_p, plaq_m = self.measure_rigidity(
            beta, delta_beta, n_sweeps)
        rigidity = abs(plaq_p - plaq_m) / (2 * delta_beta)

        # Stage 2: Confinement
        conf = self.measure_confinement(beta, R_max, T, n_sweeps)

        # Stage 3: Verdict
        sigma = conf["sigma"]
        confinement = conf["confinement"]
        r_sq = conf["r_squared"]

        if rigidity > self.rigidity_threshold:
            verdict = "TRANSITIONAL"
        elif confinement:
            verdict = "CONFINED"
        else:
            verdict = "COULOMB"

        details = (f"plaq={plaq_c:.4f}, rigidity={rigidity:.4f} "
                   f"(thr={self.rigidity_threshold}), "
                   f"sigma={sigma:.4f} (thr={self.sigma_threshold}), "
                   f"R2={r_sq:.4f}")

        return MassGapVerdict(
            beta=beta,
            average_plaquette=round(plaq_c, 6),
            rigidity=round(rigidity, 6),
            sigma=round(sigma, 6),
            r_squared=round(r_sq, 4),
            confinement=confinement,
            verdict=verdict,
            details=details,
        )

    def scan_phase_diagram(self, beta_range: List[float],
                            n_sweeps: int = 30) -> List[MassGapVerdict]:
        """
        Scan multiple beta values and classify each.
        Returns ordered list of verdicts for phase diagram construction.
        """
        results = []
        for beta in beta_range:
            v = self.classify(beta, n_sweeps=n_sweeps)
            results.append(v)
        return results
