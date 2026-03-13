"""
Visualization utilities for Yang-Mills lattice simulations.

Generates publication-ready plots of:
- Static quark-antiquark potential V(R) with linear fit
- Plaquette evolution during thermalization
- Phase diagram <P> vs beta

Saves to PNG automatically when no display is available.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


def _get_matplotlib():
    """Import matplotlib, switching to Agg backend if no display is available."""
    import matplotlib
    if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_static_potential(
    potential_data: List[Tuple[int, float]],
    fit_result: Optional[dict] = None,
    title: str = "Static Quark-Antiquark Potential",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Scatter plot of V(R) vs R with optional linear fit V = σR + c.

    Parameters
    ----------
    potential_data : list of (R, V(R)) from static_potential()
    fit_result     : dict from detect_confinement() with sigma, intercept,
                     r_squared, verdict
    title          : plot title
    save_path      : if given, save PNG to this path
    show           : whether to call plt.show()
    """
    plt = _get_matplotlib()

    valid = [(R, V) for R, V in potential_data if np.isfinite(V)]
    if not valid:
        return

    Rs = np.array([r for r, _ in valid])
    Vs = np.array([v for _, v in valid])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Rs, Vs, color="steelblue", s=80, zorder=5, label="V(R)")

    if fit_result is not None:
        sigma = fit_result.get("sigma", 0.0)
        intercept = fit_result.get("intercept", 0.0)
        r_sq = fit_result.get("r_squared", 0.0)
        verdict = fit_result.get("verdict", "")
        R_fit = np.linspace(Rs.min(), Rs.max(), 100)
        V_fit = sigma * R_fit + intercept
        ax.plot(R_fit, V_fit, "r--",
                label=f"σR+c  (σ={sigma:.3f}, R²={r_sq:.3f})")
        ax.set_title(f"{title}\n{verdict}", fontsize=12)
    else:
        ax.set_title(title, fontsize=12)

    ax.set_xlabel("R (lattice units)", fontsize=12)
    ax.set_ylabel("V(R)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


def plot_plaquette_history(
    history: List[float],
    title: str = "Plaquette History During Thermalization",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot <P> vs sweep number to visualize thermalization convergence.

    Parameters
    ----------
    history   : list of average_plaquette() values, one per sweep
    title     : plot title
    save_path : if given, save PNG to this path
    show      : whether to call plt.show()
    """
    plt = _get_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 4))
    sweeps = np.arange(1, len(history) + 1)
    ax.plot(sweeps, history, color="darkorange", linewidth=1.5)
    ax.axhline(np.mean(history[-len(history) // 4:]), color="gray",
               linestyle="--", alpha=0.7, label="last-quarter mean")
    ax.set_xlabel("Sweep", fontsize=12)
    ax.set_ylabel("<P>", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


def plot_phase_diagram(
    scan_results: List[dict],
    title: str = "Yang-Mills Phase Diagram",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot <P> vs β (phase diagram) with error bars and verdict coloring.

    Parameters
    ----------
    scan_results : list of dicts, each with keys:
                   "beta" (float), "plaquette" (float),
                   "error" (float, optional), "verdict" (str, optional)
    title        : plot title
    save_path    : if given, save PNG to this path
    show         : whether to call plt.show()
    """
    plt = _get_matplotlib()

    COLOR_MAP = {
        "CONFINED": "steelblue",
        "COULOMB": "tomato",
        "TRANSITIONAL": "goldenrod",
    }

    betas = np.array([r["beta"] for r in scan_results])
    plaqs = np.array([r["plaquette"] for r in scan_results])
    errors = np.array([r.get("error", 0.0) for r in scan_results])
    verdicts = [r.get("verdict", "UNKNOWN") for r in scan_results]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by verdict for legend
    plotted_verdicts = set()
    for beta, plaq, err, verdict in zip(betas, plaqs, errors, verdicts):
        color = COLOR_MAP.get(verdict, "gray")
        label = verdict if verdict not in plotted_verdicts else None
        ax.errorbar(beta, plaq, yerr=err if err > 0 else None,
                    fmt="o", color=color, markersize=7,
                    capsize=4, label=label)
        plotted_verdicts.add(verdict)

    ax.set_xlabel("β", fontsize=13)
    ax.set_ylabel("<P>", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Phase")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    if show and os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)
