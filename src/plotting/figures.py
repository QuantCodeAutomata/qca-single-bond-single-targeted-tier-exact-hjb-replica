"""
Figure generation for OTC bond market-making experiments.

All plots use matplotlib with the Agg backend (no display required).
Figures are saved to the results/ directory.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


def _ensure_results_dir() -> str:
    """Ensure results/ exists and return its absolute path."""
    results_path = os.path.abspath(_RESULTS_DIR)
    os.makedirs(results_path, exist_ok=True)
    return results_path


def plot_quotes_vs_inventory(
    q_grid: np.ndarray,
    quote_dict: Dict[str, np.ndarray],
    title: str,
    filename: str,
    q_range: Optional[List[float]] = None,
) -> None:
    """
    Line plot of ask quote offset versus inventory for multiple methods.

    Parameters
    ----------
    q_grid : np.ndarray   inventory values (million notional)
    quote_dict : dict     label -> array of ask quote offsets (bp)
    title : str
    filename : str        saved to results/{filename}
    q_range : [q_lo, q_hi] optional x-axis limits
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = slice(None)
    if q_range is not None:
        mask = (q_grid >= q_range[0]) & (q_grid <= q_range[1])

    for label, vals in quote_dict.items():
        ax.plot(q_grid[mask], vals[mask], label=label)

    ax.set_xlabel("Inventory (million notional)")
    ax.set_ylabel("Ask quote offset (bp)")
    ax.set_title(title)
    ax.legend()
    if q_range is not None:
        ax.set_xlim(q_range)
    plt.tight_layout()
    out = os.path.join(_ensure_results_dir(), filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_hit_ratio_vs_inventory(
    q_grid: np.ndarray,
    hr_dict: Dict[str, np.ndarray],
    r_star: float,
    title: str,
    filename: str,
    q_range: Optional[List[float]] = None,
) -> None:
    """
    Line plot of instantaneous hit ratio versus inventory.

    Adds a horizontal dashed line at r_star.

    Parameters
    ----------
    q_grid : np.ndarray
    hr_dict : dict   label -> hit-ratio array
    r_star : float   target hit ratio
    title : str
    filename : str
    q_range : optional x-axis limits
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = slice(None)
    if q_range is not None:
        mask = (q_grid >= q_range[0]) & (q_grid <= q_range[1])

    for label, vals in hr_dict.items():
        ax.plot(q_grid[mask], vals[mask], label=label)

    ax.axhline(r_star, color="black", linestyle="--", linewidth=1.0, label=f"r* = {r_star}")
    ax.set_xlabel("Inventory (million notional)")
    ax.set_ylabel("Instantaneous hit ratio")
    ax.set_title(title)
    ax.legend()
    if q_range is not None:
        ax.set_xlim(q_range)
    plt.tight_layout()
    out = os.path.join(_ensure_results_dir(), filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_xi_vs_inventory(
    q_grid: np.ndarray,
    xi_dict: Dict[str, np.ndarray],
    title: str,
    filename: str,
    q_range: Optional[List[float]] = None,
) -> None:
    """
    Line plot of dual variable xi versus inventory.

    Parameters
    ----------
    q_grid : np.ndarray
    xi_dict : dict   label -> xi array
    title : str
    filename : str
    q_range : optional x-axis limits
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = slice(None)
    if q_range is not None:
        mask = (q_grid >= q_range[0]) & (q_grid <= q_range[1])

    for label, vals in xi_dict.items():
        ax.plot(q_grid[mask], np.asarray(vals)[mask], label=label)

    ax.set_xlabel("Inventory (million notional)")
    ax.set_ylabel("Dual variable xi")
    ax.set_title(title)
    ax.legend()
    if q_range is not None:
        ax.set_xlim(q_range)
    plt.tight_layout()
    out = os.path.join(_ensure_results_dir(), filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_metric_vs_kappa(
    kappa_list: List[float],
    metric_dict: Dict[str, List[float]],
    ylabel: str,
    title: str,
    filename: str,
    log_x: bool = True,
) -> None:
    """
    Plot a scalar metric versus kappa for multiple series.

    Parameters
    ----------
    kappa_list : list of kappa values (x-axis)
    metric_dict : dict   label -> list of metric values (same length as kappa_list)
    ylabel : str
    title : str
    filename : str
    log_x : bool   use log scale on x-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, vals in metric_dict.items():
        ax.plot(kappa_list, vals, marker="o", label=label)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("kappa" if log_x else "x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(_ensure_results_dir(), filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_surface_2d(
    q1_grid: np.ndarray,
    q2_grid: np.ndarray,
    Z: np.ndarray,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    filename: str,
) -> None:
    """
    2D heatmap (pcolormesh) of a quantity over two inventory dimensions.

    Parameters
    ----------
    q1_grid : np.ndarray  shape (M,) – rows
    q2_grid : np.ndarray  shape (N,) – columns
    Z : np.ndarray        shape (M, N)
    xlabel : str          label for q2 axis (columns)
    ylabel : str          label for q1 axis (rows)
    zlabel : str          colorbar label
    title : str
    filename : str
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    Q1, Q2 = np.meshgrid(q1_grid, q2_grid, indexing="ij")
    pcm = ax.pcolormesh(Q2, Q1, Z, shading="auto", cmap="RdYlGn_r")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(zlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    out = os.path.join(_ensure_results_dir(), filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
