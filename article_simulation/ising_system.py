# -*- coding: utf-8 -*-
"""
Tool for the Synthesis of Adaptive Probabilistic Processors Based on the Ising Model

Code workflow:

1. formulation of the Ising model and probabilistic dynamics;
2. generation of the final plots from the consolidated results used in the paper.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

CPU_POWER_W = 65.0
N_ITER = 500
SEED = 126
OUT_DIR = Path(__file__).resolve().parent / "figures_out"

PROBLEMS = ["TSP", "Coloring", "SAT", "Matching", "Segmentation", "Max-Cut"]
ALGORITHMS = ["Gibbs", "SA", "SQA", "Cluster"]

PBIT_COUNT = {
    "TSP": 30,
    "Coloring": 30,
    "SAT": 10,
    "Matching": 66,
    "Segmentation": 100,
    "Max-Cut": 15,
}

# Consolidated results used in the figures of the paper.
# energy_ising: lowest Ising energy found.
# power_w and energy_device_j: synthetic metrics inspired by physical cost.
# time_s: measured/recorded execution time in the simulator.
# convergence_iter: approximate convergence iteration.

ARTICLE_RESULTS = [
    # problem,        algorithm, mtj, iter, energy, power, device_energy, time, conv
    ("TSP",          "Gibbs",    30, 500,  -8.19, 0.075, 0.0328,  0.437, 460),
    ("TSP",          "SA",       30, 500,  -8.50, 0.083, 0.0415,  0.503, 447),
    ("TSP",          "SQA",      30, 500,  -8.49, 0.098, 0.2389,  2.450, 437),
    ("TSP",          "Cluster",  30, 500,  -5.81, 0.079, 0.0379,  0.481, 190),

    ("Coloring",     "Gibbs",    30, 500,  -8.44, 0.096, 0.0576,  0.600, 452),
    ("Coloring",     "SA",       30, 500,  -9.38, 0.106, 0.0729,  0.690, 341),
    ("Coloring",     "SQA",      30, 500,  -9.67, 0.125, 0.4193,  3.360, 320),
    ("Coloring",     "Cluster",  30, 500,  -3.75, 0.101, 0.0665,  0.660,  24),

    ("SAT",          "Gibbs",    10, 500,  -2.30, 0.028, 0.0042,  0.150, 367),
    ("SAT",          "SA",       10, 500,  -2.30, 0.031, 0.0053,  0.172, 212),
    ("SAT",          "SQA",      10, 500,  -1.81, 0.036, 0.0306,  0.840, 288),
    ("SAT",          "Cluster",  10, 500,  -2.30, 0.029, 0.0049,  0.165, 477),

    ("Matching",     "Gibbs",    66, 500, -15.10, 0.231, 0.3430,  1.485, 483),
    ("Matching",     "SA",       66, 500, -15.10, 0.254, 0.4339,  1.708, 445),
    ("Matching",     "SQA",      66, 500, -18.15, 0.300, 2.4973,  8.316, 455),
    ("Matching",     "Cluster",  66, 500,  -6.87, 0.243, 0.3962,  1.634, 172),

    ("Segmentation", "Gibbs",   100, 500, -31.29, 0.330, 0.6930,  2.100, 491),
    ("Segmentation", "SA",      100, 500, -32.44, 0.363, 0.8766,  2.415, 465),
    ("Segmentation", "SQA",     100, 500, -35.91, 0.429, 5.0450, 11.760, 427),
    ("Segmentation", "Cluster", 100, 500,  -6.89, 0.347, 0.8004,  2.310,  55),

    ("Max-Cut",      "Gibbs",    15, 500,  -3.57, 0.046, 0.0133,  0.285, 365),
    ("Max-Cut",      "SA",       15, 500,  -3.57, 0.051, 0.0168,  0.328, 484),
    ("Max-Cut",      "SQA",      15, 500,  -3.57, 0.060, 0.0965,  1.596, 431),
    ("Max-Cut",      "Cluster",  15, 500,  -2.59, 0.049, 0.0153,  0.314, 270),
]


# -----------------------------------------------------------------------------
# Theoretical core: Ising, p-bit, and update methods
# -----------------------------------------------------------------------------

@dataclass
class IsingInstance:
    name: str
    J: np.ndarray
    h: np.ndarray
    n_pbits: int
    coupling_density: float


def ising_energy(spins: np.ndarray, J: np.ndarray, h: np.ndarray) -> float:
    """E(s) = - sum_{i<j} J_ij s_i s_j - sum_i h_i s_i."""
    s = spins.astype(float)
    return float(-0.5 * s @ J @ s - h @ s)


def pbit_update(local_field: float, beta: float, rng: np.random.Generator) -> int:
    """Elementary p-bit model: s = sign(rand[-1,1] + tanh(beta*I))."""
    value = rng.uniform(-1.0, 1.0) + np.tanh(beta * local_field)
    return 1 if value >= 0.0 else -1


def build_surrogate_ising(problem: str, seed: int = SEED) -> IsingInstance:

    """
    Builds a small Ising instance suitable for local testing, while keeping
    the density and size consistent with the problem classes discussed in the paper.
    The final figures use the consolidated results and do not depend on this generator.
    """
    rng = np.random.default_rng(seed + 17 * PROBLEMS.index(problem))
    n = PBIT_COUNT[problem]

    density = {
        "TSP": 0.42,
        "Coloring": 0.34,
        "SAT": 0.22,
        "Matching": 0.18,
        "Segmentation": 0.08,
        "Max-Cut": 0.50,
    }[problem]

    mask = rng.random((n, n)) < density
    mask = np.triu(mask, 1)

    weights = rng.normal(0.0, 1.0, size=(n, n))
    if problem in {"Max-Cut", "SAT"}:
        weights = np.sign(weights) * rng.uniform(0.2, 1.0, size=(n, n))
    elif problem in {"Coloring", "Segmentation"}:
        weights = -np.abs(weights)
    else:
        weights = rng.normal(0.0, 0.7, size=(n, n))

    J = mask * weights
    J = J + J.T
    h = rng.normal(0.0, 0.08, size=n)
    np.fill_diagonal(J, 0.0)
    return IsingInstance(problem, J, h, n, density)


def run_gibbs(instance: IsingInstance, n_iter: int, seed: int) -> Tuple[float, int, List[float]]:
    rng = np.random.default_rng(seed)
    n, J, h = instance.n_pbits, instance.J, instance.h
    s = rng.choice([-1, 1], size=n)
    best = ising_energy(s, J, h)
    best_iter = 0
    trace = []

    beta = 1.1
    for it in range(1, n_iter + 1):
        for i in rng.permutation(n):
            field = J[i] @ s + h[i]
            s[i] = pbit_update(field, beta, rng)
        e = ising_energy(s, J, h)
        if e < best:
            best, best_iter = e, it
        trace.append(best)
    return best, best_iter, trace


def run_sa(instance: IsingInstance, n_iter: int, seed: int) -> Tuple[float, int, List[float]]:
    rng = np.random.default_rng(seed)
    n, J, h = instance.n_pbits, instance.J, instance.h
    s = rng.choice([-1, 1], size=n)
    best = ising_energy(s, J, h)
    best_iter = 0
    trace = []

    for it in range(1, n_iter + 1):
        beta = 0.05 + 3.0 * (it - 1) / max(1, n_iter - 1)
        for i in rng.permutation(n):
            old = s[i]
            s[i] = -old
            e_new = ising_energy(s, J, h)
            s[i] = old
            e_old = ising_energy(s, J, h)
            dE = e_new - e_old
            if dE <= 0.0 or rng.random() < np.exp(-beta * dE):
                s[i] = -old
        e = ising_energy(s, J, h)
        if e < best:
            best, best_iter = e, it
        trace.append(best)
    return best, best_iter, trace


def run_sqa(
    instance: IsingInstance,
    n_iter: int,
    seed: int,
    replicas: int = 8,
) -> Tuple[float, int, List[float]]:
    rng = np.random.default_rng(seed)
    n, J, h = instance.n_pbits, instance.J, instance.h
    states = rng.choice([-1, 1], size=(replicas, n))
    energies = np.array([ising_energy(states[r], J, h) for r in range(replicas)])
    best = float(energies.min())
    best_iter = 0
    trace = []

    beta = 1.0
    for it in range(1, n_iter + 1):
        gamma = 1.8 * (it - 1) / max(1, n_iter - 1)
        for r in range(replicas):
            left = states[(r - 1) % replicas]
            right = states[(r + 1) % replicas]
            for i in rng.permutation(n):
                field = J[i] @ states[r] + h[i]
                transverse = gamma * (left[i] + right[i])
                states[r, i] = pbit_update(field + transverse, beta, rng)
        energies = np.array([ising_energy(states[r], J, h) for r in range(replicas)])
        e = float(energies.min())
        if e < best:
            best, best_iter = e, it
        trace.append(best)
    return best, best_iter, trace


def greedy_independent_sets(J: np.ndarray) -> List[List[int]]:
    adjacency = np.abs(J) > 1e-12
    n = J.shape[0]
    colors = [-1] * n
    for node in range(n):
        forbidden = {colors[j] for j in range(n) if adjacency[node, j] and colors[j] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[node] = color
    return [[i for i, c in enumerate(colors) if c == color] for color in range(max(colors) + 1)]


def run_cluster(instance: IsingInstance, n_iter: int, seed: int) -> Tuple[float, int, List[float]]:
    rng = np.random.default_rng(seed)
    n, J, h = instance.n_pbits, instance.J, instance.h
    s = rng.choice([-1, 1], size=n)
    clusters = greedy_independent_sets(J)
    best = ising_energy(s, J, h)
    best_iter = 0
    trace = []

    beta = 0.9
    for it in range(1, n_iter + 1):
        for group in clusters:
            old_s = s.copy()
            for i in group:
                field = J[i] @ old_s + h[i]
                s[i] = pbit_update(field, beta, rng)
        e = ising_energy(s, J, h)
        if e < best:
            best, best_iter = e, it
        trace.append(best)
    return best, best_iter, trace


def smoke_test_theory(seed: int = SEED) -> pd.DataFrame:
    """Small optional test to verify that the dynamics routines execute."""
    runners = {
        "Gibbs": run_gibbs,
        "SA": run_sa,
        "SQA": run_sqa,
        "Cluster": run_cluster,
    }
    rows = []
    for problem in PROBLEMS:
        instance = build_surrogate_ising(problem, seed)
        for alg, fn in runners.items():
            t0 = perf_counter()
            energy, conv, _ = fn(instance, 80, seed + hash(problem + alg) % 10000)
            rows.append({
                "problem": problem,
                "algorithm": alg,
                "energy_demo": energy,
                "convergence_demo": conv,
                "runtime_demo_s": perf_counter() - t0,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Organization of the final data
# -----------------------------------------------------------------------------

def article_dataframe() -> pd.DataFrame:
    columns = [
        "problem",
        "algorithm",
        "n_mtj",
        "iterations",
        "energy_ising",
        "power_w",
        "energy_device_j",
        "time_s",
        "convergence_iter",
    ]
    df = pd.DataFrame(ARTICLE_RESULTS, columns=columns)
    df["problem"] = pd.Categorical(df["problem"], categories=PROBLEMS, ordered=True)
    df["algorithm"] = pd.Categorical(df["algorithm"], categories=ALGORITHMS, ordered=True)
    df["computational_energy_j"] = CPU_POWER_W * df["time_s"]
    return df.sort_values(["problem", "algorithm"]).reset_index(drop=True)


def values_by_algorithm(df: pd.DataFrame, metric: str) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for alg in ALGORITHMS:
        part = df[df["algorithm"] == alg].sort_values("problem")
        out[alg] = part[metric].astype(float).tolist()
    return out


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
        "axes.labelsize": 27,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "legend.fontsize": 24,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def grouped_bar_plot(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
    ylim: Tuple[float, float] | None = None,
    legend_y: float = -0.24,
) -> None:
    values = values_by_algorithm(df, metric)
    x = np.arange(len(PROBLEMS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.7, 5.9))
    for pos, alg in enumerate(ALGORITHMS):
        offset = (pos - 1.5) * width
        ax.bar(x + offset, values[alg], width, label=alg)

    ax.set_ylabel(ylabel, labelpad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(PROBLEMS, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=1.8, alpha=0.32)
    ax.set_axisbelow(True)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.legend(
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        handlelength=2.0,
        columnspacing=1.6,
    )
    fig.subplots_adjust(left=0.14, right=0.995, top=0.98, bottom=0.35)
    save_figure(fig, filename)


def plot_all(df: pd.DataFrame) -> None:
    set_plot_style()

    grouped_bar_plot(
        df,
        metric="computational_energy_j",
        ylabel="Computational Energy (J)",
        filename="computational_energy",
        ylim=(0, 800),
        legend_y=-0.26,
    )

    grouped_bar_plot(
        df,
        metric="energy_ising",
        ylabel="Ising Energy (I)",
        filename="ising_energy",
        ylim=(-38, 0),
        legend_y=-0.26,
    )

    grouped_bar_plot(
        df,
        metric="convergence_iter",
        ylabel="Convergence iteration",
        filename="convergence_iteration",
        ylim=(0, 520),
        legend_y=-0.25,
    )

    grouped_bar_plot(
        df,
        metric="time_s",
        ylabel="Execution time (s)",
        filename="execution_time",
        ylim=(0, 12.5),
        legend_y=-0.27,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = article_dataframe()
    df.to_csv(OUT_DIR / "results.csv", index=False)
    plot_all(df)

    print("Arquivos gerados em:", OUT_DIR.resolve())
    print(df[[
        "problem",
        "algorithm",
        "energy_ising",
        "time_s",
        "computational_energy_j",
        "convergence_iter",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
