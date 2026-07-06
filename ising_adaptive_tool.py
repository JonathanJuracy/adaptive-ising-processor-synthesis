# -*- coding: utf-8 -*-
"""
Adaptive Probabilistic Ising Processor Synthesis Tool

This script implements a complete tool-oriented workflow for problems that can be
mapped to the Ising model:

1. input data, problem definition, and structural description;
2. preprocessing and conversion to QUBO/Ising form;
3. automatic extraction of simulation and machine parameters;
4. adaptive selection of the update algorithm;
5. probabilistic Ising simulation;
6. generation of numerical results and optional convergence plots.

The objective is to keep the same theoretical basis used in the article figures,
but to expose the system as a reusable tool where the user can provide a problem
instance instead of only reproducing consolidated plots.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Global defaults
# -----------------------------------------------------------------------------

CPU_POWER_W = 65.0
DEFAULT_ITERATIONS = 500
DEFAULT_SEED = 126
DEFAULT_REPLICAS = 8
SUPPORTED_ALGORITHMS = ["Gibbs", "SA", "SQA", "Cluster"]


# -----------------------------------------------------------------------------
# Data structures used by the synthesis workflow
# -----------------------------------------------------------------------------

@dataclass
class ProblemSpec:
    """Container for the user-level problem description before preprocessing."""

    problem_type: str
    name: str
    data: Dict[str, Any]
    structure: Dict[str, Any] = field(default_factory=dict)
    simulation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IsingInstance:
    """Processed Ising instance generated from the user-level problem."""

    name: str
    problem_type: str
    J: np.ndarray
    h: np.ndarray
    offset: float
    n_pbits: int
    coupling_density: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationParameters:
    """Numerical parameters that control the simulation stage."""

    iterations: int = DEFAULT_ITERATIONS
    seed: int = DEFAULT_SEED
    algorithm: str = "auto"
    replicas: int = DEFAULT_REPLICAS
    beta_gibbs: float = 1.1
    beta_cluster: float = 0.9
    beta_sa_min: float = 0.05
    beta_sa_max: float = 3.0
    sqa_gamma_initial: float = 1.8
    sqa_gamma_final: float = 0.05


@dataclass
class MachineConfig:
    """Abstract machine parameters inferred from the Ising instance."""

    n_pbits: int
    n_couplers: int
    coupling_density: float
    field_density: float
    frustration_ratio: float
    selected_algorithm: str
    selection_reason: str
    estimated_device_power_w: float


@dataclass
class SimulationResult:
    """Result produced by one probabilistic update algorithm."""

    problem: str
    algorithm: str
    n_pbits: int
    iterations: int
    best_energy: float
    convergence_iter: int
    runtime_s: float
    computational_energy_j: float
    estimated_device_energy_j: float
    best_spins: List[int]
    decoded_solution: Dict[str, Any]


# -----------------------------------------------------------------------------
# Low-level Ising and QUBO utilities
# -----------------------------------------------------------------------------

def ising_energy(spins: np.ndarray, J: np.ndarray, h: np.ndarray, offset: float = 0.0) -> float:
    """Computes E(s) = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i + offset."""
    s = spins.astype(float)
    return float(-0.5 * s @ J @ s - h @ s + offset)


def binary_from_spins(spins: np.ndarray) -> np.ndarray:
    """Maps Ising spins in {-1,+1} to binary variables in {0,1}."""
    return ((spins + 1) // 2).astype(int)


def add_qubo_term(Q: np.ndarray, i: int, j: int, value: float) -> None:
    """Adds a coefficient to an upper-triangular QUBO matrix."""
    if i == j:
        Q[i, i] += value
    else:
        a, b = sorted((i, j))
        Q[a, b] += value


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Converts an upper-triangular QUBO energy into the Ising convention used here.

    QUBO convention:
        E(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j + offset

    Ising convention:
        E(s) = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i + offset
        with x_i = (s_i + 1)/2.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("QUBO matrix must be square.")

    n = Q.shape[0]
    J = np.zeros((n, n), dtype=float)
    linear_coeff = np.zeros(n, dtype=float)
    ising_offset = float(offset)

    for i in range(n):
        qii = Q[i, i]
        linear_coeff[i] += qii / 2.0
        ising_offset += qii / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if abs(qij) < 1e-15:
                continue
            J[i, j] += -qij / 4.0
            J[j, i] = J[i, j]
            linear_coeff[i] += qij / 4.0
            linear_coeff[j] += qij / 4.0
            ising_offset += qij / 4.0

    h = -linear_coeff
    np.fill_diagonal(J, 0.0)
    return J, h, ising_offset


def coupling_density(J: np.ndarray) -> float:
    """Returns the fraction of non-zero couplers in the upper triangular matrix."""
    n = J.shape[0]
    if n <= 1:
        return 0.0
    possible = n * (n - 1) / 2.0
    nonzero = np.count_nonzero(np.triu(np.abs(J) > 1e-12, 1))
    return float(nonzero / possible)


def frustration_ratio(J: np.ndarray) -> float:
    """Estimates the sign disorder of the coupling matrix as a simple frustration proxy."""
    values = J[np.triu_indices_from(J, k=1)]
    values = values[np.abs(values) > 1e-12]
    if values.size == 0:
        return 0.0
    positive = np.mean(values > 0.0)
    negative = np.mean(values < 0.0)
    return float(min(positive, negative))


def pbit_update(local_field: float, beta: float, rng: np.random.Generator) -> int:
    """Elementary p-bit update: s = sign(rand[-1,1] + tanh(beta * I))."""
    value = rng.uniform(-1.0, 1.0) + np.tanh(beta * local_field)
    return 1 if value >= 0.0 else -1


# -----------------------------------------------------------------------------
# Input handling
# -----------------------------------------------------------------------------

def load_problem_spec(path: Optional[Path]) -> ProblemSpec:
    """Loads a problem specification from JSON or creates a small default example."""
    if path is None:
        return default_maxcut_problem()

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    problem_type = raw.get("problem_type", raw.get("problem", raw.get("type", "custom_ising")))
    name = raw.get("name", problem_type)
    data = raw.get("data", raw)
    structure = raw.get("structure", {})
    simulation = raw.get("simulation", {})

    return ProblemSpec(
        problem_type=str(problem_type).lower(),
        name=str(name),
        data=data,
        structure=structure,
        simulation=simulation,
    )


def default_maxcut_problem() -> ProblemSpec:
    """Returns a deterministic Max-Cut instance used when no input file is provided."""
    return ProblemSpec(
        problem_type="maxcut",
        name="demo_maxcut_6_nodes",
        data={
            "n_nodes": 6,
            "edges": [
                [0, 1, 1.0], [0, 2, 0.7], [1, 2, 0.9],
                [1, 3, 1.2], [2, 4, 0.8], [3, 4, 1.0],
                [3, 5, 0.6], [4, 5, 1.1], [0, 5, 0.5],
            ],
        },
        structure={"description": "Weighted graph used as a default Max-Cut example."},
        simulation={"iterations": DEFAULT_ITERATIONS, "algorithm": "auto"},
    )


# -----------------------------------------------------------------------------
# Preprocessing: user problem to QUBO/Ising
# -----------------------------------------------------------------------------

def preprocess_problem(spec: ProblemSpec) -> IsingInstance:
    """Dispatches the user problem to the appropriate preprocessing routine."""
    problem_type = spec.problem_type.lower().replace("-", "_")

    if problem_type in {"custom_ising", "ising"}:
        return preprocess_custom_ising(spec)
    if problem_type in {"maxcut", "max_cut"}:
        return preprocess_maxcut(spec)
    if problem_type in {"coloring", "graph_coloring"}:
        return preprocess_graph_coloring(spec)
    if problem_type in {"tsp", "traveling_salesman"}:
        return preprocess_tsp(spec)
    if problem_type in {"max2sat", "sat"}:
        return preprocess_max2sat(spec)
    if problem_type in {"matching", "weighted_matching"}:
        return preprocess_matching(spec)
    if problem_type in {"segmentation", "binary_segmentation"}:
        return preprocess_segmentation(spec)

    raise ValueError(
        f"Unsupported problem type '{spec.problem_type}'. Supported types are: "
        "custom_ising, maxcut, coloring, tsp, max2sat, matching, segmentation."
    )


def build_instance_from_qubo(spec: ProblemSpec, Q: np.ndarray, metadata: Dict[str, Any], offset: float = 0.0) -> IsingInstance:
    """Converts a QUBO matrix into an Ising instance and attaches metadata."""
    J, h, ising_offset = qubo_to_ising(Q, offset)
    return IsingInstance(
        name=spec.name,
        problem_type=spec.problem_type,
        J=J,
        h=h,
        offset=ising_offset,
        n_pbits=J.shape[0],
        coupling_density=coupling_density(J),
        metadata=metadata,
    )


def preprocess_custom_ising(spec: ProblemSpec) -> IsingInstance:
    """Reads a user-provided Ising matrix J and field vector h."""
    J = np.asarray(spec.data["J"], dtype=float)
    h = np.asarray(spec.data.get("h", np.zeros(J.shape[0])), dtype=float)
    offset = float(spec.data.get("offset", 0.0))

    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("Custom Ising matrix J must be square.")
    if h.shape[0] != J.shape[0]:
        raise ValueError("Custom Ising field vector h must have the same size as J.")

    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)

    return IsingInstance(
        name=spec.name,
        problem_type=spec.problem_type,
        J=J,
        h=h,
        offset=offset,
        n_pbits=J.shape[0],
        coupling_density=coupling_density(J),
        metadata={"variables": list(range(J.shape[0])), "mapping": "custom_ising"},
    )


def preprocess_maxcut(spec: ProblemSpec) -> IsingInstance:
    """Builds a QUBO that minimizes the negative weighted cut value."""
    n_nodes = int(spec.data["n_nodes"])
    edges = spec.data.get("edges", [])
    Q = np.zeros((n_nodes, n_nodes), dtype=float)

    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        w = float(edge[2]) if len(edge) > 2 else 1.0
        add_qubo_term(Q, i, i, -w)
        add_qubo_term(Q, j, j, -w)
        add_qubo_term(Q, i, j, 2.0 * w)

    metadata = {
        "variables": {f"x_{i}": i for i in range(n_nodes)},
        "edges": edges,
        "objective": "minimize negative cut value",
    }
    return build_instance_from_qubo(spec, Q, metadata)


def preprocess_graph_coloring(spec: ProblemSpec) -> IsingInstance:
    """Builds a QUBO for graph coloring using one-hot color constraints."""
    n_nodes = int(spec.data["n_nodes"])
    n_colors = int(spec.data["n_colors"])
    edges = spec.data.get("edges", [])
    penalty_one_hot = float(spec.data.get("penalty_one_hot", 4.0))
    penalty_edge = float(spec.data.get("penalty_edge", 2.0))

    n_vars = n_nodes * n_colors
    Q = np.zeros((n_vars, n_vars), dtype=float)

    def var(node: int, color: int) -> int:
        return node * n_colors + color

    for node in range(n_nodes):
        for color in range(n_colors):
            add_qubo_term(Q, var(node, color), var(node, color), -penalty_one_hot)
        for c1 in range(n_colors):
            for c2 in range(c1 + 1, n_colors):
                add_qubo_term(Q, var(node, c1), var(node, c2), 2.0 * penalty_one_hot)

    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        for color in range(n_colors):
            add_qubo_term(Q, var(i, color), var(j, color), penalty_edge)

    metadata = {
        "n_nodes": n_nodes,
        "n_colors": n_colors,
        "edges": edges,
        "variables": {f"x_{node}_{color}": var(node, color) for node in range(n_nodes) for color in range(n_colors)},
        "objective": "satisfy one-hot color assignment and penalize equal colors on adjacent nodes",
    }
    return build_instance_from_qubo(spec, Q, metadata, offset=penalty_one_hot * n_nodes)


def preprocess_tsp(spec: ProblemSpec) -> IsingInstance:
    """Builds a QUBO for a small symmetric TSP instance."""
    distances = np.asarray(spec.data["distance_matrix"], dtype=float)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("TSP distance_matrix must be square.")

    n_cities = distances.shape[0]
    n_vars = n_cities * n_cities
    penalty = float(spec.data.get("penalty", 8.0))
    distance_weight = float(spec.data.get("distance_weight", 1.0))
    Q = np.zeros((n_vars, n_vars), dtype=float)

    def var(city: int, position: int) -> int:
        return city * n_cities + position

    # Each city must appear in exactly one tour position.
    for city in range(n_cities):
        for p in range(n_cities):
            add_qubo_term(Q, var(city, p), var(city, p), -penalty)
        for p1 in range(n_cities):
            for p2 in range(p1 + 1, n_cities):
                add_qubo_term(Q, var(city, p1), var(city, p2), 2.0 * penalty)

    # Each tour position must contain exactly one city.
    for p in range(n_cities):
        for city in range(n_cities):
            add_qubo_term(Q, var(city, p), var(city, p), -penalty)
        for c1 in range(n_cities):
            for c2 in range(c1 + 1, n_cities):
                add_qubo_term(Q, var(c1, p), var(c2, p), 2.0 * penalty)

    # Route length contribution between consecutive positions.
    for p in range(n_cities):
        pn = (p + 1) % n_cities
        for i in range(n_cities):
            for j in range(n_cities):
                if i == j:
                    continue
                add_qubo_term(Q, var(i, p), var(j, pn), distance_weight * distances[i, j])

    metadata = {
        "n_cities": n_cities,
        "distance_matrix": distances.tolist(),
        "variables": {f"x_{city}_{position}": var(city, position) for city in range(n_cities) for position in range(n_cities)},
        "objective": "minimize route length with assignment penalties",
    }
    return build_instance_from_qubo(spec, Q, metadata, offset=2.0 * penalty * n_cities)


def preprocess_max2sat(spec: ProblemSpec) -> IsingInstance:
    """
    Builds a QUBO for weighted MAX-2SAT clauses.

    Clauses must contain one or two literals. Positive literals are encoded as
    positive integers and negated literals as negative integers. Literal 1 refers
    to x_0, literal 2 to x_1, and so on.
    """
    n_vars = int(spec.data["n_vars"])
    clauses = spec.data.get("clauses", [])
    Q = np.zeros((n_vars, n_vars), dtype=float)
    offset = 0.0

    def literal_affine(lit: int) -> Tuple[float, Optional[int], float]:
        # Returns a + b*x_idx for a Boolean literal value.
        idx = abs(int(lit)) - 1
        if idx < 0 or idx >= n_vars:
            raise ValueError(f"Literal {lit} is out of range for {n_vars} variables.")
        if lit > 0:
            return 0.0, idx, 1.0
        return 1.0, idx, -1.0

    def add_linear_constant(a: float, idx: Optional[int], b: float, scale: float) -> None:
        nonlocal offset
        offset += scale * a
        if idx is not None:
            add_qubo_term(Q, idx, idx, scale * b)

    for item in clauses:
        if isinstance(item, dict):
            lits = item["literals"]
            weight = float(item.get("weight", 1.0))
        else:
            lits = item
            weight = 1.0

        if len(lits) == 1:
            a, idx, b = literal_affine(lits[0])
            # Penalty for an unsatisfied unit clause is 1 - literal.
            add_linear_constant(1.0 - a, idx, -b, weight)
        elif len(lits) == 2:
            a1, i1, b1 = literal_affine(lits[0])
            a2, i2, b2 = literal_affine(lits[1])
            # Penalty for an unsatisfied 2-literal clause is (1-l1)*(1-l2).
            c1, d1 = 1.0 - a1, -b1
            c2, d2 = 1.0 - a2, -b2
            offset += weight * c1 * c2
            if i1 is not None:
                add_qubo_term(Q, i1, i1, weight * d1 * c2)
            if i2 is not None:
                add_qubo_term(Q, i2, i2, weight * d2 * c1)
            if i1 is not None and i2 is not None:
                add_qubo_term(Q, i1, i2, weight * d1 * d2)
        else:
            raise ValueError("This compact SAT encoder supports only unit clauses and 2-literal clauses.")

    metadata = {
        "n_vars": n_vars,
        "clauses": clauses,
        "variables": {f"x_{i}": i for i in range(n_vars)},
        "objective": "minimize the weighted number of unsatisfied MAX-2SAT clauses",
    }
    return build_instance_from_qubo(spec, Q, metadata, offset=offset)


def preprocess_matching(spec: ProblemSpec) -> IsingInstance:
    """Builds a QUBO for a weighted matching problem on an undirected graph."""
    n_nodes = int(spec.data["n_nodes"])
    edges = spec.data.get("edges", [])
    penalty = float(spec.data.get("penalty", 4.0))
    n_edges = len(edges)
    Q = np.zeros((n_edges, n_edges), dtype=float)

    edge_nodes: List[Tuple[int, int]] = []
    for e_idx, edge in enumerate(edges):
        i, j = int(edge[0]), int(edge[1])
        weight = float(edge[2]) if len(edge) > 2 else 1.0
        edge_nodes.append((i, j))
        add_qubo_term(Q, e_idx, e_idx, -weight)

    for a in range(n_edges):
        for b in range(a + 1, n_edges):
            if set(edge_nodes[a]) & set(edge_nodes[b]):
                add_qubo_term(Q, a, b, penalty)

    metadata = {
        "n_nodes": n_nodes,
        "edges": edges,
        "variables": {f"edge_{idx}_{edge_nodes[idx][0]}_{edge_nodes[idx][1]}": idx for idx in range(n_edges)},
        "objective": "maximize selected edge weight while penalizing adjacent selected edges",
    }
    return build_instance_from_qubo(spec, Q, metadata)


def preprocess_segmentation(spec: ProblemSpec) -> IsingInstance:
    """Builds a binary pairwise segmentation QUBO using unary and smoothness costs."""
    unary_zero = np.asarray(spec.data["unary_zero"], dtype=float).reshape(-1)
    unary_one = np.asarray(spec.data["unary_one"], dtype=float).reshape(-1)
    if unary_zero.shape != unary_one.shape:
        raise ValueError("unary_zero and unary_one must have the same shape.")

    n_pixels = unary_zero.size
    edges = spec.data.get("edges", [])
    smoothness = float(spec.data.get("smoothness", 1.0))
    Q = np.zeros((n_pixels, n_pixels), dtype=float)
    offset = float(np.sum(unary_zero))

    for i in range(n_pixels):
        add_qubo_term(Q, i, i, unary_one[i] - unary_zero[i])

    for edge in edges:
        i, j = int(edge[0]), int(edge[1])
        w = float(edge[2]) if len(edge) > 2 else 1.0
        # Smoothness penalty: lambda*w*|x_i-x_j| = lambda*w*(x_i+x_j-2*x_i*x_j).
        add_qubo_term(Q, i, i, smoothness * w)
        add_qubo_term(Q, j, j, smoothness * w)
        add_qubo_term(Q, i, j, -2.0 * smoothness * w)

    metadata = {
        "n_pixels": n_pixels,
        "edges": edges,
        "variables": {f"pixel_{i}": i for i in range(n_pixels)},
        "objective": "minimize unary assignment cost plus pairwise smoothness penalty",
    }
    return build_instance_from_qubo(spec, Q, metadata, offset=offset)


# -----------------------------------------------------------------------------
# Machine configuration and adaptive algorithm selection
# -----------------------------------------------------------------------------

def build_simulation_parameters(spec: ProblemSpec, cli_args: argparse.Namespace) -> SimulationParameters:
    """Merges default parameters, JSON parameters, and command-line overrides."""
    params = SimulationParameters()
    for key, value in spec.simulation.items():
        if hasattr(params, key):
            setattr(params, key, value)

    if cli_args.iterations is not None:
        params.iterations = int(cli_args.iterations)
    if cli_args.seed is not None:
        params.seed = int(cli_args.seed)
    if cli_args.algorithm is not None:
        params.algorithm = str(cli_args.algorithm)
    if cli_args.replicas is not None:
        params.replicas = int(cli_args.replicas)

    params.algorithm = normalize_algorithm_name(params.algorithm)
    return params


def normalize_algorithm_name(name: str) -> str:
    """Normalizes user-provided algorithm names while preserving known labels."""
    low = str(name).strip().lower()
    if low in {"auto", "adaptive"}:
        return "auto"
    if low in {"all", "compare"}:
        return "all"
    aliases = {
        "gibbs": "Gibbs",
        "sa": "SA",
        "simulated_annealing": "SA",
        "sqa": "SQA",
        "simulated_quantum_annealing": "SQA",
        "cluster": "Cluster",
    }
    if low not in aliases:
        raise ValueError(f"Unsupported algorithm '{name}'. Use auto, all, Gibbs, SA, SQA, or Cluster.")
    return aliases[low]


def configure_machine(instance: IsingInstance, params: SimulationParameters) -> MachineConfig:
    """Infers abstract machine parameters from the processed Ising instance."""
    J = instance.J
    n = instance.n_pbits
    n_couplers = int(np.count_nonzero(np.triu(np.abs(J) > 1e-12, 1)))
    field_density = float(np.mean(np.abs(instance.h) > 1e-12)) if n > 0 else 0.0
    fr = frustration_ratio(J)

    selected_algorithm, reason = select_algorithm(instance, params, fr)
    estimated_device_power_w = estimate_device_power(n, n_couplers, selected_algorithm)

    return MachineConfig(
        n_pbits=n,
        n_couplers=n_couplers,
        coupling_density=instance.coupling_density,
        field_density=field_density,
        frustration_ratio=fr,
        selected_algorithm=selected_algorithm,
        selection_reason=reason,
        estimated_device_power_w=estimated_device_power_w,
    )


def select_algorithm(instance: IsingInstance, params: SimulationParameters, fr: float) -> Tuple[str, str]:
    """Selects the update method according to size, density, and coupling disorder."""
    if params.algorithm not in {"auto", "all"}:
        return params.algorithm, "The algorithm was explicitly selected by the user."

    n = instance.n_pbits
    density = instance.coupling_density

    if n <= 16 and density <= 0.30 and fr < 0.20:
        return "Gibbs", "Small and weakly frustrated instance; Gibbs sampling is sufficient."
    if n >= 64 and density <= 0.15:
        return "Cluster", "Large sparse structure; grouped updates reduce sequential update overhead."
    if fr >= 0.30 or (n >= 32 and density >= 0.25):
        return "SQA", "High coupling disorder or dense interaction graph; replica-based exploration is preferred."
    return "SA", "Moderate-size instance; annealing provides a robust exploration schedule."


def estimate_device_power(n_pbits: int, n_couplers: int, algorithm: str) -> float:
    """
    Estimates a simple synthetic device-level power value for relative comparison.

    This value is not intended to replace measured hardware power. It provides a
    consistent cost proxy proportional to the number of probabilistic elements,
    couplers, and algorithmic overhead.
    """
    base = 0.0025 * n_pbits + 0.00025 * n_couplers
    overhead = {"Gibbs": 1.00, "SA": 1.10, "SQA": 1.65, "Cluster": 1.05}.get(algorithm, 1.0)
    return float(base * overhead)


# -----------------------------------------------------------------------------
# Algorithm library
# -----------------------------------------------------------------------------

def run_gibbs(instance: IsingInstance, params: SimulationParameters, seed_offset: int = 0) -> Tuple[float, int, List[float], np.ndarray]:
    """Runs sequential Gibbs/p-bit updates over the Ising instance."""
    rng = np.random.default_rng(params.seed + seed_offset)
    n, J, h = instance.n_pbits, instance.J, instance.h
    s = rng.choice([-1, 1], size=n)
    best_energy = ising_energy(s, J, h, instance.offset)
    best_iter = 0
    best_spins = s.copy()
    trace: List[float] = []

    for it in range(1, params.iterations + 1):
        for i in rng.permutation(n):
            field = J[i] @ s + h[i]
            s[i] = pbit_update(field, params.beta_gibbs, rng)
        energy = ising_energy(s, J, h, instance.offset)
        if energy < best_energy:
            best_energy = energy
            best_iter = it
            best_spins = s.copy()
        trace.append(best_energy)

    return best_energy, best_iter, trace, best_spins


def run_sa(instance: IsingInstance, params: SimulationParameters, seed_offset: int = 0) -> Tuple[float, int, List[float], np.ndarray]:
    """Runs simulated annealing using single-spin Metropolis updates."""
    rng = np.random.default_rng(params.seed + seed_offset)
    n, J, h = instance.n_pbits, instance.J, instance.h
    s = rng.choice([-1, 1], size=n)
    best_energy = ising_energy(s, J, h, instance.offset)
    best_iter = 0
    best_spins = s.copy()
    trace: List[float] = []

    for it in range(1, params.iterations + 1):
        progress = (it - 1) / max(1, params.iterations - 1)
        beta = params.beta_sa_min + (params.beta_sa_max - params.beta_sa_min) * progress
        for i in rng.permutation(n):
            delta_e = 2.0 * s[i] * (J[i] @ s + h[i])
            if delta_e <= 0.0 or rng.random() < np.exp(-beta * delta_e):
                s[i] *= -1
        energy = ising_energy(s, J, h, instance.offset)
        if energy < best_energy:
            best_energy = energy
            best_iter = it
            best_spins = s.copy()
        trace.append(best_energy)

    return best_energy, best_iter, trace, best_spins


def run_sqa(instance: IsingInstance, params: SimulationParameters, seed_offset: int = 0) -> Tuple[float, int, List[float], np.ndarray]:
    """Runs a compact replica-based SQA-inspired update schedule."""
    rng = np.random.default_rng(params.seed + seed_offset)
    n, J, h = instance.n_pbits, instance.J, instance.h
    replicas = max(2, int(params.replicas))
    states = rng.choice([-1, 1], size=(replicas, n))
    energies = np.array([ising_energy(states[r], J, h, instance.offset) for r in range(replicas)])
    best_idx = int(np.argmin(energies))
    best_energy = float(energies[best_idx])
    best_spins = states[best_idx].copy()
    best_iter = 0
    trace: List[float] = []

    beta = 1.0
    for it in range(1, params.iterations + 1):
        progress = (it - 1) / max(1, params.iterations - 1)
        gamma = params.sqa_gamma_initial * (1.0 - progress) + params.sqa_gamma_final * progress
        for r in range(replicas):
            left = states[(r - 1) % replicas]
            right = states[(r + 1) % replicas]
            for i in rng.permutation(n):
                classical_field = J[i] @ states[r] + h[i]
                replica_field = gamma * (left[i] + right[i])
                states[r, i] = pbit_update(classical_field + replica_field, beta, rng)
        energies = np.array([ising_energy(states[r], J, h, instance.offset) for r in range(replicas)])
        current_idx = int(np.argmin(energies))
        current_energy = float(energies[current_idx])
        if current_energy < best_energy:
            best_energy = current_energy
            best_iter = it
            best_spins = states[current_idx].copy()
        trace.append(best_energy)

    return best_energy, best_iter, trace, best_spins


def greedy_independent_sets(J: np.ndarray) -> List[List[int]]:
    """Partitions the interaction graph into greedy independent update groups."""
    adjacency = np.abs(J) > 1e-12
    n = J.shape[0]
    colors = [-1] * n

    for node in range(n):
        forbidden = {colors[j] for j in range(n) if adjacency[node, j] and colors[j] >= 0}
        color = 0
        while color in forbidden:
            color += 1
        colors[node] = color

    return [[idx for idx, color in enumerate(colors) if color == group] for group in range(max(colors) + 1)]


def run_cluster(instance: IsingInstance, params: SimulationParameters, seed_offset: int = 0) -> Tuple[float, int, List[float], np.ndarray]:
    """Runs grouped p-bit updates using independent sets of the interaction graph."""
    rng = np.random.default_rng(params.seed + seed_offset)
    n, J, h = instance.n_pbits, instance.J, instance.h
    s = rng.choice([-1, 1], size=n)
    groups = greedy_independent_sets(J)
    best_energy = ising_energy(s, J, h, instance.offset)
    best_iter = 0
    best_spins = s.copy()
    trace: List[float] = []

    for it in range(1, params.iterations + 1):
        for group in groups:
            old_state = s.copy()
            for i in group:
                field = J[i] @ old_state + h[i]
                s[i] = pbit_update(field, params.beta_cluster, rng)
        energy = ising_energy(s, J, h, instance.offset)
        if energy < best_energy:
            best_energy = energy
            best_iter = it
            best_spins = s.copy()
        trace.append(best_energy)

    return best_energy, best_iter, trace, best_spins


RUNNERS: Dict[str, Callable[[IsingInstance, SimulationParameters, int], Tuple[float, int, List[float], np.ndarray]]] = {
    "Gibbs": run_gibbs,
    "SA": run_sa,
    "SQA": run_sqa,
    "Cluster": run_cluster,
}


# -----------------------------------------------------------------------------
# Simulation and solution decoding
# -----------------------------------------------------------------------------

def run_simulation(instance: IsingInstance, params: SimulationParameters, config: MachineConfig) -> Tuple[pd.DataFrame, Dict[str, List[float]], List[SimulationResult]]:
    """Executes the selected algorithm or all algorithms from the internal library."""
    algorithms = SUPPORTED_ALGORITHMS if params.algorithm == "all" else [config.selected_algorithm]
    result_rows: List[Dict[str, Any]] = []
    traces: Dict[str, List[float]] = {}
    results: List[SimulationResult] = []

    for idx, algorithm in enumerate(algorithms):
        t0 = perf_counter()
        best_energy, convergence_iter, trace, best_spins = RUNNERS[algorithm](instance, params, seed_offset=101 * idx)
        runtime_s = perf_counter() - t0
        device_power_w = estimate_device_power(instance.n_pbits, config.n_couplers, algorithm)
        decoded = decode_solution(instance, best_spins)

        result = SimulationResult(
            problem=instance.name,
            algorithm=algorithm,
            n_pbits=instance.n_pbits,
            iterations=params.iterations,
            best_energy=best_energy,
            convergence_iter=convergence_iter,
            runtime_s=runtime_s,
            computational_energy_j=CPU_POWER_W * runtime_s,
            estimated_device_energy_j=device_power_w * runtime_s,
            best_spins=[int(v) for v in best_spins.tolist()],
            decoded_solution=decoded,
        )
        results.append(result)
        traces[algorithm] = trace
        result_rows.append(asdict(result) | {"decoded_solution": json.dumps(decoded)})

    return pd.DataFrame(result_rows), traces, results


def decode_solution(instance: IsingInstance, spins: np.ndarray) -> Dict[str, Any]:
    """Decodes the best spin vector into a problem-level interpretation."""
    x = binary_from_spins(spins)
    ptype = instance.problem_type.lower().replace("-", "_")
    meta = instance.metadata

    if ptype in {"maxcut", "max_cut"}:
        edges = meta.get("edges", [])
        cut_value = 0.0
        for edge in edges:
            i, j = int(edge[0]), int(edge[1])
            w = float(edge[2]) if len(edge) > 2 else 1.0
            if x[i] != x[j]:
                cut_value += w
        return {"partition": x.tolist(), "cut_value": cut_value}

    if ptype in {"coloring", "graph_coloring"}:
        n_nodes = int(meta["n_nodes"])
        n_colors = int(meta["n_colors"])
        colors = []
        for node in range(n_nodes):
            block = x[node * n_colors:(node + 1) * n_colors]
            colors.append(int(np.argmax(block)))
        conflicts = sum(1 for edge in meta.get("edges", []) if colors[int(edge[0])] == colors[int(edge[1])])
        return {"colors": colors, "conflicts": int(conflicts)}

    if ptype in {"tsp", "traveling_salesman"}:
        n_cities = int(meta["n_cities"])
        assignment = x.reshape(n_cities, n_cities)
        route = []
        for position in range(n_cities):
            route.append(int(np.argmax(assignment[:, position])))
        distances = np.asarray(meta["distance_matrix"], dtype=float)
        length = sum(float(distances[route[p], route[(p + 1) % n_cities]]) for p in range(n_cities))
        return {"route": route, "route_length": length, "assignment_matrix": assignment.tolist()}

    if ptype in {"max2sat", "sat"}:
        clauses = meta.get("clauses", [])
        satisfied = 0
        for item in clauses:
            literals = item["literals"] if isinstance(item, dict) else item
            clause_ok = False
            for lit in literals:
                idx = abs(int(lit)) - 1
                value = bool(x[idx])
                clause_ok = clause_ok or (value if lit > 0 else not value)
            satisfied += int(clause_ok)
        return {"assignment": x.tolist(), "satisfied_clauses": satisfied, "total_clauses": len(clauses)}

    if ptype in {"matching", "weighted_matching"}:
        edges = meta.get("edges", [])
        selected_edges = [edges[i] for i, bit in enumerate(x.tolist()) if bit == 1]
        total_weight = sum(float(edge[2]) if len(edge) > 2 else 1.0 for edge in selected_edges)
        return {"selected_edges": selected_edges, "matching_weight": total_weight}

    if ptype in {"segmentation", "binary_segmentation"}:
        return {"labels": x.tolist()}

    return {"binary_variables": x.tolist(), "spins": spins.astype(int).tolist()}


# -----------------------------------------------------------------------------
# Output generation
# -----------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    spec: ProblemSpec,
    instance: IsingInstance,
    params: SimulationParameters,
    config: MachineConfig,
    results_df: pd.DataFrame,
    traces: Dict[str, List[float]],
    results: List[SimulationResult],
    make_plot: bool,
) -> None:
    """Writes all reproducibility files to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "simulation_results.csv", index=False)
    save_matrix(output_dir / "ising_J.csv", instance.J)
    save_vector(output_dir / "ising_h.csv", instance.h)

    trace_df = pd.DataFrame({"iteration": np.arange(1, params.iterations + 1)})
    for algorithm, trace in traces.items():
        trace_df[algorithm] = trace
    trace_df.to_csv(output_dir / "energy_trace.csv", index=False)

    summary = {
        "problem_spec": asdict(spec),
        "simulation_parameters": asdict(params),
        "machine_config": asdict(config),
        "ising_instance": {
            "name": instance.name,
            "problem_type": instance.problem_type,
            "n_pbits": instance.n_pbits,
            "coupling_density": instance.coupling_density,
            "offset": instance.offset,
            "metadata": instance.metadata,
        },
        "results": [asdict(r) for r in results],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if make_plot:
        plot_energy_trace(trace_df, output_dir / "energy_trace.png")


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    """Saves a dense matrix as CSV with numerical precision suitable for reproducibility."""
    pd.DataFrame(matrix).to_csv(path, index=False, header=False, float_format="%.10g")


def save_vector(path: Path, vector: np.ndarray) -> None:
    """Saves a vector as a single-column CSV file."""
    pd.DataFrame({"value": vector}).to_csv(path, index=False, float_format="%.10g")


def plot_energy_trace(trace_df: pd.DataFrame, path: Path) -> None:
    """Plots the best-energy trace of each executed algorithm."""
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for column in trace_df.columns:
        if column == "iteration":
            continue
        ax.plot(trace_df["iteration"], trace_df[column], label=column, linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Ising energy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Defines the command-line interface for the synthesis tool."""
    parser = argparse.ArgumentParser(
        description="Adaptive probabilistic Ising processor synthesis tool."
    )
    parser.add_argument("--input", type=Path, default=None, help="Path to a JSON problem specification.")
    parser.add_argument("--output", type=Path, default=Path("ising_tool_out"), help="Output directory.")
    parser.add_argument("--algorithm", type=str, default=None, help="auto, all, Gibbs, SA, SQA, or Cluster.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of simulation iterations.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--replicas", type=int, default=None, help="Number of replicas used by SQA.")
    parser.add_argument("--no-plot", action="store_true", help="Disable convergence plot generation.")
    return parser.parse_args()


def main() -> None:
    """Runs the complete synthesis workflow from input problem to output files."""
    args = parse_args()
    spec = load_problem_spec(args.input)
    params = build_simulation_parameters(spec, args)

    instance = preprocess_problem(spec)
    config = configure_machine(instance, params)

    if params.algorithm == "all":
        config.selected_algorithm = "all"
        config.selection_reason = "All algorithms were executed for comparison."

    results_df, traces, results = run_simulation(instance, params, config)

    save_outputs(
        output_dir=args.output,
        spec=spec,
        instance=instance,
        params=params,
        config=config,
        results_df=results_df,
        traces=traces,
        results=results,
        make_plot=not args.no_plot,
    )

    print("Adaptive Ising synthesis completed.")
    print(f"Problem: {instance.name} ({instance.problem_type})")
    print(f"p-bits: {config.n_pbits} | couplers: {config.n_couplers} | density: {config.coupling_density:.4f}")
    print(f"Selected algorithm: {config.selected_algorithm}")
    print(f"Reason: {config.selection_reason}")
    print(f"Output directory: {args.output.resolve()}")
    print(results_df[[
        "problem",
        "algorithm",
        "best_energy",
        "convergence_iter",
        "runtime_s",
        "computational_energy_j",
        "estimated_device_energy_j",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
