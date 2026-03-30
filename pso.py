"""
pso.py  –  Particle Swarm Optimization core implementations
Lab 9 | Evolutionary Computing | NMIMS MPSTME

Tasks:
  Task 1 – Minimize  f(x1, x2) = 100*(x1 - x2^2)^2 + (1 - x1)^2
            subject to  -5 <= x1, x2 <= 5
  Task 2 – Compare gbest-PSO vs lbest-PSO (ring topology) on time & convergence
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional


# ---------------------------------------------------------------------------
# Objective function  (Task 1)
# ---------------------------------------------------------------------------

def rosenbrock_constrained(x: np.ndarray) -> float:
    """
    f(x1, x2) = 100*(x1 - x2^2)^2 + (1 - x1)^2
    Global minimum at (1, 1) where f = 0.
    Domain: -5 <= x1, x2 <= 5
    """
    x1, x2 = x[0], x[1]
    return 100.0 * (x1 - x2 ** 2) ** 2 + (1.0 - x1) ** 2


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PSOResult:
    best_position: np.ndarray
    best_fitness: float
    convergence: List[float]          # best fitness per iteration
    all_positions: List[np.ndarray]   # swarm positions per iteration (for viz)
    elapsed_time: float
    n_iterations: int
    n_particles: int
    algorithm: str
    evaluations: int


# ---------------------------------------------------------------------------
# Global-Best PSO  (gbest / star topology)
# ---------------------------------------------------------------------------

def gbest_pso(
    func: Callable,
    bounds: List[Tuple[float, float]],
    n_particles: int = 30,
    n_iterations: int = 200,
    w: float = 0.7,        # inertia weight
    c1: float = 1.5,       # cognitive coefficient
    c2: float = 1.5,       # social coefficient
    seed: Optional[int] = None,
    store_positions: bool = True,
) -> PSOResult:
    """
    Standard gbest PSO.

    Velocity update:
        v_i(t+1) = w * v_i(t)
                 + c1 * r1 * (y_i - x_i(t))     [cognitive]
                 + c2 * r2 * (y_hat - x_i(t))   [social / gbest]

    Position update:
        x_i(t+1) = x_i(t) + v_i(t+1)
    """
    rng = np.random.default_rng(seed)
    ndim = len(bounds)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # ---- Initialise --------------------------------------------------------
    X = rng.uniform(lo, hi, (n_particles, ndim))          # positions
    V = rng.uniform(-(hi - lo), (hi - lo), (n_particles, ndim))  # velocities

    # Personal best
    pbest_pos = X.copy()
    pbest_fit = np.array([func(X[i]) for i in range(n_particles)])

    # Global best
    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    convergence: List[float] = []
    all_positions: List[np.ndarray] = []
    evaluations = n_particles

    start = time.perf_counter()

    for _ in range(n_iterations):
        r1 = rng.random((n_particles, ndim))
        r2 = rng.random((n_particles, ndim))

        # Velocity update
        V = (
            w * V
            + c1 * r1 * (pbest_pos - X)
            + c2 * r2 * (gbest_pos - X)
        )

        # Position update
        X = X + V
        X = np.clip(X, lo, hi)          # enforce bounds

        # Evaluate
        fitness = np.array([func(X[i]) for i in range(n_particles)])
        evaluations += n_particles

        # Update personal bests
        improved = fitness < pbest_fit
        pbest_pos[improved] = X[improved].copy()
        pbest_fit[improved] = fitness[improved]

        # Update global best
        best_idx = np.argmin(pbest_fit)
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit = pbest_fit[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()

        convergence.append(gbest_fit)
        if store_positions:
            all_positions.append(X.copy())

    elapsed = time.perf_counter() - start

    return PSOResult(
        best_position=gbest_pos,
        best_fitness=gbest_fit,
        convergence=convergence,
        all_positions=all_positions,
        elapsed_time=elapsed,
        n_iterations=n_iterations,
        n_particles=n_particles,
        algorithm="gbest PSO",
        evaluations=evaluations,
    )


# ---------------------------------------------------------------------------
# Local-Best PSO  (lbest / ring topology)
# ---------------------------------------------------------------------------

def lbest_pso(
    func: Callable,
    bounds: List[Tuple[float, float]],
    n_particles: int = 30,
    n_iterations: int = 200,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    neighborhood_size: int = 3,   # each particle sees k neighbours on ring
    seed: Optional[int] = None,
    store_positions: bool = True,
) -> PSOResult:
    """
    lbest PSO with ring topology.
    Each particle's social component uses the best among its k nearest
    neighbours (ring indices).
    """
    rng = np.random.default_rng(seed)
    ndim = len(bounds)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    X = rng.uniform(lo, hi, (n_particles, ndim))
    V = rng.uniform(-(hi - lo), (hi - lo), (n_particles, ndim))

    pbest_pos = X.copy()
    pbest_fit = np.array([func(X[i]) for i in range(n_particles)])

    convergence: List[float] = []
    all_positions: List[np.ndarray] = []
    evaluations = n_particles

    # Ring neighbours for each particle
    half = neighborhood_size // 2

    def get_lbest(i: int) -> np.ndarray:
        """Return best position among ring neighbours of particle i."""
        indices = [(i + k) % n_particles for k in range(-half, half + 1)]
        fits = pbest_fit[indices]
        best_local_idx = indices[int(np.argmin(fits))]
        return pbest_pos[best_local_idx]

    start = time.perf_counter()

    for _ in range(n_iterations):
        lbest_positions = np.array([get_lbest(i) for i in range(n_particles)])

        r1 = rng.random((n_particles, ndim))
        r2 = rng.random((n_particles, ndim))

        V = (
            w * V
            + c1 * r1 * (pbest_pos - X)
            + c2 * r2 * (lbest_positions - X)
        )

        X = X + V
        X = np.clip(X, lo, hi)

        fitness = np.array([func(X[i]) for i in range(n_particles)])
        evaluations += n_particles

        improved = fitness < pbest_fit
        pbest_pos[improved] = X[improved].copy()
        pbest_fit[improved] = fitness[improved]

        convergence.append(float(np.min(pbest_fit)))
        if store_positions:
            all_positions.append(X.copy())

    elapsed = time.perf_counter() - start

    best_idx = np.argmin(pbest_fit)

    return PSOResult(
        best_position=pbest_pos[best_idx],
        best_fitness=pbest_fit[best_idx],
        convergence=convergence,
        all_positions=all_positions,
        elapsed_time=elapsed,
        n_iterations=n_iterations,
        n_particles=n_particles,
        algorithm="lbest PSO",
        evaluations=evaluations,
    )


# ---------------------------------------------------------------------------
# Comparison utility  (Task 2)
# ---------------------------------------------------------------------------

def compare_algorithms(
    func: Callable,
    bounds: List[Tuple[float, float]],
    n_particles: int = 30,
    n_iterations: int = 200,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    neighborhood_size: int = 3,
    seed: int = 42,
    n_runs: int = 5,
) -> dict:
    """
    Run both algorithms n_runs times and collect statistics.
    Returns a dict with keys 'gbest' and 'lbest', each containing
    lists of PSOResult objects and aggregate stats.
    """
    results = {"gbest": [], "lbest": []}

    for run in range(n_runs):
        g = gbest_pso(func, bounds, n_particles, n_iterations,
                      w, c1, c2, seed=seed + run, store_positions=False)
        l = lbest_pso(func, bounds, n_particles, n_iterations,
                      w, c1, c2, neighborhood_size, seed=seed + run,
                      store_positions=False)
        results["gbest"].append(g)
        results["lbest"].append(l)

    def stats(runs: List[PSOResult]) -> dict:
        fits = [r.best_fitness for r in runs]
        times = [r.elapsed_time for r in runs]
        return {
            "best_fitness_mean": float(np.mean(fits)),
            "best_fitness_std": float(np.std(fits)),
            "best_fitness_min": float(np.min(fits)),
            "time_mean": float(np.mean(times)),
            "time_std": float(np.std(times)),
            "evaluations": runs[0].evaluations,
        }

    results["gbest_stats"] = stats(results["gbest"])
    results["lbest_stats"] = stats(results["lbest"])
    return results


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BOUNDS = [(-5, 5), (-5, 5)]

    print("=" * 55)
    print("  Task 1 – Minimize Rosenbrock (constrained) via gbest PSO")
    print("=" * 55)
    res = gbest_pso(rosenbrock_constrained, BOUNDS, n_particles=30,
                    n_iterations=200, seed=42)
    print(f"  Best position : x1={res.best_position[0]:.6f}, "
          f"x2={res.best_position[1]:.6f}")
    print(f"  Best fitness  : {res.best_fitness:.8f}")
    print(f"  Time elapsed  : {res.elapsed_time*1000:.2f} ms")
    print(f"  Evaluations   : {res.evaluations}")

    print()
    print("=" * 55)
    print("  Task 2 – Compare gbest vs lbest  (5 runs each)")
    print("=" * 55)
    cmp = compare_algorithms(rosenbrock_constrained, BOUNDS,
                             n_particles=30, n_iterations=200, n_runs=5)
    for name in ("gbest", "lbest"):
        s = cmp[f"{name}_stats"]
        print(f"\n  [{name.upper()} PSO]")
        print(f"    Mean fitness  : {s['best_fitness_mean']:.8f}  "
              f"± {s['best_fitness_std']:.8f}")
        print(f"    Best fitness  : {s['best_fitness_min']:.8f}")
        print(f"    Mean time     : {s['time_mean']*1000:.2f} ms  "
              f"± {s['time_std']*1000:.2f} ms")
        print(f"    Evaluations   : {s['evaluations']}")
