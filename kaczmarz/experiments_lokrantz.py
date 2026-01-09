# -------------------------------------------------------------------------------
# Name           : experiments_lokrantz.py
# Author         : Berilo Santos (@beriloosantos)
# Version        : 1.0
# Description    : Numerical experiments inspired by Lokrantz (Sections 5.1.2,
#                  5.1.3, 5.1.5) to compare Kaczmarz-type methods, including
#                  Randomized and Block variants.
# Date           : 07-01-2026
# -------------------------------------------------------------------------------

from kaczmarz_necoara import *
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import scienceplots
plt.style.use(['science', 'notebook', 'no-latex'])

colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#a98467", "#8fbcbb", "#1d70a3"]
mpl.rcParams.update({
    'lines.linewidth': 2,
    # 'font.family': 'Times New Roman',
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'font.size': 22,
    'axes.prop_cycle': cycler(color=colors)
})

def generate_random_system(m, n, system_type="gaussian", condition_number=1e2, correlation_factor=0.8):
    """
    Generates consistent linear systems with controlled numerical properties
    for benchmarking Kaczmarz-type methods.

    Parameters
    ----------
    m, n : int
        Dimensions of the system.
    system_type : {"gaussian", "correlated", "ill-conditioned"}
        Type of linear system to generate.
    condition_number : float
        Desired condition number (used only for ill-conditioned systems).
    correlation_factor : float in [0, 1]
        Controls row coherence (used only for correlated systems).

    Returns
    -------
    A : ndarray, shape (m, n)
        System matrix.
    b : ndarray, shape (m,)
        Right-hand side vector (consistent system).
    """

    # True solution
    x_true = np.random.randn(n)

    if system_type == "gaussian":
        # Section 5.1.2 — Well-behaved Gaussian system
        A = np.random.randn(m, n)

    elif system_type == "correlated":
        # Section 5.1.3 — Coherent / Correlated system
        # Rows are nearly parallel (high coherence)
        A = np.random.randn(m, n)
        v = np.random.randn(1, n)
        v /= np.linalg.norm(v)

        A = (1 - correlation_factor) * A + correlation_factor * v

        # Normalize rows for numerical stability
        A /= np.linalg.norm(A, axis=1, keepdims=True)

    elif system_type == "ill-conditioned":
        # Section 5.1.5 — Ill-conditioned system via SVD
        G = np.random.randn(m, n)
        # Singular values decay from 1 to 1/condition_number
        sigmas = np.logspace(0, -np.log10(condition_number), n)
        D = np.diag(sigmas)
        A = G @ D

    else:
        raise ValueError("Unknown system_type.")

    # Consistent right-hand side
    b = A @ x_true

    return A, b


def run_experiment(A, b, TOL=1e-6, block_size=10):
    """
    Run Kaczmarz-type solvers on a given linear system and collect performance data.

    The following methods are evaluated:
    - K   : Deterministic (cyclic) Kaczmarz
    - RK  : Randomized Kaczmarz
    - RBK : Randomized Block Kaczmarz (Necoara-type)
    - RBK-adapt : Adaptive Randomized Block Kaczmarz (Necoara-type)

    For each method, the error history, number of iterations, and execution
    time are recorded.

    Parameters
    ----------
    A : ndarray
        System matrix.
    b : ndarray
        Right-hand side vector.
    TOL : float, optional
        Stopping tolerance for the relative residual.
    block_size : int, optional
        Block size used in the RBK method.

    Returns
    -------
    results : dict
        Dictionary mapping method names to tuples:
        (error_history, iteration_count, elapsed_time).
    """

    results = {}

    # Deterministic Kaczmarz
    t0 = time.perf_counter()
    _, _, err, it = solve_kaczmarz(A, b, TOL=TOL)
    results["K"] = (err, it, time.perf_counter() - t0)

    # Randomized Kaczmarz
    t0 = time.perf_counter()
    _, _, err, it = solve_kaczmarz(A, b, TOL=TOL, randomize=True)
    results["RK"] = (err, it, time.perf_counter() - t0)

    # Randomized Relaxed Kaczmarz
    # t0 = time.perf_counter()
    # _, _, err, it = solve_kaczmarz(A, b, TOL=TOL, randomize=True, alpha=0.8)
    # results["RKr"] = (err, it, time.perf_counter() - t0)

    # Randomized Block Kaczmarz
    t0 = time.perf_counter()
    _, _, err, it = solve_kaczmarz(A, b, TOL=TOL, block_size=block_size)
    results["RBK"] = (err, it, time.perf_counter() - t0)

    # Adaptive Randomized Block Kaczmarz
    t0 = time.perf_counter()
    _, _, err, it = solve_kaczmarz(A, b, TOL=TOL, block_size=block_size, adaptive=True)
    results["RBK-adapt"] = (err, it, time.perf_counter() - t0)

    return results


def print_results_table(results, title):
    """
    Print a formatted summary table of the experimental results.

    Parameters
    ----------
    results : dict
        Output of run_experiment().
    title : str
        Title identifying the experiment.
    """

    print(f"\n=== {title} ===")
    print(f"{'Method':<10} {'Iter':>8} {'Time (s)':>12} {'Final err':>14}")

    for method, (err, it, t) in results.items():
        print(f"{method:<10} {it:>8} {t:>12.4e} {err[-1]:>14.2e}")

def block_size_study(A, b, block_sizes, TOL=1e-6):
    """
    Perform a block size study for the Randomized Block Kaczmarz method.

    For each block size, the method records:
    - number of iterations until convergence,
    - total execution time,
    - full convergence history (residual norm per iteration).

    Parameters
    ----------
    A : ndarray
        System matrix.
    b : ndarray
        Right-hand side vector.
    block_sizes : list of int
        Block sizes to be tested.
    TOL : float, optional
        Stopping tolerance.

    Returns
    -------
    its : list of int
        Number of iterations for each block size.
    times : list of float
        Execution times for each block size.
    histories : dict
        Dictionary mapping block size to error history.
    """

    its = []
    times = []
    histories = {}

    for bs in block_sizes:
        t0 = time.perf_counter()

        _, _, err, it = solve_kaczmarz(
            A, b, TOL=TOL, block_size=bs
        )

        times.append(time.perf_counter() - t0)
        its.append(it)
        histories[bs] = err

    return its, times, histories

def plot_block_study(block_sizes, its, times, histories):
    """
    Plot the effect of the block size on convergence and computational cost.

    The figure contains two panels:
    (1) Iteration count and execution time as functions of the block size.
    (2) Convergence curves (residual norm vs iteration) for each block size.

    Parameters
    ----------
    block_sizes : list of int
        Block sizes tested.
    its : list of int
        Iteration counts corresponding to each block size.
    times : list of float
        Execution times corresponding to each block size.
    histories : dict
        Dictionary mapping block size to error history.
    title : str
        Overall title of the figure.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # -------------------------------------------------
    # Left panel: iterations and time vs block size
    # -------------------------------------------------
    ax1.semilogy(block_sizes, its, 'o-', label="Iterations")
    ax1.set_xlabel("Block size")
    ax1.set_ylabel("Iterations")

    ax1_twin = ax1.twinx()
    ax1_twin.semilogy(block_sizes, times, 's--', label="Time")
    ax1_twin.set_ylabel("Time [s]")

    ax1.set_title("Performance metrics")
    ax1.grid(True)

    # Build a combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    # -------------------------------------------------
    # Right panel: convergence histories
    # -------------------------------------------------
    for bs in block_sizes:
        err = histories[bs]
        ax2.semilogy(err, label=f"{bs}")

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Residual")
    ax2.set_title("Convergence behavior")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("block_size_study.png", dpi=300)
    plt.savefig("block_size_study.pdf", dpi=300)
    # plt.show()


if __name__ == "__main__":

    # Fix random seed for reproducibility
    np.random.seed(13)

    m, n = 1000, 500
    block_size = 100

    # -------------------------------------------------------------------------
    # Section 5.1.2: Gaussian systems
    # -------------------------------------------------------------------------
    A, b = generate_random_system(m, n, "gaussian")
    results = run_experiment(A, b, block_size=block_size)

    plot_convergence(
        [results[k][0] for k in results],
        list(results.keys()),
        title="Gaussian system",
        filename=f"convergence_gaussian_{m}x{n}"
    )

    print_results_table(results, "Gaussian system")

    # -------------------------------------------------------------------------
    # Section 5.1.3: Normalized rows
    # -------------------------------------------------------------------------
    A, b = generate_random_system(m, n, "correlated")
    results = run_experiment(A, b, block_size=block_size)

    plot_convergence(
        [results[k][0] for k in results],
        list(results.keys()),
        title="Correlated system",
        filename=f"convergence_correlated_{m}x{n}"
    )

    print_results_table(results, "Correlated system")

    # -------------------------------------------------------------------------
    # Section 5.1.5: Ill-conditioned system
    # -------------------------------------------------------------------------
    A, b = generate_random_system(m, n, "ill-conditioned")
    results = run_experiment(A, b, block_size=block_size)

    plot_convergence(
        [results[k][0] for k in results],
        list(results.keys()),
        title="Ill-conditioned system",
        filename=f"convergence_ill_conditioned_{m}x{n}"
    )

    print_results_table(results, "Ill-conditioned system")

    # -------------------------------------------------------------------------
    # Block size sensitivity study
    # -------------------------------------------------------------------------
    block_sizes = [1, 5, 10, 20, 50]

    A, b = generate_random_system(500, 100, "gaussian")
    its, times, histories = block_size_study(A, b, block_sizes)

    plot_block_study(block_sizes, its, times, histories)
