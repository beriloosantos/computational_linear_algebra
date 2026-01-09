# -------------------------------------------------------------------------------
# Name           : kaczmarz_necoara.py
# Author         : Berilo Santos (@beriloosantos)
# Version        : 1.0
# Description    : Kaczmarz and Randomized Block Kaczmarz algorithms (Necoara, 2019)
# Data           : 07-01-2026
# -------------------------------------------------------------------------------

import numpy as np
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


def solve_kaczmarz(A, b, TOL=1.e-6, randomize=False, block_size=None, alpha=1.0, adaptive=False):
    """
    Solves the linear system Ax = b using variants of the Kaczmarz method.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        System matrix.
    b : ndarray, shape (m,)
        Right-hand side vector.
    TOL : float, optional
        Stopping tolerance based on the infinity norm of the residual.
    randomize : bool, optional
        If True and block_size is None, applies the Randomized Kaczmarz method
        with probabilities proportional to the squared row norms.
    block_size : int or None, optional
        If None, uses the classical (or randomized) Kaczmarz method.
        If an integer k is provided, applies the Randomized Block Kaczmarz
        method, selecting k rows uniformly at random at each iteration and
        solving a least-squares subproblem.
    alpha : float, optional
        Relaxation (step size) parameter. The classical Kaczmarz and
        Block Kaczmarz methods correspond to alpha = 1. For consistent
        systems, alpha = 1 yields orthogonal projections, while values
        alpha ≠ 1 may affect convergence speed.
    adaptative : bool, optional
        If True, uses an adaptative step size strategy.

    Returns
    -------
    X : ndarray, shape (n,)
        Approximate solution.
    Xhistory : ndarray
        History of iterates.
    err_history : ndarray
        History of residual norms.
    k : int
        Number of iterations performed.
    """

    def calc_err(X):
        return np.linalg.norm(A @ X - b, ord=2)

    m, n = A.shape
    X = np.zeros(n)
    Xhistory = [X.copy()]
    err_history = [calc_err(X)]
    k = 0

    if randomize and block_size is None:
        p = (np.linalg.norm(A, axis=1) / np.linalg.norm(A))**2

    while True:


        if block_size is not None:
            # Randomized Block Kaczmarz
            block = np.random.choice(m, block_size, replace=False)
            A_B = A[block, :]
            b_B = b[block]

            r = b_B - A_B @ X

            # Solve least squares subproblem
            delta = A_B.T @ np.linalg.solve(A_B @ A_B.T, r)

            if adaptive:
                # Adaptive stepsize (Section 4.2, Necoara)
                G = A_B @ A_B.T
                lambda_max = np.linalg.eigvalsh(G).max()
                alpha_k = 1.0 / lambda_max
            else:
                alpha_k = alpha

            Xnew = X + alpha_k * delta

        else:
            # Classical or Randomized Kaczmarz
            if randomize:
                i = np.random.choice(m, p=p)
            else:
                i = k % m
            ai = A[i, :]
            Xnew = X + alpha * (b[i] - ai @ X) / np.linalg.norm(ai)**2 * ai

        err = calc_err(Xnew)
        Xhistory.append(Xnew.copy())
        err_history.append(err)

        X = Xnew
        k += 1

        if err < TOL:
            break

    return X, np.array(Xhistory), np.array(err_history), k

def plot_trajectory(A, b, Xhistory, title):
    """
    Docstring for plot_trajectory

    :param A: Description
    :param b: Description
    :param Xhistory: Description
    :param title: Description
    """
    m, _ = A.shape

    plt.figure()
    # Hyperplanes
    x = np.linspace(-5, 5, 2)
    for i in range(m):
        y = (b[i] - A[i, 0] * x) / A[i, 1]
        plt.plot(x, y)

    # Trajectory
    plt.plot(Xhistory[:, 0], Xhistory[:, 1], 'k-o', markersize=4)
    plt.title(title)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis('square')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"trajectory_{title.replace(' ', '_')}.png", dpi=300)
    plt.savefig(f"trajectory_{title.replace(' ', '_')}.pdf", dpi=300)

def plot_convergence(err_histories, labels, title=None, filename="convergence_comparison"):
    """
    Docstring for plot_convergence
    
    :param err_histories: Description
    :param labels: Description
    """
    plt.figure(figsize=(15, 6))
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (1, 1))]

    for i, (err, label) in enumerate(zip(err_histories, labels)):
        relative_err = err / err[0]
        plt.semilogy(relative_err, label=label, linestyle=styles[i % len(styles)], linewidth=2)

    plt.xlabel("Iteration ($k$)")
    plt.ylabel(r"Relative error $\frac{\|Ax_k-b\|_2}{\|Ax_0-b\|_2}$")
    if title: plt.title(title)
    plt.grid(True, alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.savefig(f"{filename}.pdf", dpi=300)

if __name__ == "__main__":

    np.random.seed(13)  # Reproducibility

    A = np.array([[-4, 1], [2, 0.5], [3, 1.5], [0, 1]])
    b = np.array([-2, 3, 6, 2.])

    alpha_test = 0.8

    # Classical Kaczmarz
    X_K, Xh_K, err_K, it_K = solve_kaczmarz(A, b)
    plot_trajectory(A, b, Xh_K, 'Classical Kaczmarz')

    # Relaxed Kaczmarz
    X_Ka, Xh_Ka, err_Ka, it_Ka = solve_kaczmarz(A, b, alpha=alpha_test)
    plot_trajectory(A, b, Xh_Ka, f'Relaxed Kaczmarz (alpha = {alpha_test})')

    # Randomized Kaczmarz
    X_RK, Xh_RK, err_RK, it_RK = solve_kaczmarz(A, b, randomize=True)
    plot_trajectory(A, b, Xh_RK, 'Randomized Kaczmarz')

    # Randomized Relaxed Kaczmarz
    X_RKa, Xh_RKa, err_RKa, it_RKa = solve_kaczmarz(A, b, randomize=True, alpha=alpha_test)
    plot_trajectory(A, b, Xh_RKa, f'Relaxed RK (alpha = {alpha_test})')

    # Randomized Block Kaczmarz (Necoara, 2019)
    X_RBK, Xh_RBK, err_RBK, it_RBK = solve_kaczmarz(A, b, block_size=2)
    plot_trajectory(A, b, Xh_RBK, 'Randomized Block Kaczmarz (block size = 2)')

    # Randomized Relaxed Block Kaczmarz
    X_RBKa, Xh_RBKa, err_RBKa, it_RBKa = solve_kaczmarz(A, b, block_size=2, alpha=alpha_test)
    plot_trajectory(A, b, Xh_RBKa, f'Relaxed RBK (block size = 2, alpha = {alpha_test})')

    # Adaptive Relaxed Block Kaczmarz
    X_RBKa_adapt, Xh_RBKa_adapt, err_RBKa_adapt, it_RBKa_adapt = solve_kaczmarz(A, b, block_size=2, adaptive=True)
    plot_trajectory(A, b, Xh_RBKa_adapt, 'Adaptive RBK (block size = 2)')

    # Convergence comparison
    plot_convergence(
        [err_K, err_RK, err_RBK, err_Ka, err_RKa, err_RBKa, err_RBKa_adapt],
        ['K', 'RK', 'RBK', 'Relaxed K', 'Relaxed RK', 'Relaxed RBK', 'Adaptive RBK']
    )

    # Console summary
    print('\n=== Summary ===')
    print(f'Kaczmarz:                iterations = {it_K:3d}, final error = {err_K[-1]:.2e}')
    print(f'Relaxed Kaczmarz:       iterations = {it_Ka:3d}, final error = {err_Ka[-1]:.2e}')
    print(f'Randomized Kaczmarz:     iterations = {it_RK:3d}, final error = {err_RK[-1]:.2e}')
    print(f'Randomized Relaxed Kaczmarz: iterations = {it_RKa:3d}, final error = {err_RKa[-1]:.2e}')
    print(f'Block Randomized (RBK):  iterations = {it_RBK:3d}, final error = {err_RBK[-1]:.2e}')
    print(f'Randomized Relaxed (RBK):iterations = {it_RBKa:3d}, final error = {err_RBKa[-1]:.2e}')
    print(f'Adaptive Relaxed (RBK):  iterations = {it_RBKa_adapt:3d}, final error = {err_RBKa_adapt[-1]:.2e}')
