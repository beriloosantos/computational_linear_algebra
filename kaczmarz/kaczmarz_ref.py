# -------------------------------------------------------------------------------
# Name           : kaczmarz_ref.py
# Author         : Scipython / Christian Hill - adapted by Berilo Santos (@beriloosantos)
# Version        : 1.0
# Description    : Code avaiable on the reference link - https://scipython.com/blog/visualizing-kaczmarzs-algorithm/
# Data           : 18-03-2022
# -------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# The linear equation system to solve: AX = b.
A = np.array([[-4, 1], [2, 0.5], [3, 1.5], [0, 1]])
b = np.array([-2, 3, 6, 2.])
m, n = A.shape

def solve_kaczmarz(A, b, TOL=1.e-6, randomize=False):

    def calc_err(X):
        return np.max(np.abs(A @ X - b))

    X = np.zeros(n)
    Xhistory = [X]
    err_history = [calc_err(X)]
    k = 0
    if randomize:
        p = (np.linalg.norm(A, axis=1) / np.linalg.norm(A))**2
    else:
        p = range(m)
    print(p)
    while True:
        if randomize:
            i = np.random.choice(range(m), p=p)
        else:
            i = k % m
        ai = A[i,:]
        Xnew = X + (b[i] - ai @ X) / np.linalg.norm(ai)**2 * ai
        err = np.max(np.abs(A @ Xnew - b))
        Xhistory.append(Xnew)
        err_history.append(err)
        if err < TOL:
            break
        X = Xnew
        k += 1
    print (f'\nRandomized Kaczmarz converged in {k} iterations.' if randomize else f'\nKaczmarz converged in {k} iterations.')
    return X, np.array(Xhistory), np.array(err_history)

def plot_trajectory(Xhistory):
    # Draw the problem array hyperplanes, which for n = 2 are lines.
    x = np.linspace(-5, 5, 2)
    M = np.vstack((x, np.ones(2))).T
    for i in range(m):
        y = M @ (-A[i,0], b[i]) / A[i,1]
        plt.plot(x, y)

    # Plot the trajectory of the Kaczmarz solution algorithm.
    plt.plot(Xhistory[:,0], Xhistory[:,1], 'k-o')
    # Set the Axes aspect ratio so that right angles look correct.
    plt.axis('square')
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 2.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

X, Xhistory, _ = solve_kaczmarz(A, b, 1.e-2, randomize=False)
plot_trajectory(Xhistory)
print(f'Solution vector: {X}.')
X, Xhistory, _ = solve_kaczmarz(A, b, 1.e-2, randomize=True)
plot_trajectory(Xhistory)
print(f'Solution vector: {X}.')