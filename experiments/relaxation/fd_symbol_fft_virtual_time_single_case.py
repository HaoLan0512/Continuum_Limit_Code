"""Solve one relaxation case with finite-difference symbols applied by FFT.

Method:
    Second- or fourth-order finite-difference Laplacian symbols with Picard or
    semi-implicit virtual-time iteration in Fourier space.
Purpose:
    Provide both solvers and run one virtual-time demonstration.
Inputs:
    Eta, grid size, step or mixing factor, tolerance, iteration limit, and order.
Outputs:
    Console convergence information and an interactive figure; no saved files.
Output location:
    None.
Dependencies:
    NumPy and Matplotlib; no local module imports.
Related files:
    Base method for the constraint and diagnostics comparison experiments.
"""

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#                            finite-difference symbol                          #
# --------------------------------------------------------------------------- #
def fd_symbol(N, dx, fd_order=2):
    """
    Return λ_fd[k]  =  discrete symbol of  -∂xx on a uniform 2π-periodic grid.
    fd_order = 2 -> 4 sin²(θ/2)/dx²   (3-point)
    fd_order = 4 -> (5/2 − 8/3 cosθ + 1/6 cos2θ)/dx²   (5-point)
    """
    k_phys = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    theta = k_phys * dx
    if fd_order == 2:
        lam = 4.0 * np.sin(theta / 2.0)**2 / dx**2  # λ₂(θ)
    elif fd_order == 4:
        lam = (2.5 - (8.0 / 3.0) * np.cos(theta) +
               (1.0 / 6.0) * np.cos(2.0 * theta)) / dx**2  # λ₄(θ)
    else:
        raise ValueError("fd_order must be 2 or 4")
    # lam[0] = 0.0  # enforce mean-zero condition (k=0 mode)
    return lam


# --------------------------------------------------------------------------- #
#                            fixed-point relaxation method                      #
# --------------------------------------------------------------------------- #
def solve_relaxation(eta,
                     N=512,
                     alpha=0.3,
                     fd_order=2,
                     tol=1e-8,
                     max_iter=10_000):
    """Fixed-point Picard iteration with relaxation α."""
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    lam = fd_symbol(N, dx, fd_order)  # λ_fd(k)  (positive)
    U = np.zeros(N)

    for k in range(max_iter):
        U_old = U.copy()
        f_val = -(np.sqrt(2) * eta**2 / np.pi) * np.sin(x + 2 * np.pi *
                                                        np.sqrt(2) * U)
        # f_val -= f_val.mean()  # solvability

        # Poisson solve:  λ_fd * W_hat = f_hat  ⇒  W_hat = f_hat/λ_fd
        f_hat = np.fft.fft(f_val)
        W_hat = np.zeros_like(f_hat)
        nz_idx = lam > 0  # skip k=0
        W_hat[nz_idx] = f_hat[nz_idx] / lam[nz_idx]
        W = np.real(np.fft.ifft(W_hat))

        U = (1 - alpha) * U + alpha * W
        U -= U.mean()  # enforce ⟨U⟩=0
        relerr = np.linalg.norm(U - U_old) / max(1e-30, np.linalg.norm(U_old))
        if relerr < tol:
            return U, True, k + 1
    return U, False, max_iter


# --------------------------------------------------------------------------- #
#                          semi-implicit Euler method for virtual time        #
# --------------------------------------------------------------------------- #
def solve_virtual_time(eta,
                       N=512,
                       h=0.4,
                       fd_order=2,
                       tol=1e-8,
                       max_iter=10_000):
    """Semi-implicit Euler:  (I - h∇²)^{-1}(U + h f(U))."""
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]
    lam = fd_symbol(N, dx, fd_order)  # λ_fd ≥ 0
    denom = 1.0 + h * lam  # (I - h∇²) = (I + h(-Δ)) = (I + h λ_fd) in Fourier
    # denom[0] = 1.0  # enforce mean-zero condition (k=0 mode)
    U = np.zeros(N) + 10
    print(U)

    for k in range(max_iter):
        U_old = U.copy()
        f_val = -(np.sqrt(2) * eta**2 / np.pi) * np.sin(x + 2 * np.pi *
                                                        np.sqrt(2) * U)
        R = U + h * f_val
        # R -= R.mean()  # keep solvability condition ⟨R⟩=0
        R_hat = np.fft.fft(R)
        U_hat = np.zeros_like(R_hat)
        nz_idx = lam > 0
        U_hat[nz_idx] = R_hat[nz_idx] / denom[nz_idx]

        U = np.real(np.fft.ifft(U_hat))
        # U -= U.mean()  # enforce ⟨U⟩=0
        relerr = np.linalg.norm(U - U_old) / max(1e-30, np.linalg.norm(U_old))
        if relerr < tol:
            return U, True, k + 1
    return U, False, max_iter


# --------------------------------------------------------------------------- #
#                            parameters table                                 #
# --------------------------------------------------------------------------- #
# eta  N    alpha  h     fd_order
# 1    512  0.3    0.4   4
# 1.3  512  0.2    0.2   4
# 3    512  0.04   0.035 4
# 10   512  0.004  0.003 4


# --------------------------------------------------------------------------- #
#                            main function                                     #
# --------------------------------------------------------------------------- #
def main():
    eta = 10
    N = 512
    # ---- parameters you want to test ----
    alpha = 0.004  # relaxation
    h = 0.003  # pseudo-time
    fd_order = 4  # 2  or  4
    # -------------------------------------

    # U_rel, ok_rel, nit_rel = solve_relaxation(eta,
    #                                           N=N,
    #                                           alpha=alpha,
    #                                           fd_order=fd_order)
    U_tim, ok_tim, nit_tim = solve_virtual_time(eta,
                                                N=N,
                                                h=h,
                                                fd_order=fd_order)

    # print(f"[Relaxation] converged={ok_rel}  iterations={nit_rel}")
    print(f"[Pseudo-time] converged={ok_tim}  iterations={nit_tim}")
    # if not ok_rel:
    #     print("WARNING  relaxation method did not converge.")
    if not ok_tim:
        print("WARNING  pseudo-time method did not converge.")

    # diff = np.linalg.norm(U_rel - U_tim)
    # print(f"L2-norm difference between solutions = {diff:.3e}")

    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # plt.plot(x, U_rel, label="Relaxation")
    plt.plot(x, U_tim, "--", label="Pseudo-time")
    plt.title(rf"η={eta}, FD order={fd_order}")
    plt.xlabel("X")
    plt.ylabel("U(X)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
