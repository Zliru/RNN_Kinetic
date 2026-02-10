"""
Generate data for Figure 2 and Figure 3 (RNN simulation vs DMFT prediction).
This script contains two parts:
1) RNN simulation: estimate steady-state kinetic energy (mean of squared velocity).
2) DMFT computation: solve for Delta0 (self-consistency) and compute Gamma0.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import math
import warnings
from scipy.optimize import root_scalar, minimize_scalar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = os.path.dirname(os.path.abspath(__file__))
os.makedirs(folder, exist_ok=True)
def phi(x):
    return torch.tanh(x)
def IntPhi(x):
    return torch.log(torch.cosh(x))

"""RNN simulation"""
def run_simulation(g, N, T=200.0, dt=0.1, T_avg=60, num_trials=100):
    n_steps = int(T / dt)
    avg_steps = n_steps - T_avg
    kinetic_list = []
    for _ in range(num_trials):
        J = torch.randn(N, N, device=device) * g / N**0.5
        x = torch.randn(N, device=device)
        dx_dt_history = []
        for step in range(n_steps):
            dx_dt = -x + J @ phi(x)
            x = x + dt * dx_dt
            if step >= avg_steps:
                dx_dt_history.append(dx_dt.clone())
        dx_dt_tensor = torch.stack(dx_dt_history)
        kinetic = (dx_dt_tensor**2).mean().item()
        kinetic_list.append(kinetic)
    return kinetic_list
""" DMFT computation """
def simulate_rnn_kinetic(N):
    g_list = torch.linspace(1.0, 1.1, 8)
    mean_kinetics = []
    std_kinetics = []
    NN = N
    for g in g_list:
        kinetics = run_simulation(g.item(),NN)
        mean_kinetics.append(torch.tensor(kinetics).mean().item())
        std_kinetics.append(torch.tensor(kinetics).std().item())
    filename = f"simulation_kinetic_vs_g{g_list[0]:.2f}to{g_list[-1]:.2f}_N{NN}.csv"
    df = pd.DataFrame({
        'g': g_list.cpu().numpy(),
        'mean_kinetic': mean_kinetics,
        'std_kinetic': std_kinetics  })
    df.to_csv(os.path.join(folder, filename), index=False)

"""DMFT computation"""
def Equation_D0_ggt1(x_tensor, g, y, z):
    x = torch.abs(x_tensor[0])
    term1 = -0.5 * x**2
    term2 = g**2 * (IntPhi(torch.sqrt(x) * z)**2).mean()
    term3 = g**2 * (IntPhi(torch.sqrt(x) * y).mean())**2
    return (term1 + term2 - term3).cpu().numpy()
def Solve_Delta0_ggt1(g):
    y = torch.randn(10_000_000, device=device)
    z = torch.randn(10_000_000, device=device)
    def scalar_eqn(delta):
        if delta <= 0:
            return 1e-6
        x_tensor = torch.tensor([delta], device=device)
        return Equation_D0_ggt1(x_tensor, g, y, z)
    sigma = g - 1.0
    approx = max(sigma , 1e-7)
    a = max(1e-8, approx / 10)
    b = approx * 4
    result = root_scalar(scalar_eqn, method='brentq', bracket=[a, b], xtol=1e-12)
    if result.converged and abs(scalar_eqn(result.root)) < 1e-6:
        return result
    else:
        raise RuntimeError("Delta0 solver failed or residual is too large.")
def Equation_D0_glt1(x_tensor, g, y, z):
    x = torch.clamp(torch.abs(x_tensor[0]), min=1e-8)
    sqrt_x = torch.sqrt(x)
    term1 = -0.5 * x**2
    term2 = g**2 * (IntPhi(sqrt_x * z)**2).mean()
    term3 = g**2 * (IntPhi(sqrt_x * y).mean())**2
    return (term1 + term2 - term3).item()
def Solve_Delta0_glt1(g):
    torch.manual_seed(0)
    y = torch.randn(10_000_000, dtype=torch.float64, device=device)
    z = torch.randn(10_000_000, dtype=torch.float64, device=device)
    def objective(x_scalar):
        x_tensor = torch.tensor([x_scalar], dtype=torch.float64, device=device)
        return abs(Equation_D0_glt1(x_tensor, g, y, z))
    bounds = (1e-6, 5.0)
    result = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-10})
    if result.success and result.fun < 1e-6:
        return result.x
    else:
        raise RuntimeError("Delta0 solver failed.")
def Calculate_Tau0vsg():
    g_arr = torch.linspace(1.0, 1.08, 80, device=device)
    z = torch.randn(200_000_000, device=device)
    g_valid = []
    Gamma0_list = []
    Delta0_list = []
    funResult = []
    Delta0_std = []
    Gamma0_std = []
    num = 200

    for g in g_arr:
        Deltas, Gammas = [], []
        for _ in range(num):
            try:
                if g < 1.0:
                    Delta0 = Solve_Delta0_glt1(g)
                else:
                    r = Solve_Delta0_ggt1(g.item())
                    Delta0 = r.root
                Delta0 = torch.tensor(Delta0, device=device)
                Gamma0 = g**2 * (phi(torch.sqrt(torch.abs(Delta0)) * z)**2).mean() - Delta0
                if Gamma0 < -0.000045:
                    continue
                else:
                    Deltas.append(Delta0.item())
                    Gammas.append(Gamma0.item())
            except Exception as e:
                continue
        if len(Deltas) == 0:
            print(rf"Warning: No valid results g = {g:.3f}")
            continue
        print(rf"Finished: Valid results g = {g:.3f}")

        g_valid.append(float(g))
        Delta0_arr = np.array(Deltas)
        Gamma0_arr = np.array(Gammas)
        Delta0_list.append(Delta0_arr.mean())
        Gamma0_list.append(Gamma0_arr.mean())
        Delta0_std.append(Delta0_arr.std())
        Gamma0_std.append(Gamma0_arr.std())
        funResult.append(Equation_D0_ggt1(torch.tensor([Delta0_arr.mean()], device=device), g, z, z))

    return Gamma0_list, g_valid,Delta0_list, funResult, Delta0_std, Gamma0_std
def simulate_dmft_kinetic():
    Gamma0, g_arr, Delta0, funResult, Delta0_std, Gamma0_std = Calculate_Tau0vsg()
    filename = f"DMFT_kinetic_vs_g{g_arr[0]:.2f}to{g_arr[-1]:.2f}.csv"
    df = pd.DataFrame({'g': g_arr, 'Gamma0': Gamma0, 'Delta0': Delta0,
        'funResult': funResult,'Delta0_std': Delta0_std, 'Gamma0_std': Gamma0_std   })
    df.to_csv(os.path.join(folder, filename), index=False)
    print("DMFT computation finished and saved.")

# ========== Main ==========
if __name__ == "__main__":
    start = time.time()
    N1=5000
    N2 = 10000
    simulate_dmft_kinetic()
    simulate_rnn_kinetic(N2)
    simulate_rnn_kinetic(N1)
    print(f"Total runtime: {time.time() - start:.2f} s")
