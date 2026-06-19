#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 10:38:56 2026

@author: junaidrehman
Difference from V12: Plotting average RMSE of estimation 
instead of the standard deviation. std is influenced by the randomness
of channels, MSE is a function of n_samples
"""

import numpy as np
from QKD_DE_V10 import rand_U_direct, no_opt_QKD
import matplotlib.pyplot as plt

eps = 0.8
prob = [eps, (1-eps)/3, (1-eps)/3, (1-eps)/3]

# n_samples = np.array(range(1, 7))
n_samples = np.linspace(1, 4, 10)
n_samples = np.round(10**n_samples)
n_samples = [int(v) for v in n_samples]
n_avg = 2000
st_p = 1001 # start point for seeds
seeds = range(st_p, st_p + n_avg + 1)

# Asymptotic runs (n_samples=None) for each seed
asym_skr_bb84 = []
asym_skr_ss = []

for seed in seeds:
    res = rand_U_direct(prob, seed=seed, n_samples=None)
    asym_skr_bb84.append(res['skr_bb84'])
    asym_skr_ss.append(res['skr_ss'])

# Finite sample runs for each n_samples
finite_means_bb84 = []
finite_rmse_bb84 = []
finite_means_ss = []
finite_rmse_ss = []

for n in n_samples:
    skr_bb84_list = []
    skr_ss_list = []
    errors_bb84 = []
    errors_ss = []
    
    for i, seed in enumerate(seeds):
        res = rand_U_direct(prob, seed=seed, n_samples=n)
        skr_bb84_list.append(res['skr_bb84'])
        skr_ss_list.append(res['skr_ss'])
        
        # Error for this seed: (true - estimated)^2
        errors_bb84.append((asym_skr_bb84[i] - res['skr_bb84'])**2)
        errors_ss.append((asym_skr_ss[i] - res['skr_ss'])**2)
    
    # Mean of finite sample estimates
    finite_means_bb84.append(np.mean(skr_bb84_list))
    finite_means_ss.append(np.mean(skr_ss_list))
    
    # RMSE: sqrt of mean squared errors
    finite_rmse_bb84.append(np.sqrt(np.mean(errors_bb84)))
    finite_rmse_ss.append(np.sqrt(np.mean(errors_ss)))

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# BB84 plot
ax1.axhline(y=np.mean(asym_skr_bb84), color='blue', linestyle='--', 
            label=f'Asymptotic = {np.mean(asym_skr_bb84):.4f}')
ax1.errorbar(n_samples, finite_means_bb84, yerr=finite_rmse_bb84, 
             fmt='ro-', capsize=5, label='Finite sample (RMSE)')
ax1.set_xscale('log')
ax1.set_xlabel('Number of samples (n)')
ax1.set_ylabel('Secret Key Rate (BB84)')
ax1.set_title('BB84 Protocol: Convergence with Sample Size')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Six-state plot
ax2.axhline(y=np.mean(asym_skr_ss), color='blue', linestyle='--', 
            label=f'Asymptotic = {np.mean(asym_skr_ss):.4f}')
ax2.errorbar(n_samples, finite_means_ss, yerr=finite_rmse_ss, 
             fmt='ro-', capsize=5, label='Finite sample (RMSE)')
ax2.set_xscale('log')
ax2.set_xlabel('Number of samples (n)')
ax2.set_ylabel('Secret Key Rate (Six-State)')
ax2.set_title('Six-State Protocol: Convergence with Sample Size')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary
print(f"\n{'='*60}")
print(f"SUMMARY RESULTS")
print(f"{'='*60}")
print(f"Asymptotic BB84 (mean over seeds):  {np.mean(asym_skr_bb84):.6f} ± {np.std(asym_skr_bb84):.6f}")
print(f"Asymptotic Six-State (mean over seeds): {np.mean(asym_skr_ss):.6f} ± {np.std(asym_skr_ss):.6f}")
print(f"\nFinite sample results (Mean ± RMSE):")
print(f"{'Samples':>10} {'BB84 Mean':>12} {'BB84 RMSE':>12} {'SS Mean':>12} {'SS RMSE':>12}")
print(f"{'-'*60}")
for i, n in enumerate(n_samples):
    print(f"{n:>10} {finite_means_bb84[i]:>12.6f} {finite_rmse_bb84[i]:>12.6f} {finite_means_ss[i]:>12.6f} {finite_rmse_ss[i]:>12.6f}")