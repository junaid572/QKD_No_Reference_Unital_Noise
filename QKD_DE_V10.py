# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:30:38 2026

@author: junaid
module for importing. No plotting.
"""

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import numpy as np
import qutip as qt
import cvxpy as cp
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from functools import partial
from datetime import datetime
import os
from tqdm import tqdm




def _validate_prob(prob):
    prob = np.asarray(prob, dtype=float)
    if np.any(prob < 0):
        raise ValueError("All probabilities must be non-negative.")
    total = prob.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"Probabilities must sum to 1, got {total:.6f}.")
    return prob

def h2(x):
    x = np.asarray(x)
    result = np.zeros_like(x)
    mask = (x > 0) & (x < 1)
    result[mask] = -x[mask]*np.log2(x[mask]) - (1-x[mask])*np.log2(1-x[mask])
    return result

def skr_BB84(Qz, Qx):
    '''
    key generation rate eq. 3 and 4 of https://journals.aps.org/pra/pdf/10.1103/PhysRevA.101.062321


    Parameters
    ----------
    Qz : TYPE
        DESCRIPTION.
    Qx : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return np.maximum(0, 1 - h2(Qz) - h2(Qx))
def H(p):
    p = np.asarray(p, float)
    m = p > 0
    return -np.sum(p[m] * np.log2(p[m]))
# ── main functions ──────────────────────────────────────────────────────────────

def rand_U_channel(rho, prob, seed=42):
    """
    Implements random unitary channel with len(prob) random unitaries
    with probabilities prob. The output is the state N(rho).

    Parameters
    ----------
    rho   : qt.Qobj  — input density matrix (qubit)
    prob  : array-like of float — probabilities for each unitary; must sum to 1
    seed  : int — RNG seed for reproducibility

    Returns
    -------
    qt.Qobj — output density matrix N(rho)
    """
    prob = _validate_prob(prob)
    n = len(prob)

    # # Generate n random unitaries via U3 parametrization
    # rng = np.random.default_rng(seed)
    # # Sample three Euler angles per unitary:
    # #   theta in [0, pi], phi in [0, 2pi), lam in [0, 2pi)
    # thetas = rng.uniform(0, np.pi,  n)
    # phis   = rng.uniform(0, 2*np.pi, n)
    # lams   = rng.uniform(0, 2*np.pi, n)
    # unitaries = [_u3(thetas[k], phis[k], lams[k]) for k in range(n)]
    unitaries = [qt.rand_unitary(2, distribution="haar", seed = seed + k) for k in range(n)]
    # unitaries = [qt.rand_unitary_haar(2, seed = seed + k) for k in range(n)]

    # N(rho) = sum_k p_k * U_k * rho * U_k^dagger
    out = sum(p * U * rho * U.dag() for p, U in zip(prob, unitaries))
    out.dims = rho.dims  # preserve qubit structure
    return out


def rand_U_statistics(rho, prob, projector, n_samples=10000, seed=42):
    """
    Applies rand_U_channel to rho, then performs projective measurement
    {|v><v|, I - |v><v|} on the output state n_samples times.

    Parameters
    ----------
    rho       : qt.Qobj  — input density matrix
    prob      : array-like — probabilities for the random unitary channel
    projector : qt.Qobj  — ket defining the first POVM element
    n_samples : int — number of measurement shots
    seed      : int — RNG seed (also forwarded to rand_U_channel)

    Returns
    -------
    samples : np.ndarray of shape (n_samples,) with values in {0, 1}
              0 → outcome proj_0, 1 → outcome proj_1
    counts  : np.ndarray of shape (2,) — [count_0, count_1]
    """
    output_state = rand_U_channel(rho, prob, seed=seed)

    proj_0 = qt.ket2dm(projector)
    proj_1 = qt.qeye(2) - proj_0  # complementary projector

    # Born-rule probabilities
    p0 = float(np.real((proj_0 * output_state).tr()))
    p1 = float(np.real((proj_1 * output_state).tr()))
    p0 = np.clip(p0, 0, 1)        # guard against tiny floating-point negatives
    p1 = 1.0 - p0

    # Multinomial sampling with a fresh seed offset to avoid correlating
    # with the unitary generation seed
    rng = np.random.default_rng(seed + 1)
    counts = rng.multinomial(n_samples, [p0, p1])
    samples = np.repeat([0, 1], counts)
    rng.shuffle(samples)

    return samples, counts



def rand_U_direct(prob, seed=42, n_samples = None, verbose = False):
    """
    Directly constructing M, then calculating the skr. n_samples = None -> gives
    asymptotic, otherwise finite sample estimation.
    Returns skr, verifications removed. See V7 for verification of correctness
    """
    prob = _validate_prob(prob)
    
    # no reference frame. Alice is rotated by an unknown unitary U_r_A and Bob
    # by another U_r_B. We can use one w.l.o.g but I will keep two so that not to 
    # favor any one of them
    
    U_r_A = qt.rand_unitary(2, distribution="haar", seed = seed+100) # for consistency
    U_r_B = qt.rand_unitary(2, distribution="haar", seed = seed+101) # for consistency
    # U_r_A = qt.qeye(2)
    # U_r_B = qt.qeye(2)
    ket0 = qt.basis(2, 0)
    ket1 = qt.basis(2, 1)
    ketp = (ket0 + ket1).unit()
    ketm = (ket0 - ket1).unit()
    keti = (ket0 + 1j * ket1).unit()
    ketj = (ket0 - 1j * ket1).unit()
    
    ket0_A = U_r_A * ket0
    ket1_A = U_r_A * ket1
    ketp_A = U_r_A * ketp
    ketm_A = U_r_A * ketm
    keti_A = U_r_A * keti
    ketj_A = U_r_A * ketj
    
    ket0_B = U_r_B * ket0
    ket1_B = U_r_B * ket1
    ketp_B = U_r_B * ketp
    ketm_B = U_r_B * ketm
    keti_B = U_r_B * keti
    ketj_B = U_r_B * ketj



    # probe states: +1 eigenstates of X, Y, Z
    probes = {
        'X': qt.ket2dm(ketp_A),
        'Y': qt.ket2dm(keti_A),
        'Z': qt.ket2dm(ket0_A),
    }
    # measurement projectors: +1 eigenstates of X, Y, Z
    meas_kets = {'X': ketp_B, 'Y': keti_B, 'Z': ket0_B}

    paulis = {
        'X': qt.sigmax().full(),
        'Y': qt.sigmay().full(),
        'Z': qt.sigmaz().full(),
    }

    # ── calculate Bloch map M ─────────────────────────────────────────────────
    # M[i,j] = <sigma_i>_out when sigma_j/2 + I/2 is input
    # <sigma_i>_out = 2 * P(+1 outcome in basis i) - 1
    # = (counts[0] - counts[1]) / n_samples

    axes = ['X', 'Y', 'Z']
    M = np.zeros((3, 3), dtype=float)
    
    if n_samples is None: # asymtptic estimate
        for j, probe_axis in enumerate(axes):
            rho_in = probes[probe_axis]
            rho_out = rand_U_channel(rho_in, prob, seed=seed)
        
            for i, meas_axis in enumerate(axes):
                proj = meas_kets[meas_axis]        # |v⟩
                P = 2 * qt.ket2dm(proj) - qt.qeye(2)   # Pauli observable along that axis
                M[i, j] = (P * rho_out).tr().real
    else:
        for j, probe_axis in enumerate(axes): # estimate with samples
           rho_in = probes[probe_axis]
           for i, meas_axis in enumerate(axes):
               proj_ket = meas_kets[meas_axis]
               _, counts = rand_U_statistics(rho_in, prob, proj_ket,
                                             n_samples=n_samples, seed=seed)
               M[i, j] = (counts[0] - counts[1]) / n_samples
               # 0.5 * above is omitted because P_j = 2Pi_0 - I -> unital -> ...


    # ── reconstruct channel from M ───────────────────────────────────────────
    # N(rho) = I/2 + sum_{i,j} M[i,j] * tr(sigma_j * rho) * sigma_i / 2
    # For a density matrix rho = (I + r_x X + r_y Y + r_z Z)/2:
    #   r_out = M @ r_in
    # So N(rho) = (I + (M @ r_in) . sigma) / 2

    
    
    def orth(ket):
        a, b = ket.full().flatten()
        return qt.Qobj([[-np.conj(b)], [np.conj(a)]]).unit()

    
    U_svd, sigma, Vt = np.linalg.svd(M)
    V = Vt.T
    
    # fix determinant signs if needed (SO3 not O3)
    if np.linalg.det(U_svd) < 0:
        U_svd[:, -1] *= -1
        sigma[-1]    *= -1
    if np.linalg.det(V) < 0:
        V[:, -1]  *= -1
        sigma[-1] *= -1
    
    # # Alice's unitary: rotates her state into the channel's principal frame
    
    # Alice's kth state
    k = 0
    ketc_0_A = state_from_bloch_vec(V[:, k])    # V^T in SO(3) -> SU(2)
    
    # Alice's orthogonal state
    ketc_1_A = orth(ketc_0_A)
    ketc_p_A = state_from_bloch_vec(V[:, k+1])
    # phase is ket_p_A = ket_0_A + e^{ij*phase}*ket_1_A, 
    # this is what we should extract
    alpha = ketc_0_A.dag() * ketc_p_A
    ketc_p_A = ketc_p_A * np.exp(-1j * np.angle(alpha))
    phase = np.angle((ketc_1_A.dag() * ketc_p_A))
    U_Alice = ketc_0_A * ket0_A.dag() + np.exp(1j*phase) * ketc_1_A * ket1_A.dag()

    # # Bob's Unitary
    k = 0
    # Bob's kth direction
    ketc_0_B = state_from_bloch_vec(U_svd[:, k])
    ketc_1_B = orth(ketc_0_B)
    ketc_p_B = state_from_bloch_vec(U_svd[:, k+1])
    # phase is ket_p_A = ket_0_A + e^{ij*phase}*ket_1_A, 
    # this is what we should extract
    
    alpha = ketc_0_B.dag() * ketc_p_B
    ketc_p_B = ketc_p_B * np.exp(-1j * np.angle(alpha))
    phase = np.angle((ketc_1_B.dag() * ketc_p_B))
    
    # U_Bob = qt.Qobj(
    #     [[ket0.dag()*ket_0_B, np.exp(1j*phase)*ket0.dag()*ket_1_B], 
    #      [ket1.dag()*ket_0_B, np.exp(1j*phase)*ket1.dag()*ket_1_B]]
    #     )
    U_Bob = ketc_0_B * ket0_B.dag() + np.exp(1j*phase) * ketc_1_B * ket1_B.dag()
    # print(U_Alice)
    # print(U_Bob)
    # print(M)
    # ch_in = qt.ket2dm(U_Alice*ket0_A)
    ch_in = qt.ket2dm(U_r_A*U_Alice*ket0_A)
    meas = U_r_B*U_Bob*ket0_B
    meas = qt.ket2dm(meas) 
    rho_out = rand_U_channel(ch_in, prob, seed = seed)
    Q1 = (meas*rho_out).tr().real
    rng = np.random.default_rng(seed + 1)
    if n_samples is None:
        Q1 = Q1
    else:
        Q1 = rng.multinomial(n_samples, [Q1, 1-Q1])
        Q1 = Q1[0]/n_samples
    Q1 = min(Q1, 1 - Q1)
    
    ch_in = qt.ket2dm(U_r_A*U_Alice*ketp_A)
    meas = U_r_B*U_Bob*ketp_B
    meas = qt.ket2dm(meas) 
    rho_out = rand_U_channel(ch_in, prob, seed = seed)
    Q2 = (meas*rho_out).tr().real
    if n_samples is None:
        Q2 = Q2
    else:
        Q2 = rng.multinomial(n_samples, [Q2, 1-Q2])
        Q2 = Q2[0]/n_samples
    Q2 = min(Q2, 1 - Q2)
    
    ch_in = qt.ket2dm(U_r_A*U_Alice*keti_A)
    meas = U_r_B*U_Bob*keti_B
    meas = qt.ket2dm(meas) 
    rho_out = rand_U_channel(ch_in, prob, seed = seed)
    Q3 = (meas*rho_out).tr().real
    if n_samples is None:
        Q3 = Q3
    else:
        Q3 = rng.multinomial(n_samples, [Q3, 1-Q3])
        Q3 = Q3[0]/n_samples
    Q3 = min(Q3, 1 - Q3)
    Q_s = sorted([Q1, Q2, Q3])
    Q1, Q2, Q3 = Q_s[0], Q_s[1], Q_s[2]
    # print(Q1, Q2, Q3)
    # print("*********", 1 - Q)
    # Q_from_s = (1 - sigma) / 2
    # print("Q_from_s:", Q_from_s)


    # BB84: two best bases
    skr_bb84 = max(0, 1 - h2(Q1) - h2(Q2))
    Qz, Qx, Qy = Q1, Q2, Q3

    # six-state: all three bases
    l00 = 1 - (Qx + Qy + Qz)/2
    l01 = (Qx + Qy - Qz)/2
    l10 = (-Qx + Qy + Qz)/2
    l11 = (Qx - Qy + Qz)/2

    skr_six = max(0, 1 - H([l00, l01, l10, l11]))
    if verbose:
        if skr_six - skr_bb84 < -0.05:
            print(f"BB84_skr: {skr_bb84:.4f}, SS_skr: {skr_six:.4f}, Qz: {Qz:.4f}, Qx: {Qx:.4f}, Qy: {Qy:.4f}")
        
    return {
        'skr_bb84': skr_bb84,
        'skr_ss': skr_six,
        'Qz': Q1,
        'Qx': Q2,
        'Qy': Q3
    }


def no_opt_QKD(prob, seed=42, verbose=False):
    """
    Standard QKD (BB84 and six-state) without optimization.
    For BB84, uses the two best bases among X, Y, Z.
    For six-state, uses all three bases.
    """
    
    # Define standard states
    ket_Z0 = qt.basis(2, 0)
    ket_Z1 = qt.basis(2, 1)
    ket_Xp = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    ket_Xm = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
    ket_Yp = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
    ket_Ym = (qt.basis(2, 0) - 1j * qt.basis(2, 1)).unit()
    
    bases = {
        'Z': ([ket_Z0, ket_Z1], [ket_Z0, ket_Z1]),
        'X': ([ket_Xp, ket_Xm], [ket_Xp, ket_Xm]),
        'Y': ([ket_Yp, ket_Ym], [ket_Yp, ket_Ym])
    }
    
    def basis_statistics(basis_name):
        state_pairs, meas_pairs = bases[basis_name]
        error_rates_list = []
        
        for i, ket_s in enumerate(state_pairs):
            rho_s = qt.ket2dm(ket_s)
            rho_out = rand_U_channel(rho_s, prob, seed=seed)
            
            counts = []
            for ket_m in meas_pairs:
                P_m = qt.ket2dm(ket_m)
                p_click = (P_m * rho_out).tr().real
                counts.append(p_click)
            
            counts = np.array(counts) / np.sum(counts)
            # For each prepared state, best measurement outcome
            best_prob = np.max(counts)
            error_rates_list.append(1 - best_prob)
        
        return np.mean(error_rates_list)
    
    # Compute error rates for all three bases
    error_rates = {}
    for basis_name in bases.keys():
        error_rates[basis_name] = basis_statistics(basis_name)
        if verbose:
            print(f"{basis_name}-basis error rate: {error_rates[basis_name]:.6f}")
    
    def h2(x):
        if x <= 0 or x >= 1:
            return 0
        return -x * np.log2(x) - (1-x) * np.log2(1-x)
    
    def H(prob):
        """
        Shannon entropy H(p) = -sum_i p_i log2 p_i
        with safe handling of zero probabilities.
        """
        prob = np.asarray(prob, dtype=float)
        # mask out zeros to avoid log(0)
        mask = (prob > 0)
        return -np.sum(prob[mask] * np.log2(prob[mask]))
    
    # BB84: two best bases, basis-specific formula
    sorted_errors = sorted([error_rates['Z'], error_rates['X'], error_rates['Y']])
    Q1, Q2 = sorted_errors[0], sorted_errors[1]
    skr_bb84 = max(0, 1 - h2(Q1) - h2(Q2))
    

    # Six-state: all three bases, Bruss (1998)
    Qz, Qx, Qy = sorted_errors
    l00 = 1 - (Qx + Qy + Qz)/2
    l01 = (Qx + Qy - Qz)/2
    l10 = (-Qx + Qy + Qz)/2
    l11 = (Qx - Qy + Qz)/2
    skr_six = max(0, 1 - H([l00, l01, l10, l11]))
    
    # print("===========No Optimization=========")
    # print("Prob:", prob, "Seed: ", seed)
    # print(f"BB84 (2 best bases) - QBER: {qber_bb84:.6f}, SKR: {skr_bb84:.6f}")
    # print("=======================================")
    
    return {
        'skr_bb84': skr_bb84,
        'skr_ss': skr_six,
        'Qz': Qz,
        'Qx': Qx,
        'Qy':Qy
    }




def state_from_bloch_vec(n):
    """
    Given unit vector n, return pure state rho = (I + n·σ)/2.
    """
    nx, ny, nz = n
    I = qt.qeye(2)
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    rho = 0.5 * (I + nx * X + ny * Y + nz * Z)
    # pick eigenstate with eigenvalue 1 (pure state along n)
    evals, evecs = rho.eigenstates()
    idx = np.argmax(evals)
    return evecs[idx].unit()

