# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:30:38 2026

@author: junaid
CORRECTIONS: 
1) key generation rate eq. 3 and 4 of https://journals.aps.org/pra/pdf/10.1103/PhysRevA.101.062321
2) Global optimization based on the skr, instead of basis with least errors...
3) using qutip random unitaries
4) exporting all data in addition to statistical results...
"""

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


# Set global plotting parameters for LaTeX-style plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ── helpers ────────────────────────────────────────────────────────────────────

def _u3(theta, phi, lam):
    """SU(2) gate parametrized by three Euler angles (U3 convention)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return qt.Qobj(np.array([
        [c,                        -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s,      np.exp(1j * (phi + lam)) * c]
    ]))


def _validate_prob(prob):
    prob = np.asarray(prob, dtype=float)
    if np.any(prob < 0):
        raise ValueError("All probabilities must be non-negative.")
    total = prob.sum()
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"Probabilities must sum to 1, got {total:.6f}.")
    return prob


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



# --- helpers ---------------------------------------------------------

def bloch_state(theta, phi):
    """
    |psi(theta,phi)> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>
    """
    return (np.cos(theta/2) * qt.basis(2, 0)
            + np.exp(1j * phi) * np.sin(theta/2) * qt.basis(2, 1)).unit()

def projector_from_state(ket):
    return qt.ket2dm(ket)

def bloch_vector(theta, phi):
    """
    Return unit Bloch vector n = (nx, ny, nz) for angles (theta, phi).
    """
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    return np.array([nx, ny, nz])

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

def orthonormal_bloch_basis(n):
    """
    Given unit vector n, construct two orthonormal unit vectors m, k
    such that {n, m, k} is an ONB in R^3.
    """
    n = n / np.linalg.norm(n)
    # pick any vector not parallel to n
    if abs(n[2]) < 0.9:
        tmp = np.array([0, 0, 1.0])
    else:
        tmp = np.array([1.0, 0, 0])
    m = tmp - np.dot(tmp, n) * n
    m /= np.linalg.norm(m)
    k = np.cross(n, m)
    k /= np.linalg.norm(k)
    return n, m, k

# --- main routine ----------------------------------------------------



def opt_QKD(prob, seed=42, n_samples=10000, verbose=False):
    """
    Joint optimization of BB84 and Six-State bases.
    Returns SKRs and QBERs in all relevant bases.
    """

    rng = np.random.default_rng(seed)

    # --- Helpers -------------------------------------------------------------

    def h2(x):
        return 0 if x <= 0 or x >= 1 else -x*np.log2(x) - (1-x)*np.log2(1-x)

    def H(p):
        p = np.asarray(p, float)
        m = p > 0
        return -np.sum(p[m] * np.log2(p[m]))

    def orth(ket):
        a, b = ket.full().flatten()
        return qt.Qobj([[-np.conj(b)], [np.conj(a)]]).unit()

    def superpos(k0, k1, phase):
        return (k0 + np.exp(1j*phase)*k1).unit()

    def bloch_vec(ket):
        rho = qt.ket2dm(ket)
        return np.array([
            (rho*qt.sigmax()).tr().real,
            (rho*qt.sigmay()).tr().real,
            (rho*qt.sigmaz()).tr().real
        ])

    def state_from_vec(n):
        n = n / np.linalg.norm(n)
        theta = np.arccos(np.clip(n[2], -1, 1))
        phi = np.arctan2(n[1], n[0])
        return bloch_state(theta, phi)

    # --- BB84 cost -----------------------------------------------------------

    def cost_bb84(p):
        th_s, ph_s, th_m, ph_m, a, b = p

        ket_s = bloch_state(th_s, ph_s)
        ket_m = bloch_state(th_m, ph_m)

        rho_s = qt.ket2dm(ket_s)
        rho_out = rand_U_channel(rho_s, prob, seed=seed)

        pZ = (qt.ket2dm(ket_m) * rho_out).tr().real

        ket_1 = orth(ket_s)
        ket_p = superpos(ket_s, ket_1, a)
        meas_p = superpos(ket_m, orth(ket_m), b)

        _, counts = rand_U_statistics(qt.ket2dm(ket_p), prob, meas_p,
                                      n_samples=n_samples, seed=seed)
        pX = counts[0]/np.sum(counts) if len(counts) else 0

        return -(1 - h2(pZ) - h2(pX))

    # --- Six-State cost ------------------------------------------------------

    def cost_six(p):
        th_s, ph_s, th_m, ph_m, a, b = p

        ket_s = bloch_state(th_s, ph_s)
        ket_m = bloch_state(th_m, ph_m)

        rho_s = qt.ket2dm(ket_s)
        rho_out = rand_U_channel(rho_s, prob, seed=seed)

        pZ = (qt.ket2dm(ket_m) * rho_out).tr().real
        pZ = max(pZ, 1 - pZ)

        ket_1 = orth(ket_s)
        ket_p = superpos(ket_s, ket_1, a)
        meas_p = superpos(ket_m, orth(ket_m), b)

        _, counts = rand_U_statistics(qt.ket2dm(ket_p), prob, meas_p,
                                      n_samples=n_samples, seed=seed)
        pX = max(counts)/np.sum(counts) if len(counts) else 0

        # Third basis via cross product
        nZ, nX = bloch_vec(ket_s), bloch_vec(ket_p)
        mZ, mX = bloch_vec(ket_m), bloch_vec(meas_p)

        nY = np.cross(nZ, nX)
        if np.linalg.norm(nY) < 1e-6:
            nY = np.array([-nZ[1], nZ[0], 0]) if abs(nZ[2]) < 0.9 else np.array([0, -nZ[2], nZ[1]])
        nY /= np.linalg.norm(nY)

        mY = np.cross(mZ, mX)
        if np.linalg.norm(mY) < 1e-6:
            mY = np.array([-mZ[1], mZ[0], 0]) if abs(mZ[2]) < 0.9 else np.array([0, -mZ[2], mZ[1]])
        mY /= np.linalg.norm(mY)

        ket_Y = state_from_vec(nY)
        meas_Y = state_from_vec(mY)

        _, cY = rand_U_statistics(qt.ket2dm(ket_Y), prob, meas_Y,
                                  n_samples=n_samples, seed=seed)
        pY = max(cY)/np.sum(cY)

        Qz, Qx, Qy = 1-pZ, 1-pX, 1-pY

        l00 = 1 - (Qx + Qy + Qz)/2
        l01 = (Qx + Qy - Qz)/2
        l10 = (-Qx + Qy + Qz)/2
        l11 = (Qx - Qy + Qz)/2

        return -(1 - H([l00, l01, l10, l11]))

    # --- Optimization wrapper ------------------------------------------------

    starts = [
        [0.1*np.pi, 0, 0.9*np.pi, np.pi, np.pi, np.pi],
        [0.5*np.pi, 0, 0.5*np.pi, np.pi, 3*np.pi/2, 3*np.pi/2],
        [0.8*np.pi, np.pi/2, 0.2*np.pi, 3*np.pi/2, 5*np.pi/4, 5*np.pi/4],
        [0.3*np.pi, np.pi/4, 0.7*np.pi, 5*np.pi/4, 5*np.pi/4, 5*np.pi/4],
        [0, 0, 0, 0, 5*np.pi/4, 5*np.pi/4],
        [np.pi, 0, np.pi, 0, 5*np.pi/4, 5*np.pi/4],
    ]

    bounds = [(0, 2*np.pi), (0, 2*np.pi)]*3

    def run_opt(cost):
        best = (np.inf, None)
        for x0 in starts:
            r = minimize(cost, x0, method='Powell', bounds=bounds,
                         options={'ftol':1e-8,'xtol':1e-8,'maxiter':800})
            if r.fun < best[0]:
                best = (r.fun, r.x)
        return best

    # --- Run BB84 optimization ----------------------------------------------

    cost_bb84_val, p_bb84 = run_opt(cost_bb84)
    skr_bb84 = max(0, -cost_bb84_val)

    # Extract optimized QBERs
    th_s, ph_s, th_m, ph_m, a, b = p_bb84
    ket_s = bloch_state(th_s, ph_s)
    ket_m = bloch_state(th_m, ph_m)
    ket_1 = orth(ket_s)
    ket_p = superpos(ket_s, ket_1, a)
    meas_p = superpos(ket_m, orth(ket_m), b)

    rho_s = qt.ket2dm(ket_s)
    rho_out = rand_U_channel(rho_s, prob, seed=seed)

    pZ = (qt.ket2dm(ket_m) * rho_out).tr().real
    _, cX = rand_U_statistics(qt.ket2dm(ket_p), prob, meas_p,
                              n_samples=n_samples, seed=seed)
    pX = cX[0]/np.sum(cX) if len(cX) else 0

    Qz_opt_bb84, Qx_opt_bb84 = min(pZ, 1-pZ), min(pX, 1-pX)

    # --- Run Six-State optimization -----------------------------------------

    cost_six_val, p_six = run_opt(cost_six)
    skr_ss = max(0, -cost_six_val)

    # Extract optimized QBERs for six-state
    th_s, ph_s, th_m, ph_m, a, b = p_six
    ket_s = bloch_state(th_s, ph_s)
    ket_m = bloch_state(th_m, ph_m)
    ket_1 = orth(ket_s)
    ket_p = superpos(ket_s, ket_1, a)
    meas_p = superpos(ket_m, orth(ket_m), b)

    # Z and X QBERs same as BB84 logic
    rho_s = qt.ket2dm(ket_s)
    rho_out = rand_U_channel(rho_s, prob, seed=seed)
    pZ = (qt.ket2dm(ket_m) * rho_out).tr().real
    _, cX = rand_U_statistics(qt.ket2dm(ket_p), prob, meas_p,
                              n_samples=n_samples, seed=seed)
    pX = max(cX)/np.sum(cX) if len(cX) else 0

    # Y basis
    nZ, nX = bloch_vec(ket_s), bloch_vec(ket_p)
    nY = np.cross(nZ, nX)
    nY = nY/np.linalg.norm(nY)
    ket_Y = state_from_vec(nY)

    mZ, mX = bloch_vec(ket_m), bloch_vec(meas_p)
    mY = np.cross(mZ, mX)
    mY = mY/np.linalg.norm(mY)
    meas_Y = state_from_vec(mY)

    _, cY = rand_U_statistics(qt.ket2dm(ket_Y), prob, meas_Y,
                              n_samples=n_samples, seed=seed)
    pY = max(cY)/np.sum(cY)

    Qz_opt_ss, Qx_opt_ss, Qy_opt_ss = min(pZ, 1-pZ), min(pX, 1-pX), min(pY, 1-pY)

    return {
    'skr_bb84': skr_bb84,
    'skr_ss': skr_ss,

    # Optimized QBERs for BB84
    'Qz_opt_bb84': Qz_opt_bb84,
    'Qx_opt_bb84': Qx_opt_bb84,

    # Optimized QBERs for Six-State
    'Qz_opt_ss': Qz_opt_ss,
    'Qx_opt_ss': Qx_opt_ss,
    'Qy_opt_ss': Qy_opt_ss,
}




def no_opt_QKD(prob, seed=42, n_samples=10000, verbose=False):
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
    qber_bb84 = 0.5*(Q1 + Q2)
    

    # Six-state: all three bases, Bruss (1998)
    Qz, Qx, Qy = sorted_errors
    l00 = 1 - (Qx + Qy + Qz)/2
    l01 = (Qx + Qy - Qz)/2
    l10 = (-Qx + Qy + Qz)/2
    l11 = (Qx - Qy + Qz)/2
    skr_six = max(0, 1 - H([l00, l01, l10, l11]))
    
    qber_six = (Qx + Qy + Qz)/3
    # print("===========No Optimization=========")
    # print("Prob:", prob, "Seed: ", seed)
    # print(f"BB84 (2 best bases) - QBER: {qber_bb84:.6f}, SKR: {skr_bb84:.6f}")
    # print("=======================================")
    if verbose:
        print(f"\nStandard QKD Results:")
        print(f"BB84 (2 best bases) - QBER: {qber_bb84:.6f}, SKR: {skr_bb84:.6f}")
        print(f"Six-State (3 bases) - QBER: {qber_six:.6f}, SKR: {skr_six:.6f}")
    
    return {
        'skr_bb84': skr_bb84,
        'skr_ss': skr_six,
        'Qz': error_rates['Z'],
        'Qx': error_rates['X'],
        'Qy':error_rates['Y']
    }

def wrapper(args):
    return compute_for_parameters(*args)


def compute_for_parameters(eps, seed, n_samples=10000):
    """Compute SKRs and QBERs for given epsilon and seed."""
    
    prob = [eps, (1 - eps)/3, (1 - eps)/3, (1 - eps)/3]

    try:
        res_opt = opt_QKD(prob, seed=seed, n_samples=n_samples, verbose=False)
        res_std = no_opt_QKD(prob, seed=seed, n_samples=n_samples, verbose=False)

        return {
            'eps': eps,
            'seed': seed,

            # --- Optimized SKRs ---
            'opt_skr_bb84': res_opt['skr_bb84'],
            'opt_skr_six':  res_opt['skr_ss'],

            # --- Standard SKRs ---
            'std_skr_bb84': res_std['skr_bb84'],
            'std_skr_six':  res_std['skr_ss'],

            # --- Improvements ---
            'improvement_bb84': res_opt['skr_bb84'] - res_std['skr_bb84'],
            'improvement_six':  res_opt['skr_ss']  - res_std['skr_ss'],

            # --- Optimized QBERs (BB84) ---
            'Qz_opt_bb84': res_opt['Qz_opt_bb84'],
            'Qx_opt_bb84': res_opt['Qx_opt_bb84'],

            # --- Optimized QBERs (Six-State) ---
            'Qz_opt_six': res_opt['Qz_opt_ss'],
            'Qx_opt_six': res_opt['Qx_opt_ss'],
            'Qy_opt_six': res_opt['Qy_opt_ss'],

            # --- Standard QBERs ---
            'Qz_std': res_std['Qz'],
            'Qx_std': res_std['Qx'],
            'Qy_std': res_std['Qy'],
        }

    except Exception as e:
        print(f"Error for eps={eps}, seed={seed}: {e}")
        return None


def save_results_to_file(statistics, filename_prefix="QKD_results"):
    """Save SKR + QBER statistics to a tab-separated data file."""

    os.makedirs("data_files", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_files/{filename_prefix}_{timestamp}.dat"

    # --- Header (clean, ordered, complete) ---
    header = (
        "eps\t"
        "opt_skr_bb84_mean\topt_skr_bb84_std\t"
        "opt_skr_six_mean\topt_skr_six_std\t"
        "std_skr_bb84_mean\tstd_skr_bb84_std\t"
        "std_skr_six_mean\tstd_skr_six_std\t"
        "improvement_bb84_mean\timprovement_bb84_std\t"
        "improvement_six_mean\timprovement_six_std\t"
        "Qz_std_mean\tQz_std_std\t"
        "Qx_std_mean\tQx_std_std\t"
        "Qy_std_mean\tQy_std_std\t"
        "Qz_opt_bb84_mean\tQz_opt_bb84_std\t"
        "Qx_opt_bb84_mean\tQx_opt_bb84_std\t"
        "Qz_opt_six_mean\tQz_opt_six_std\t"
        "Qx_opt_six_mean\tQx_opt_six_std\t"
        "Qy_opt_six_mean\tQy_opt_six_std"
    )

    # --- Data rows ---
    data = []
    for s in statistics:
        row = [
            s['eps'],

            # SKRs
            s['opt_skr_bb84_mean'], s['opt_skr_bb84_std'],
            s['opt_skr_six_mean'],  s['opt_skr_six_std'],
            s['std_skr_bb84_mean'], s['std_skr_bb84_std'],
            s['std_skr_six_mean'],  s['std_skr_six_std'],

            # Improvements
            s['improvement_bb84_mean'], s['improvement_bb84_std'],
            s['improvement_six_mean'],  s['improvement_six_std'],

            # Standard QBERs
            s['Qz_std_mean'], s['Qz_std_std'],
            s['Qx_std_mean'], s['Qx_std_std'],
            s['Qy_std_mean'], s['Qy_std_std'],

            # Optimized QBERs (BB84)
            s['Qz_opt_bb84_mean'], s['Qz_opt_bb84_std'],
            s['Qx_opt_bb84_mean'], s['Qx_opt_bb84_std'],

            # Optimized QBERs (Six-State)
            s['Qz_opt_six_mean'], s['Qz_opt_six_std'],
            s['Qx_opt_six_mean'], s['Qx_opt_six_std'],
            s['Qy_opt_six_mean'], s['Qy_opt_six_std'],
        ]
        data.append(row)

    np.savetxt(filename, np.array(data), delimiter='\t', header=header, comments='')
    print(f"\nResults saved to: {filename}")
    return filename


def run_qkd_simulation(epss_v, seeds, n_samples=10000, n_workers=4):
    """Run QKD simulations over epsilon values and seeds using multiprocessing."""
    print(f"\n{'='*60}")
    print(f"Running QKD simulations")
    print(f"{'='*60}")
    print(f"Epsilon values: {len(epss_v)} (from {epss_v[0]:.4f} to {epss_v[-1]:.4f})")
    print(f"Seeds: {len(seeds)}")
    print(f"Total simulations: {len(epss_v) * len(seeds)}")
    print(f"Using {n_workers} workers")
    print(f"Samples per simulation: {n_samples}")
    
    param_list = [(eps, seed, n_samples) for eps in epss_v for seed in seeds]
    
    # with mp.Pool(processes=n_workers) as pool:
    #     results = pool.starmap(compute_for_parameters, param_list)
    with mp.Pool(processes=n_workers) as pool:
        results = []
        for r in tqdm(pool.imap_unordered(wrapper, param_list),
                      total=len(param_list),
                      desc="Running simulations"):
            results.append(r)

    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Organize results by epsilon
    results_by_eps = {}
    for res in results:
        eps = res['eps']
        if eps not in results_by_eps:
            results_by_eps[eps] = {key: [] for key in res.keys() if key not in ['eps', 'seed']}
        
        for key in results_by_eps[eps].keys():
            results_by_eps[eps][key].append(res[key])
    
    # ------------------------------------------------------------
    # Save full raw data BEFORE computing statistics
    # ------------------------------------------------------------
    os.makedirs("data_files", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"data_files/full_data_{timestamp}.dat"

    # Build header dynamically from keys
    sample_keys = [k for k in results_by_eps[epss_v[0]].keys()]
    header = "eps\tseed\t" + "\t".join(sample_keys)

    # Build rows
    full_rows = []
    for res in sorted(results, key=lambda r: (r['eps'], r['seed'])):
        row = [res['eps'], res['seed']] + [res[k] for k in sample_keys]
        full_rows.append(row)

    np.savetxt(full_filename, np.array(full_rows), delimiter='\t',
               header=header, comments='')

    print(f"\nFull raw data saved to: {full_filename}\n")

    
    
    # Calculate statistics
    statistics = []
    
    for eps in sorted(results_by_eps.keys()):
        data = results_by_eps[eps]
    
        stats = {
            'eps': eps,
    
            # --- Optimized SKRs ---
            'opt_skr_bb84_mean': np.mean(data['opt_skr_bb84']),
            'opt_skr_bb84_std':  np.std(data['opt_skr_bb84']),
            'opt_skr_six_mean':  np.mean(data['opt_skr_six']),
            'opt_skr_six_std':   np.std(data['opt_skr_six']),
    
            # --- Standard SKRs ---
            'std_skr_bb84_mean': np.mean(data['std_skr_bb84']),
            'std_skr_bb84_std':  np.std(data['std_skr_bb84']),
            'std_skr_six_mean':  np.mean(data['std_skr_six']),
            'std_skr_six_std':   np.std(data['std_skr_six']),
    
            # --- Improvements ---
            'improvement_bb84_mean': np.mean(data['improvement_bb84']),
            'improvement_bb84_std':  np.std(data['improvement_bb84']),
            'improvement_six_mean':  np.mean(data['improvement_six']),
            'improvement_six_std':   np.std(data['improvement_six']),
    
            # --- Standard QBERs ---
            'Qz_std_mean': np.mean(data['Qz_std']),
            'Qz_std_std':  np.std(data['Qz_std']),
            'Qx_std_mean': np.mean(data['Qx_std']),
            'Qx_std_std':  np.std(data['Qx_std']),
            'Qy_std_mean': np.mean(data['Qy_std']),
            'Qy_std_std':  np.std(data['Qy_std']),
    
            # --- Optimized QBERs (BB84) ---
            'Qz_opt_bb84_mean': np.mean(data['Qz_opt_bb84']),
            'Qz_opt_bb84_std':  np.std(data['Qz_opt_bb84']),
            'Qx_opt_bb84_mean': np.mean(data['Qx_opt_bb84']),
            'Qx_opt_bb84_std':  np.std(data['Qx_opt_bb84']),
    
            # --- Optimized QBERs (Six-State) ---
            'Qz_opt_six_mean': np.mean(data['Qz_opt_six']),
            'Qz_opt_six_std':  np.std(data['Qz_opt_six']),
            'Qx_opt_six_mean': np.mean(data['Qx_opt_six']),
            'Qx_opt_six_std':  np.std(data['Qx_opt_six']),
            'Qy_opt_six_mean': np.mean(data['Qy_opt_six']),
            'Qy_opt_six_std':  np.std(data['Qy_opt_six']),
        }
    
        statistics.append(stats)
    
    return statistics

def plot_qkd_results_latex(stats, save_plot=True, show_plot=True):
    """
    Publication-quality QKD results plot.
    2×2 grid:
        (1) BB84 SKR
        (2) Six-State SKR
        (3) Standard QBERs
        (4) Optimized QBERs (BB84 + Six-State)
    """

    eps = np.array([s['eps'] for s in stats])

    # Helper to extract arrays
    def A(key):
        return np.array([s[key] for s in stats])

    # --- SKR statistics ---
    opt_bb84_mean, opt_bb84_std = A('opt_skr_bb84_mean'), A('opt_skr_bb84_std')
    std_bb84_mean, std_bb84_std = A('std_skr_bb84_mean'), A('std_skr_bb84_std')

    opt_six_mean, opt_six_std = A('opt_skr_six_mean'), A('opt_skr_six_std')
    std_six_mean, std_six_std = A('std_skr_six_mean'), A('std_skr_six_std')

    # --- Standard QBERs ---
    Qz_std, Qx_std, Qy_std = A('Qz_std_mean'), A('Qx_std_mean'), A('Qy_std_mean')
    Qz_std_s, Qx_std_s, Qy_std_s = A('Qz_std_std'), A('Qx_std_std'), A('Qy_std_std')

    # --- Optimized QBERs (BB84) ---
    Qz_opt_bb84, Qx_opt_bb84 = A('Qz_opt_bb84_mean'), A('Qx_opt_bb84_mean')
    Qz_opt_bb84_s, Qx_opt_bb84_s = A('Qz_opt_bb84_std'), A('Qx_opt_bb84_std')

    # --- Optimized QBERs (Six-State) ---
    Qz_opt_six, Qx_opt_six, Qy_opt_six = (
        A('Qz_opt_six_mean'), A('Qx_opt_six_mean'), A('Qy_opt_six_mean')
    )
    Qz_opt_six_s, Qx_opt_six_s, Qy_opt_six_s = (
        A('Qz_opt_six_std'), A('Qx_opt_six_std'), A('Qy_opt_six_std')
    )

    # Colors
    C = {
        'opt_bb84': '#2E86AB',
        'std_bb84': '#A23B72',
        'opt_six':  '#18A999',
        'std_six':  '#F18F01',

        # Standard QBERs
        'Qz_std': '#1f77b4',
        'Qx_std': '#ff7f0e',
        'Qy_std': '#2ca02c',

        # Optimized QBERs
        'Qz_opt': '#003f5c',
        'Qx_opt': '#bc5090',
        'Qy_opt': '#ffa600',
    }

    # Helper for shaded plot
    def shaded(ax, x, m, s, color, label):
        ax.fill_between(x, m - s, m + s, alpha=0.25, color=color)
        ax.plot(x, m, 'o-', color=color, lw=1.6, ms=4, label=label)

    # Layout: 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # --- (1) BB84 SKR ---
    shaded(ax1, eps, opt_bb84_mean, opt_bb84_std, C['opt_bb84'], "Optimized")
    shaded(ax1, eps, std_bb84_mean, std_bb84_std, C['std_bb84'], "Standard")
    ax1.set_title("BB84 Secret Key Rate")
    ax1.set_xlabel(r"$\varepsilon$")
    ax1.set_ylabel("SKR (bits/pulse)")
    ax1.grid(True, ls='--', alpha=0.3)
    ax1.legend()

    # --- (2) Six-State SKR ---
    shaded(ax2, eps, opt_six_mean, opt_six_std, C['opt_six'], "Optimized")
    shaded(ax2, eps, std_six_mean, std_six_std, C['std_six'], "Standard")
    ax2.set_title("Six-State Secret Key Rate")
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel("SKR (bits/pulse)")
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.legend()

    # --- (3) Standard QBERs ---
    shaded(ax3, eps, Qz_std, Qz_std_s, C['Qz_std'], "Qz (std)")
    shaded(ax3, eps, Qx_std, Qx_std_s, C['Qx_std'], "Qx (std)")
    shaded(ax3, eps, Qy_std, Qy_std_s, C['Qy_std'], "Qy (std)")
    ax3.set_title("Standard QBERs")
    ax3.set_xlabel(r"$\varepsilon$")
    ax3.set_ylabel("QBER")
    ax3.grid(True, ls='--', alpha=0.3)
    ax3.legend()

    # --- (4) Optimized QBERs ---
    shaded(ax4, eps, Qz_opt_bb84, Qz_opt_bb84_s, C['Qz_opt'], "Qz (opt BB84)")
    shaded(ax4, eps, Qx_opt_bb84, Qx_opt_bb84_s, C['Qx_opt'], "Qx (opt BB84)")

    shaded(ax4, eps, Qz_opt_six, Qz_opt_six_s, C['Qz_opt'], "Qz (opt Six)")
    shaded(ax4, eps, Qx_opt_six, Qx_opt_six_s, C['Qx_opt'], "Qx (opt Six)")
    shaded(ax4, eps, Qy_opt_six, Qy_opt_six_s, C['Qy_opt'], "Qy (opt Six)")

    ax4.set_title("Optimized QBERs (BB84 + Six-State)")
    ax4.set_xlabel(r"$\varepsilon$")
    ax4.set_ylabel("QBER")
    ax4.grid(True, ls='--', alpha=0.3)
    ax4.legend(ncol=2, fontsize=8)

    plt.tight_layout()

    # Save
    if save_plot:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf = f"data_files/QKD_plot_{ts}.pdf"
        png = f"data_files/QKD_plot_{ts}.png"
        plt.savefig(pdf, bbox_inches='tight')
        plt.savefig(png, dpi=300, bbox_inches='tight')
        print(f"Saved: {pdf}\nSaved: {png}")

    if show_plot:
        plt.show()

    return fig


def plot_combined_comparison(statistics, save_plot=True, show_plot=True):
    """
    Create a combined comparison plot for both protocols.
    """
    
    eps_values = [s['eps'] for s in statistics]
    
    # Extract data
    opt_bb84_mean = [s['opt_skr_bb84_mean'] for s in statistics]
    opt_bb84_std = [s['opt_skr_bb84_std'] for s in statistics]
    opt_six_mean = [s['opt_skr_six_mean'] for s in statistics]
    opt_six_std = [s['opt_skr_six_std'] for s in statistics]
    std_bb84_mean = [s['std_skr_bb84_mean'] for s in statistics]
    std_bb84_std = [s['std_skr_bb84_std'] for s in statistics]
    std_six_mean = [s['std_skr_six_mean'] for s in statistics]
    std_six_std = [s['std_skr_six_std'] for s in statistics]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'opt': '#2E86AB', 'std': '#A23B72', 
              'opt_six': '#18A999', 'std_six': '#F18F01'}
    
    # BB84
    ax.fill_between(eps_values,
                    [opt_bb84_mean[i] - opt_bb84_std[i] for i in range(len(eps_values))],
                    [opt_bb84_mean[i] + opt_bb84_std[i] for i in range(len(eps_values))],
                    alpha=0.2, color=colors['opt'], label='_nolegend_')
    ax.fill_between(eps_values,
                    [std_bb84_mean[i] - std_bb84_std[i] for i in range(len(eps_values))],
                    [std_bb84_mean[i] + std_bb84_std[i] for i in range(len(eps_values))],
                    alpha=0.2, color=colors['std'], label='_nolegend_')
    
    ax.plot(eps_values, opt_bb84_mean, 'o-', color=colors['opt'], 
            linewidth=1.5, markersize=4, label='BB84 (Optimized)')
    ax.plot(eps_values, std_bb84_mean, 's-', color=colors['std'], 
            linewidth=1.5, markersize=4, label='BB84 (Standard)')
    
    # Six-State
    ax.fill_between(eps_values,
                    [opt_six_mean[i] - opt_six_std[i] for i in range(len(eps_values))],
                    [opt_six_mean[i] + opt_six_std[i] for i in range(len(eps_values))],
                    alpha=0.2, color=colors['opt_six'], label='_nolegend_')
    ax.fill_between(eps_values,
                    [std_six_mean[i] - std_six_std[i] for i in range(len(eps_values))],
                    [std_six_mean[i] + std_six_std[i] for i in range(len(eps_values))],
                    alpha=0.2, color=colors['std_six'], label='_nolegend_')
    
    ax.plot(eps_values, opt_six_mean, '^-', color=colors['opt_six'], 
            linewidth=1.5, markersize=5, label='Six-State (Optimized)')
    ax.plot(eps_values, std_six_mean, 'd-', color=colors['std_six'], 
            linewidth=1.5, markersize=5, label='Six-State (Standard)')
    
    ax.set_xlabel('Channel fidelity $\\varepsilon$', fontsize=12)
    ax.set_ylabel('Secret key rate (bits/pulse)', fontsize=12)
    ax.set_title('QKD Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim(eps_values[0] - 0.02, eps_values[-1] + 0.02)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_files/QKD_combined_{timestamp}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        print(f"\nCombined plot saved as: {filename}")
    
    if show_plot:
        plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Parameters
    epss_v = np.linspace(0.4, 1, 10)
    #seeds = np.arange(100)
    n_avg = 100
    rng = np.random.seed(42)
    seeds = np.random.randint(5, 2000, size = n_avg)
    n_workers = 20
    
    print(f"Starting QKD simulations...")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run simulations
    statistics = run_qkd_simulation(epss_v, seeds, n_samples=10000, n_workers=n_workers)
    
    # Save results
    save_results_to_file(statistics)
    
    # Create publication-quality plots
    fig1 = plot_qkd_results_latex(statistics, save_plot=False, show_plot=True)
    fig2 = plot_combined_comparison(statistics, save_plot=False, show_plot=True)
    

    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'eps':<10} {'BB84_opt':<12} {'BB84_std':<12} {'Six_opt':<12} {'Six_std':<12} {'Imp_BB84':<12} {'Imp_Six':<12}")
    print(f"{'-'*80}")
    
    for s in statistics:
        print(f"{s['eps']:<10.4f} {s['opt_skr_bb84_mean']:<12.6f} {s['std_skr_bb84_mean']:<12.6f} "
              f"{s['opt_skr_six_mean']:<12.6f} {s['std_skr_six_mean']:<12.6f} "
              f"{s['improvement_bb84_mean']:<12.6f} {s['improvement_six_mean']:<12.6f}")
    
    print(f"{'='*80}")