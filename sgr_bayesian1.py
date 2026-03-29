#!/usr/bin/env python3
"""
Bayesian Shifted Gamma-Ray Index (SGR)
=======================================
Python implementation of:

  Oughton, R.H., Wooff, D.A. & O'Connor, S.A. (2014).
  A Bayesian shifting method for uncertainty in the open-hole
  gamma-ray log around casing points.
  Petroleum Geoscience, 20, 375–391.
  https://doi.org/10.1144/petgeo2014-006

The method fits a piecewise linear function to the GR log in a depth
window around each casing point using a Gibbs sampler (Markov Chain
Monte Carlo). This produces the Shifted Gamma-Ray Index (SGR), which
accounts for step-changes in the GR log caused by borehole-diameter
changes, mud changes, or different logging tools between casing strings.

Model structure
---------------
For data in the window [z_top, z_bot] around casing point z_cas:

  Zone 1 (z_top  < z < z_dtop):  GR_i ~ N(γ₁,         τ_v⁻¹)
  Zone 2 (z_dtop ≤ z ≤ z_dbot):  GR_i ~ N(γ₁ + θ·f_i,  τ_w⁻¹)   f_i ∈ [0,1]
  Zone 3 (z_dbot < z < z_bot):   GR_i ~ N(γ₁ + θ,      τ_z⁻¹)

Priors:
  τ_v, τ_w, τ_z  ~ Gamma(a, b)
  γ₁ | τ_v       ~ N(μ_p, τ_v⁻¹)
  θ  | τ_z       ~ N(0,   τ_z⁻¹)
  z_dtop          ~ Uniform on depth grid in [z_top + d_min, z_cas)
  z_dbot          ~ Uniform on depth grid in (z_cas, z_bot − d_min]

All conjugate conditionals are derived analytically; discrete depth
parameters are sampled by enumerating their finite support.

Usage
-----
  # Synthetic demo (no LAS file needed):
  python sgr_bayesian.py

  # With a real LAS file:
  python sgr_bayesian.py --las well.las --casings 1000 2500 4000

  # Full options:
  python sgr_bayesian.py --help

Dependencies
------------
  numpy, scipy, matplotlib, pandas
  lasio  (optional – only needed for reading LAS files)

Install:
  pip install numpy scipy matplotlib pandas lasio
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ── Optional LAS I/O ────────────────────────────────────────────────────────
try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False
    warnings.warn(
        "lasio not installed – LAS file loading unavailable. "
        "Install with:  pip install lasio",
        ImportWarning,
        stacklevel=2,
    )


# ============================================================================
# Configuration dataclass
# ============================================================================

@dataclass
class GibbsConfig:
    """
    Configuration for the Gibbs sampler and SGR workflow.

    Parameters
    ----------
    n_iter : int
        Number of MCMC iterations per chain (after burn-in).
    n_burn : int
        Burn-in iterations to discard.
    n_chains : int
        Number of independent parallel chains (for convergence diagnostics).
    thin : int
        Thinning interval (keep every `thin`-th sample).
    d_max : float
        Half-window around casing point (metres).  The algorithm uses data
        in [z_cas − d_max, z_cas + d_max].
    d_min : float
        Minimum length (metres) of each constant segment. This restricts
        how narrow the transition zone can be.
    a : float
        Shape hyperparameter for Gamma priors on precisions τ_v, τ_w, τ_z.
        a = 1, b = 1 gives a non-informative prior (paper default).
    b : float
        Rate hyperparameter for Gamma priors on precisions.
    mu_p : float or None
        Prior mean for γ₁.  Auto-set to the mean GR above z_cas if None.
    max_discrete_candidates : int
        Maximum number of candidate depth values when sampling z_dtop or
        z_dbot. Values are subsampled uniformly if the grid has more points.
        Increase for finer resolution; decrease for speed.
    seed : int
        Base random seed. Chain k uses seed + k*1000.
    """
    n_iter: int = 3000
    n_burn: int = 1000
    n_chains: int = 3
    thin: int = 1
    d_max: float = 40.0
    d_min: float = 3.0
    a: float = 1.0
    b: float = 1.0
    mu_p: Optional[float] = None
    max_discrete_candidates: int = 200
    seed: int = 42


# ============================================================================
# Posterior samples container
# ============================================================================

@dataclass
class PosteriorSamples:
    """
    Posterior samples from one run of the Gibbs sampler (all chains pooled).

    Attributes
    ----------
    gamma1, theta, z_dtop, z_dbot, tau_v, tau_w, tau_z : np.ndarray
        1-D arrays of posterior draws.
    casing_depth : float
        The casing point z_cas for which this was computed.
    n_chains : int
        Number of chains (for reference).
    """
    gamma1: np.ndarray
    theta: np.ndarray
    z_dtop: np.ndarray
    z_dbot: np.ndarray
    tau_v: np.ndarray
    tau_w: np.ndarray
    tau_z: np.ndarray
    casing_depth: float
    n_chains: int = 3

    # ── convenience accessors ────────────────────────────────────────────────

    @property
    def theta_mean(self) -> float:
        return float(np.mean(self.theta))

    @property
    def z_dtop_mean(self) -> float:
        return float(np.mean(self.z_dtop))

    @property
    def z_dbot_mean(self) -> float:
        return float(np.mean(self.z_dbot))

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of posterior summary statistics (Table 1 style)."""
        params = {
            "gamma1": self.gamma1,
            "theta":  self.theta,
            "z_dtop": self.z_dtop,
            "z_dbot": self.z_dbot,
            "tau_v":  self.tau_v,
            "tau_w":  self.tau_w,
            "tau_z":  self.tau_z,
        }
        rows = []
        for name, s in params.items():
            rows.append({
                "Parameter": name,
                "Mean":  float(np.mean(s)),
                "SD":    float(np.std(s)),
                "Min":   float(np.min(s)),
                "Max":   float(np.max(s)),
                "2.5%":  float(np.percentile(s, 2.5)),
                "97.5%": float(np.percentile(s, 97.5)),
            })
        return pd.DataFrame(rows).set_index("Parameter").round(4)


# ============================================================================
# Data loading
# ============================================================================

def load_las(filepath: str | Path, curves: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load a LAS file into a DataFrame indexed by measured depth.

    Parameters
    ----------
    filepath : str or Path
        Path to the .las file.
    curves : list of str, optional
        Subset of curves to load.  Loads all if None.

    Returns
    -------
    pd.DataFrame
        Depth-indexed DataFrame of log curves.
    """
    if not HAS_LASIO:
        raise ImportError(
            "lasio is required for LAS I/O. Install with:  pip install lasio"
        )
    las = lasio.read(str(filepath))
    df = las.df()
    df.index.name = "DEPTH"
    if curves:
        available = [c for c in curves if c in df.columns]
        missing   = [c for c in curves if c not in df.columns]
        if missing:
            warnings.warn(f"Curves not found in LAS file: {missing}", UserWarning)
        df = df[available]
    print(f"Loaded: {filepath}")
    print(f"  Depth range : {df.index.min():.1f} – {df.index.max():.1f} m")
    print(f"  Curves      : {list(df.columns)}")
    return df


def load_synthetic_data(seed: int = 0) -> Tuple[pd.DataFrame, List[float]]:
    """
    Generate synthetic GR data with known step-changes at casing points.
    Useful for testing without a real well.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with 'GR' column indexed by depth.
    casing_depths : list of float
    """
    rng = np.random.default_rng(seed)
    depth = np.arange(500.0, 5001.0, 0.5)
    n = len(depth)

    casing_depths = [1000.0, 2500.0, 4000.0]
    shifts        = [18.0, -30.0, 42.0]
    trans_widths  = [15.0,  10.0, 20.0]

    # Baseline shale GR with mild compaction trend + noise
    gr = 70.0 + 0.002 * depth + rng.normal(0.0, 6.0, n)

    # Impose gradual transitions at each casing point
    for z_cas, shift, tw in zip(casing_depths, shifts, trans_widths):
        z_trans_top = z_cas - 2.0
        z_trans_bot = z_cas + tw
        idx = np.where(depth > z_trans_top)[0]
        frac = np.clip((depth[idx] - z_trans_top) / (z_trans_bot - z_trans_top), 0.0, 1.0)
        gr[idx] += shift * frac

    # Insert some sand (low-GR) intervals
    for z0, z1 in [(1500, 1600), (3000, 3100), (3500, 3600)]:
        mask = (depth >= z0) & (depth <= z1)
        gr[mask] = 30.0 + rng.normal(0.0, 5.0, mask.sum())

    gr = np.maximum(gr, 5.0)

    df = pd.DataFrame({"GR": gr}, index=depth)
    df.index.name = "DEPTH"

    print("Synthetic well created:")
    for z, s in zip(casing_depths, shifts):
        print(f"  Casing at {z:.0f} m  |  imposed GR shift = {s:+.1f} API")
    return df, casing_depths


# ============================================================================
# Piecewise linear curve
# ============================================================================

def piecewise_linear(
    depth:  np.ndarray,
    gamma1: float,
    theta:  float,
    z_dtop: float,
    z_dbot: float,
) -> np.ndarray:
    """
    Evaluate the piecewise linear mean curve (Fig. 1 in the paper).

    Parameters
    ----------
    depth  : 1-D array of depths
    gamma1 : mean GR in Zone 1
    theta  : total shift (Zone 3 level = gamma1 + theta)
    z_dtop : top of transition zone
    z_dbot : bottom of transition zone

    Returns
    -------
    mu : 1-D array of mean GR values
    """
    mu = np.full_like(depth, gamma1, dtype=float)
    if z_dbot > z_dtop:
        mask_trans = (depth >= z_dtop) & (depth <= z_dbot)
        frac = (depth[mask_trans] - z_dtop) / (z_dbot - z_dtop)
        mu[mask_trans] = gamma1 + theta * frac
    mu[depth > z_dbot] = gamma1 + theta
    return mu


# ============================================================================
# Conjugate Gibbs sampler – full conditional samplers
# ============================================================================

def _sample_gamma1(
    gr: np.ndarray, depth: np.ndarray,
    theta: float, tau_v: float, tau_w: float, tau_z: float,
    z_dtop: float, z_dbot: float, mu_p: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample γ₁ from its Normal full conditional.

    Prior: γ₁ | τ_v ~ N(μ_p, τ_v⁻¹)
    All three zones contribute to the posterior.
    """
    m1 = depth < z_dtop
    m2 = (depth >= z_dtop) & (depth <= z_dbot)
    m3 = depth > z_dbot

    prec     = tau_v                      # from prior
    mean_num = tau_v * mu_p               # from prior

    # Zone 1: GR_i ~ N(γ₁, τ_v⁻¹)
    if m1.any():
        prec     += tau_v * m1.sum()
        mean_num += tau_v * gr[m1].sum()

    # Zone 2: GR_i ~ N(γ₁ + θ·f_i, τ_w⁻¹)  →  residual = GR_i − θ·f_i
    if m2.any() and z_dbot > z_dtop:
        frac      = (depth[m2] - z_dtop) / (z_dbot - z_dtop)
        r2        = gr[m2] - theta * frac
        prec     += tau_w * m2.sum()
        mean_num += tau_w * r2.sum()

    # Zone 3: GR_i ~ N(γ₁ + θ, τ_z⁻¹)  →  residual = GR_i − θ
    if m3.any():
        r3        = gr[m3] - theta
        prec     += tau_z * m3.sum()
        mean_num += tau_z * r3.sum()

    return rng.normal(mean_num / prec, 1.0 / np.sqrt(prec))


def _sample_theta(
    gr: np.ndarray, depth: np.ndarray,
    gamma1: float, tau_w: float, tau_z: float,
    z_dtop: float, z_dbot: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample θ from its Normal full conditional.

    Prior: θ | τ_z ~ N(0, τ_z⁻¹)
    Zones 2 and 3 contribute to the posterior.
    """
    m2 = (depth >= z_dtop) & (depth <= z_dbot)
    m3 = depth > z_dbot

    prec     = tau_z      # from prior (mean 0)
    mean_num = 0.0

    # Zone 2 contribution:  ∂/∂θ  τ_w · Σ f_i·(GR_i − γ₁) − τ_w · Σ f_i² · θ
    if m2.any() and z_dbot > z_dtop:
        frac      = (depth[m2] - z_dtop) / (z_dbot - z_dtop)
        prec     += tau_w * np.sum(frac ** 2)
        mean_num += tau_w * np.sum((gr[m2] - gamma1) * frac)

    # Zone 3 contribution:  ∂/∂θ  τ_z · Σ(GR_i − γ₁) − τ_z · n₃ · θ
    if m3.any():
        prec     += tau_z * m3.sum()
        mean_num += tau_z * np.sum(gr[m3] - gamma1)

    return rng.normal(mean_num / prec, 1.0 / np.sqrt(prec))


def _sample_tau_v(
    gr: np.ndarray, depth: np.ndarray,
    gamma1: float, z_dtop: float, mu_p: float,
    a: float, b: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample τ_v (Zone 1 precision) from its Gamma full conditional.

    τ_v appears in both the zone-1 likelihood and the prior for γ₁, so:
      Posterior: Gamma(a + (n₁ + 1)/2,  b + (SS₁ + (γ₁ − μ_p)²) / 2)
    """
    m1 = depth < z_dtop
    n1 = m1.sum()
    ss1 = np.sum((gr[m1] - gamma1) ** 2) if n1 > 0 else 0.0

    shape = a + (n1 + 1) / 2.0
    rate  = b + (ss1 + (gamma1 - mu_p) ** 2) / 2.0
    return rng.gamma(shape, 1.0 / rate)


def _sample_tau_w(
    gr: np.ndarray, depth: np.ndarray,
    gamma1: float, theta: float,
    z_dtop: float, z_dbot: float,
    a: float, b: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample τ_w (transition-zone precision) from its Gamma full conditional.

      Posterior: Gamma(a + n₂/2,  b + SS₂/2)
    """
    m2 = (depth >= z_dtop) & (depth <= z_dbot)
    n2 = m2.sum()

    if n2 > 0 and z_dbot > z_dtop:
        frac = (depth[m2] - z_dtop) / (z_dbot - z_dtop)
        mu2  = gamma1 + theta * frac
        ss2  = np.sum((gr[m2] - mu2) ** 2)
    else:
        ss2 = 0.0

    shape = max(a + n2 / 2.0, 1e-6)
    rate  = max(b + ss2 / 2.0, 1e-12)
    return rng.gamma(shape, 1.0 / rate)


def _sample_tau_z(
    gr: np.ndarray, depth: np.ndarray,
    gamma1: float, theta: float, z_dbot: float,
    a: float, b: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample τ_z (Zone 3 precision) from its Gamma full conditional.

    τ_z appears in both the zone-3 likelihood and the prior for θ, so:
      Posterior: Gamma(a + (n₃ + 1)/2,  b + (SS₃ + θ²) / 2)
    """
    m3 = depth > z_dbot
    n3 = m3.sum()
    ss3 = np.sum((gr[m3] - gamma1 - theta) ** 2) if n3 > 0 else 0.0

    shape = a + (n3 + 1) / 2.0
    rate  = b + (ss3 + theta ** 2) / 2.0
    return rng.gamma(shape, 1.0 / rate)


def _sample_z_dtop(
    gr: np.ndarray, depth: np.ndarray,
    gamma1: float, theta: float, tau_v: float, tau_w: float,
    z_top: float, z_cas: float, d_min: float, z_dbot: float,
    max_candidates: int,
    rng: np.random.Generator,
) -> float:
    """
    Sample z_dtop from its discrete full conditional.

    Valid range: [z_top + d_min,  min(z_cas, z_dbot))
    Log probability ∝ zone-1 + zone-2 log-likelihood.
    """
    candidates = depth[
        (depth >= z_top + d_min) &
        (depth <  z_cas) &
        (depth <  z_dbot)
    ]
    if len(candidates) == 0:
        return z_cas - d_min

    # Sub-sample if grid is very fine to keep iterations fast
    if len(candidates) > max_candidates:
        idx = np.round(np.linspace(0, len(candidates) - 1, max_candidates)).astype(int)
        candidates = candidates[idx]

    r        = gr - gamma1                # residuals w.r.t. gamma1  (shape = n)
    log_prob = np.empty(len(candidates))

    for k, z_dt in enumerate(candidates):
        m1 = depth < z_dt
        m2 = (depth >= z_dt) & (depth <= z_dbot)
        ll = 0.0
        if m1.any():
            ll += 0.5 * m1.sum() * np.log(tau_v) - 0.5 * tau_v * np.sum(r[m1] ** 2)
        if m2.any() and z_dbot > z_dt:
            frac = (depth[m2] - z_dt) / (z_dbot - z_dt)
            res2 = r[m2] - theta * frac
            ll  += 0.5 * m2.sum() * np.log(tau_w) - 0.5 * tau_w * np.sum(res2 ** 2)
        log_prob[k] = ll

    # Numerically stable categorical draw
    log_prob -= log_prob.max()
    prob = np.exp(log_prob)
    prob /= prob.sum()
    return candidates[rng.choice(len(candidates), p=prob)]


def _sample_z_dbot(
    gr: np.ndarray, depth: np.ndarray,
    gamma1: float, theta: float, tau_w: float, tau_z: float,
    z_cas: float, z_bot: float, d_min: float, z_dtop: float,
    max_candidates: int,
    rng: np.random.Generator,
) -> float:
    """
    Sample z_dbot from its discrete full conditional.

    Valid range: (max(z_cas, z_dtop),  z_bot − d_min]
    Log probability ∝ zone-2 + zone-3 log-likelihood.
    """
    candidates = depth[
        (depth >  z_cas) &
        (depth >  z_dtop) &
        (depth <= z_bot - d_min)
    ]
    if len(candidates) == 0:
        return z_cas + d_min

    if len(candidates) > max_candidates:
        idx = np.round(np.linspace(0, len(candidates) - 1, max_candidates)).astype(int)
        candidates = candidates[idx]

    r        = gr - gamma1
    log_prob = np.empty(len(candidates))

    for k, z_db in enumerate(candidates):
        m2 = (depth >= z_dtop) & (depth <= z_db)
        m3 = depth > z_db
        ll = 0.0
        if m2.any() and z_db > z_dtop:
            frac = (depth[m2] - z_dtop) / (z_db - z_dtop)
            res2 = r[m2] - theta * frac
            ll  += 0.5 * m2.sum() * np.log(tau_w) - 0.5 * tau_w * np.sum(res2 ** 2)
        if m3.any():
            ll  += 0.5 * m3.sum() * np.log(tau_z) - 0.5 * tau_z * np.sum((r[m3] - theta) ** 2)
        log_prob[k] = ll

    log_prob -= log_prob.max()
    prob = np.exp(log_prob)
    prob /= prob.sum()
    return candidates[rng.choice(len(candidates), p=prob)]


# ============================================================================
# Gibbs sampler – full run for one casing point
# ============================================================================

def run_gibbs_sampler(
    gr:     np.ndarray,
    depth:  np.ndarray,
    z_cas:  float,
    config: GibbsConfig,
) -> PosteriorSamples:
    """
    Run the Gibbs sampler for a single casing point.

    Parameters
    ----------
    gr    : GR values in the depth window (no NaNs)
    depth : corresponding depths (sorted ascending)
    z_cas : casing point depth
    config : GibbsConfig

    Returns
    -------
    PosteriorSamples  (all chains pooled after burn-in)
    """
    z_top = z_cas - config.d_max
    z_bot = z_cas + config.d_max
    d_min = config.d_min
    mu_p  = config.mu_p if config.mu_p is not None else float(
        np.mean(gr[depth < z_cas]) if np.any(depth < z_cas) else gr.mean()
    )

    # Storage across all chains
    store: Dict[str, list] = {k: [] for k in
                               ("gamma1","theta","z_dtop","z_dbot","tau_v","tau_w","tau_z")}

    for chain in range(config.n_chains):
        rng = np.random.default_rng(config.seed + chain * 1000)

        # ── Initialise parameters ────────────────────────────────────────────
        gamma1 = float(np.mean(gr[depth < z_cas])) if np.any(depth < z_cas) else gr.mean()
        theta  = 0.0
        tau_v  = tau_w = tau_z = 1.0

        # Initial transition zone: narrow band around z_cas
        z_dtop = float(np.max(depth[(depth < z_cas) & (depth >= z_top + d_min)],
                              initial=z_top + d_min))
        z_dbot = float(np.min(depth[(depth > z_cas) & (depth <= z_bot - d_min)],
                              initial=z_bot - d_min))

        n_total = config.n_burn + config.n_iter
        chain_store: Dict[str, list] = {k: [] for k in store}

        for it in range(n_total):
            # ── Continuous parameters (conjugate) ────────────────────────────
            gamma1 = _sample_gamma1(gr, depth, theta, tau_v, tau_w, tau_z,
                                    z_dtop, z_dbot, mu_p, rng)
            theta  = _sample_theta(gr, depth, gamma1, tau_w, tau_z,
                                   z_dtop, z_dbot, rng)
            tau_v  = _sample_tau_v(gr, depth, gamma1, z_dtop, mu_p,
                                   config.a, config.b, rng)
            tau_w  = _sample_tau_w(gr, depth, gamma1, theta, z_dtop, z_dbot,
                                   config.a, config.b, rng)
            tau_z  = _sample_tau_z(gr, depth, gamma1, theta, z_dbot,
                                   config.a, config.b, rng)

            # ── Discrete depth parameters ─────────────────────────────────────
            z_dtop = _sample_z_dtop(gr, depth, gamma1, theta, tau_v, tau_w,
                                    z_top, z_cas, d_min, z_dbot,
                                    config.max_discrete_candidates, rng)
            z_dbot = _sample_z_dbot(gr, depth, gamma1, theta, tau_w, tau_z,
                                    z_cas, z_bot, d_min, z_dtop,
                                    config.max_discrete_candidates, rng)

            # ── Store after burn-in ───────────────────────────────────────────
            if it >= config.n_burn and (it - config.n_burn) % config.thin == 0:
                chain_store["gamma1"].append(gamma1)
                chain_store["theta"].append(theta)
                chain_store["z_dtop"].append(z_dtop)
                chain_store["z_dbot"].append(z_dbot)
                chain_store["tau_v"].append(tau_v)
                chain_store["tau_w"].append(tau_w)
                chain_store["tau_z"].append(tau_z)

        for k in store:
            store[k].extend(chain_store[k])

    return PosteriorSamples(
        gamma1       = np.array(store["gamma1"]),
        theta        = np.array(store["theta"]),
        z_dtop       = np.array(store["z_dtop"]),
        z_dbot       = np.array(store["z_dbot"]),
        tau_v        = np.array(store["tau_v"]),
        tau_w        = np.array(store["tau_w"]),
        tau_z        = np.array(store["tau_z"]),
        casing_depth = z_cas,
        n_chains     = config.n_chains,
    )


# ============================================================================
# SGR computation – full well
# ============================================================================

def compute_sgr(
    gr_raw:        np.ndarray,
    depth:         np.ndarray,
    casing_depths: List[float],
    config:        Optional[GibbsConfig] = None,
) -> Tuple[np.ndarray, List[PosteriorSamples]]:
    """
    Compute the Shifted Gamma-Ray Index (SGR) for a full well.

    For each casing point (processed in ascending depth order):
      1. Extract the GR window.
      2. Run the Gibbs sampler to estimate θ, z_dtop, z_dbot.
      3. Remove the transition zone (set to NaN).
      4. Shift all data below z_dbot by −θ_mean.

    After all casing points the working GR is rescaled to [0, 1]:
        SGR = (GR_shifted − GR_min) / (GR_max − GR_min)

    Parameters
    ----------
    gr_raw        : raw GR values (API), same length as depth
    depth         : measured depth array (m), sorted ascending
    casing_depths : list of casing-point depths (m)
    config        : GibbsConfig (uses defaults if None)

    Returns
    -------
    sgr       : SGR values in [0, 1]  (NaN in transition zones)
    posteriors : list of PosteriorSamples, one per processed casing point
    """
    if config is None:
        config = GibbsConfig()

    gr_work    = gr_raw.copy().astype(float)
    posteriors = []

    for i, z_cas in enumerate(sorted(casing_depths)):
        print(f"\nCasing point {i+1}/{len(casing_depths)} at {z_cas:.1f} m ...")

        z_top = z_cas - config.d_max
        z_bot = z_cas + config.d_max
        mask_win = (depth >= z_top) & (depth <= z_bot) & np.isfinite(gr_work)

        if mask_win.sum() < 10:
            print(f"  Only {mask_win.sum()} valid points in window – skipping.")
            continue

        gr_win  = gr_work[mask_win]
        dep_win = depth[mask_win]

        # Auto-compute mu_p for this window if not set globally
        mu_p_local = config.mu_p
        if mu_p_local is None:
            above = gr_win[dep_win < z_cas]
            mu_p_local = float(np.mean(above)) if len(above) > 0 else float(np.mean(gr_win))

        local_cfg       = GibbsConfig(**{**config.__dict__, "mu_p": mu_p_local})
        posterior       = run_gibbs_sampler(gr_win, dep_win, z_cas, local_cfg)
        posteriors.append(posterior)

        theta_mean  = posterior.theta_mean
        z_dtop_mean = posterior.z_dtop_mean
        z_dbot_mean = posterior.z_dbot_mean

        print(f"  θ  = {theta_mean:+.2f} ± {np.std(posterior.theta):.2f} API")
        print(f"  z_dtop = {z_dtop_mean:.1f} m   z_dbot = {z_dbot_mean:.1f} m")

        # Remove transition zone
        mask_trans = (depth >= z_dtop_mean) & (depth <= z_dbot_mean)
        gr_work[mask_trans] = np.nan

        # Shift all data below the transition zone
        gr_work[depth > z_dbot_mean] -= theta_mean

    # Rescale to [0, 1]
    gr_min = np.nanmin(gr_work)
    gr_max = np.nanmax(gr_work)
    sgr = (gr_work - gr_min) / (gr_max - gr_min) if gr_max > gr_min else np.zeros_like(gr_work)

    return sgr, posteriors


# ============================================================================
# Validation helpers
# ============================================================================

def validate_casing_zone(
    df:          pd.DataFrame,
    z_cas:       float,
    window:      float = 40.0,
    log_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Test whether other wireline logs show a statistically significant
    shift at the casing point (Welch t-test, above vs. below).

    A significant result (p < 0.05) suggests the GR shift may reflect
    a true lithological change rather than a tool artefact, i.e. the
    assumption of homogeneous shale around the casing point may be
    violated.

    Parameters
    ----------
    df          : DataFrame of wireline logs, depth-indexed
    z_cas       : casing depth
    window      : half-window for sampling (m)
    log_columns : columns to test (auto-selected if None)

    Returns
    -------
    pd.DataFrame with one row per log tested.
    """
    if log_columns is None:
        exclude = {"GR", "DEPTH", "MD", "TVD", "CALI", "CAL"}
        log_columns = [c for c in df.columns if c.upper() not in exclude]

    depth = df.index.values
    mask  = (depth >= z_cas - window) & (depth <= z_cas + window)

    rows = []
    for col in log_columns:
        if col not in df.columns:
            continue
        vals = df.loc[mask, col].values
        deps = depth[mask]
        ok   = np.isfinite(vals)
        if ok.sum() < 6:
            continue

        above = vals[ok & (deps < z_cas)]
        below = vals[ok & (deps >= z_cas)]
        if len(above) < 3 or len(below) < 3:
            continue

        t_stat, p_val = stats.ttest_ind(above, below, equal_var=False)
        rows.append({
            "Log":            col,
            "Mean above":     float(np.mean(above)),
            "Mean below":     float(np.mean(below)),
            "Shift":          float(np.mean(below) - np.mean(above)),
            "t-statistic":    float(t_stat),
            "p-value":        float(p_val),
            "Significant?":   p_val < 0.05,
        })

    return pd.DataFrame(rows).set_index("Log") if rows else pd.DataFrame()


# ============================================================================
# Plotting
# ============================================================================

def plot_casing_point_fit(
    gr:        np.ndarray,
    depth:     np.ndarray,
    posterior: PosteriorSamples,
    n_curves:  int = 100,
    ax_left:   Optional[plt.Axes] = None,
    ax_right:  Optional[plt.Axes] = None,
) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plot fitted posterior curves and the theta posterior density
    (reproduces Figure 4 / 7 / 10 style from the paper).
    """
    z_cas = posterior.casing_depth
    if ax_left is None or ax_right is None:
        _, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5))

    # ── Left panel: data + sampled fitted curves ─────────────────────────────
    ax_left.scatter(depth, gr, s=10, c="k", alpha=0.5, zorder=3, label="GR data")
    ax_left.axvline(z_cas, color="k", lw=1.5, zorder=4, label=f"z_cas = {z_cas:.0f} m")

    n_samp  = len(posterior.theta)
    indices = np.random.default_rng(0).choice(n_samp, size=min(n_curves, n_samp), replace=False)
    d_fine  = np.linspace(depth.min(), depth.max(), 600)
    for idx in indices:
        mu = piecewise_linear(d_fine,
                              posterior.gamma1[idx], posterior.theta[idx],
                              posterior.z_dtop[idx], posterior.z_dbot[idx])
        ax_left.plot(d_fine, mu, color="steelblue", alpha=0.05, lw=0.8)

    # Mean curve
    mu_mean = piecewise_linear(d_fine,
                                np.mean(posterior.gamma1), np.mean(posterior.theta),
                                np.mean(posterior.z_dtop), np.mean(posterior.z_dbot))
    ax_left.plot(d_fine, mu_mean, color="navy", lw=1.5, label="Posterior mean")
    ax_left.set_xlabel("Depth (m)")
    ax_left.set_ylabel("GR (API)")
    ax_left.set_title(f"Casing at {z_cas:.0f} m – fitted curves")
    ax_left.legend(fontsize=7)

    # ── Right panel: posterior density of θ ──────────────────────────────────
    ax_right.hist(posterior.theta, bins=50, density=True,
                  color="steelblue", alpha=0.7, edgecolor="white")
    ax_right.axvline(np.mean(posterior.theta),
                     color="k", lw=2, label=f"Mean = {np.mean(posterior.theta):.2f}")
    ax_right.axvline(np.percentile(posterior.theta, 2.5),
                     color="k", lw=1, ls="--", label="95% CI")
    ax_right.axvline(np.percentile(posterior.theta, 97.5),
                     color="k", lw=1, ls="--")
    ax_right.set_xlabel("θ (API)")
    ax_right.set_ylabel("Density")
    ax_right.set_title("Posterior of shift θ")
    ax_right.legend(fontsize=8)

    return ax_left, ax_right


def plot_sgr_comparison(
    gr_raw:        np.ndarray,
    sgr:           np.ndarray,
    depth:         np.ndarray,
    casing_depths: List[float],
    title:         str = "",
) -> plt.Figure:
    """
    Side-by-side plot of raw GR and SGR vs depth
    (reproduces Figures 6, 9, 11 from the paper).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 14), sharey=True)
    if title:
        fig.suptitle(title, fontsize=12)

    ax1.plot(gr_raw, depth, "k-", lw=0.5)
    for z in casing_depths:
        ax1.axhline(z, color="grey", lw=1)
    ax1.set_xlabel("GR (API)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Raw GR")
    ax1.invert_yaxis()

    valid = np.isfinite(sgr)
    ax2.plot(sgr[valid], depth[valid], "k-", lw=0.5)
    for z in casing_depths:
        ax2.axhline(z, color="grey", lw=1)
    ax2.set_xlabel("SGR")
    ax2.set_title("Shifted GR Index (SGR)")
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="grey", lw=0.5, ls=":")  # reference

    plt.tight_layout()
    return fig


def plot_all_casing_points(
    gr_raw:        np.ndarray,
    depth:         np.ndarray,
    casing_depths: List[float],
    posteriors:    List[PosteriorSamples],
    config:        GibbsConfig,
    output_path:   str = "sgr_posteriors.png",
) -> None:
    """Plot fitted curves and theta densities for every casing point."""
    n = len(posteriors)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (post, z_cas) in enumerate(zip(posteriors, sorted(casing_depths))):
        z_top = z_cas - config.d_max
        z_bot = z_cas + config.d_max
        mask  = (depth >= z_top) & (depth <= z_bot) & np.isfinite(gr_raw)
        plot_casing_point_fit(
            gr_raw[mask], depth[mask], post,
            ax_left=axes[i, 0], ax_right=axes[i, 1],
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.show()


# ============================================================================
# High-level workflow
# ============================================================================

def process_well(
    las_path:       Optional[str]   = None,
    gr_column:      str             = "GR",
    casing_depths:  Optional[List[float]] = None,
    config:         Optional[GibbsConfig] = None,
    output_csv:     str             = "sgr_output.csv",
) -> Tuple[pd.DataFrame, List[PosteriorSamples]]:
    """
    End-to-end SGR workflow.

    1. Load data (LAS file or synthetic demo).
    2. Validate lithological consistency at casing points.
    3. Run the Bayesian Gibbs sampler at each casing point.
    4. Compute and save SGR.
    5. Generate diagnostic plots.

    Parameters
    ----------
    las_path      : path to LAS file (uses synthetic data if None)
    gr_column     : name of GR curve in the LAS file
    casing_depths : list of casing depths in metres
    config        : GibbsConfig
    output_csv    : path for output CSV

    Returns
    -------
    df         : input DataFrame with 'SGR' column appended
    posteriors : list of PosteriorSamples
    """
    if config is None:
        config = GibbsConfig()

    # ── Load ─────────────────────────────────────────────────────────────────
    if las_path is not None:
        df = load_las(las_path)
        if gr_column not in df.columns:
            raise ValueError(
                f"Column '{gr_column}' not in LAS file. "
                f"Available: {list(df.columns)}"
            )
        if casing_depths is None:
            raise ValueError("casing_depths must be provided with a real LAS file.")
    else:
        print("No LAS file provided – running synthetic demonstration.\n")
        df, casing_depths = load_synthetic_data()

    depth  = df.index.values.astype(float)
    gr_raw = df[gr_column].values.astype(float)

    # ── Filter valid casing depths ────────────────────────────────────────────
    valid_casings = [
        z for z in sorted(casing_depths)
        if depth.min() + config.d_max < z < depth.max() - config.d_max
    ]
    skipped = set(casing_depths) - set(valid_casings)
    if skipped:
        print(f"\nWarning: casing depths {skipped} skipped (outside usable depth range).")

    # ── Validate lithological consistency ────────────────────────────────────
    print("\n── Validating lithological consistency at casing points ──")
    for z_cas in valid_casings:
        results = validate_casing_zone(df, z_cas, window=config.d_max)
        if results.empty:
            print(f"  z={z_cas:.0f} m: no validation logs available")
        else:
            for log, row in results.iterrows():
                flag = "⚠ SIGNIFICANT" if row["Significant?"] else "OK"
                print(f"  z={z_cas:.0f} m | {log}: "
                      f"shift={row['Shift']:+.3f}, p={row['p-value']:.3f} [{flag}]")

    # ── Compute SGR ───────────────────────────────────────────────────────────
    print("\n── Running Bayesian SGR computation ──")
    sgr, posteriors = compute_sgr(gr_raw, depth, valid_casings, config)
    df["SGR"] = sgr

    # ── Summaries ─────────────────────────────────────────────────────────────
    print("\n── Posterior parameter summaries ──")
    for post in posteriors:
        print(f"\n  Casing at {post.casing_depth:.0f} m:")
        print(post.summary().to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    if posteriors:
        plot_all_casing_points(gr_raw, depth, valid_casings, posteriors, config)
        fig = plot_sgr_comparison(
            gr_raw, sgr, depth, valid_casings,
            title="Bayesian SGR – Oughton et al. (2014)",
        )
        fig.savefig("sgr_comparison.png", dpi=150, bbox_inches="tight")
        print("Saved: sgr_comparison.png")
        plt.show()

    # ── Export ────────────────────────────────────────────────────────────────
    df.to_csv(output_csv)
    print(f"\nResults written to: {output_csv}")
    return df, posteriors


# ============================================================================
# Command-line interface
# ============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Bayesian Shifted Gamma-Ray Index (SGR)\n"
            "Oughton, Wooff & O'Connor (2014), Petroleum Geoscience, 20, 375-391"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--las",      metavar="FILE",   default=None,
                   help="Path to LAS file (omit for synthetic demo)")
    p.add_argument("--gr-col",   metavar="NAME",   default="GR",
                   help="GR curve name in LAS file (default: GR)")
    p.add_argument("--casings",  metavar="Z",      type=float, nargs="+", default=None,
                   help="Casing depths in metres")
    p.add_argument("--dmax",     metavar="M",      type=float, default=40.0,
                   help="Window half-width around casing point in metres (default: 40)")
    p.add_argument("--dmin",     metavar="M",      type=float, default=3.0,
                   help="Min constant-segment length in metres (default: 3)")
    p.add_argument("--n-iter",   metavar="N",      type=int,   default=3000,
                   help="MCMC iterations per chain after burn-in (default: 3000)")
    p.add_argument("--n-burn",   metavar="N",      type=int,   default=1000,
                   help="Burn-in iterations (default: 1000)")
    p.add_argument("--n-chains", metavar="N",      type=int,   default=3,
                   help="Number of parallel chains (default: 3)")
    p.add_argument("--thin",     metavar="K",      type=int,   default=1,
                   help="Thinning interval (default: 1)")
    p.add_argument("--output",   metavar="FILE",   default="sgr_output.csv",
                   help="Output CSV path (default: sgr_output.csv)")
    p.add_argument("--seed",     metavar="INT",    type=int,   default=42,
                   help="Random seed (default: 42)")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    cfg = GibbsConfig(
        n_iter    = args.n_iter,
        n_burn    = args.n_burn,
        n_chains  = args.n_chains,
        thin      = args.thin,
        d_max     = args.dmax,
        d_min     = args.dmin,
        seed      = args.seed,
    )

    process_well(
        las_path      = args.las,
        gr_column     = args.gr_col,
        casing_depths = args.casings,
        config        = cfg,
        output_csv    = args.output,
    )
