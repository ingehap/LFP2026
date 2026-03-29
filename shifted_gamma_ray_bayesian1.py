#!/usr/bin/env python3
"""
shifted_gamma_ray_bayesian.py

Bayesian Shifted Gamma-Ray workflow after Oughton, Wooff & O'Connor (2014),
with improvements inspired by an alternative Python implementation.

Key points
----------
- Supports LAS, CSV, and Parquet inputs.
- Uses an analytic Gibbs sampler with conjugate full conditionals.
- Samples z_dtop and z_dbot on the observed depth grid by discrete enumeration.
- Applies sequential casing corrections to build shifted GR and SGR.
- Exports posterior summaries, corrected logs, validation tables, and plots.
- Avoids heavy PyMC/JAGS dependencies; requires only numpy/scipy/pandas/matplotlib.

Install
-------
pip install numpy pandas scipy matplotlib lasio pyarrow

Example
-------
python shifted_gamma_ray_bayesian.py \
    --input well.las \
    --gr GR \
    --casing-points 1387 2700 4312 \
    --output-dir sgr_output
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import lasio
except Exception:
    lasio = None


# -----------------------------------------------------------------------------
# Configuration / containers
# -----------------------------------------------------------------------------

@dataclass
class GibbsConfig:
    n_iter: int = 3000
    n_burn: int = 1000
    n_chains: int = 3
    thin: int = 1
    d_max: float = 40.0
    d_min: float = 3.0
    a: float = 1.0
    b: float = 1.0
    mu_p: Optional[float] = None
    max_discrete_candidates: int = 300
    seed: int = 42
    min_points: int = 40


@dataclass
class PosteriorSamples:
    gamma1: np.ndarray
    theta: np.ndarray
    z_dtop: np.ndarray
    z_dbot: np.ndarray
    tau_v: np.ndarray
    tau_w: np.ndarray
    tau_z: np.ndarray
    casing_depth: float
    n_chains: int

    @property
    def theta_mean(self) -> float:
        return float(np.mean(self.theta))

    @property
    def theta_ci_95(self) -> Tuple[float, float]:
        return float(np.percentile(self.theta, 2.5)), float(np.percentile(self.theta, 97.5))

    @property
    def z_dtop_mean(self) -> float:
        return float(np.mean(self.z_dtop))

    @property
    def z_dbot_mean(self) -> float:
        return float(np.mean(self.z_dbot))

    def summary_frame(self) -> pd.DataFrame:
        params = {
            "gamma1": self.gamma1,
            "theta": self.theta,
            "z_dtop": self.z_dtop,
            "z_dbot": self.z_dbot,
            "tau_v": self.tau_v,
            "tau_w": self.tau_w,
            "tau_z": self.tau_z,
        }
        rows = []
        for name, s in params.items():
            rows.append(
                {
                    "parameter": name,
                    "mean": float(np.mean(s)),
                    "sd": float(np.std(s, ddof=1)),
                    "p2_5": float(np.percentile(s, 2.5)),
                    "p50": float(np.percentile(s, 50)),
                    "p97_5": float(np.percentile(s, 97.5)),
                    "min": float(np.min(s)),
                    "max": float(np.max(s)),
                }
            )
        return pd.DataFrame(rows)


@dataclass
class CasingFitResult:
    zcas: float
    ztop: float
    zbot: float
    zdtop_mean: float
    zdbot_mean: float
    gamma1_mean: float
    theta_mean: float
    theta_ci_low: float
    theta_ci_high: float
    tau_v_mean: float
    tau_w_mean: float
    tau_z_mean: float
    n_points: int
    kept_points: int
    status: str
    note: str = ""


# -----------------------------------------------------------------------------
# Data I/O
# -----------------------------------------------------------------------------

def _guess_depth_column(columns: Iterable[str]) -> str:
    candidates = ["DEPTH", "DEPT", "MD", "TDEP", "DEPTH_M", "TVD", "RKB"]
    upper_map = {str(c).upper(): c for c in columns}
    for cand in candidates:
        if cand in upper_map:
            return upper_map[cand]
    return list(columns)[0]


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def read_log_data(path: str | Path, depth_col: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".las", ".las2"}:
        if lasio is None:
            raise ImportError("lasio is required for LAS files. Install with: pip install lasio")
        las = lasio.read(path)
        df = las.df().reset_index()
        if depth_col is None:
            depth_col = df.columns[0]
        df = df.rename(columns={depth_col: "DEPTH"})
        return _coerce_numeric_df(df)

    if suffix == ".csv":
        df = pd.read_csv(path)
        if depth_col is None:
            depth_col = _guess_depth_column(df.columns)
        df = df.rename(columns={depth_col: "DEPTH"})
        return _coerce_numeric_df(df)

    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        if depth_col is None:
            depth_col = _guess_depth_column(df.columns)
        df = df.rename(columns={depth_col: "DEPTH"})
        return _coerce_numeric_df(df)

    raise ValueError(f"Unsupported file type: {suffix}. Use LAS, CSV, or Parquet.")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def robust_rescale_01(x: np.ndarray, qlow: float = 0.01, qhigh: float = 0.99) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if finite.sum() == 0:
        return out
    lo = np.nanquantile(x[finite], qlow)
    hi = np.nanquantile(x[finite], qhigh)
    if hi <= lo:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((x[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def piecewise_linear(depth: np.ndarray, gamma1: float, theta: float, z_dtop: float, z_dbot: float) -> np.ndarray:
    mu = np.full_like(depth, gamma1, dtype=float)
    if z_dbot > z_dtop:
        mask_mid = (depth >= z_dtop) & (depth <= z_dbot)
        frac = (depth[mask_mid] - z_dtop) / (z_dbot - z_dtop)
        mu[mask_mid] = gamma1 + theta * frac
    mu[depth > z_dbot] = gamma1 + theta
    return mu


def prepare_window(df: pd.DataFrame, zcas: float, curve: str, dmax: float) -> pd.DataFrame:
    ztop = zcas - dmax
    zbot = zcas + dmax
    cols = ["DEPTH", curve] + [c for c in df.columns if c not in {"DEPTH", curve}]
    out = df.loc[(df["DEPTH"] >= ztop) & (df["DEPTH"] <= zbot), cols].copy()
    out = out.rename(columns={curve: "LOG"})
    out = out[np.isfinite(out["DEPTH"]) & np.isfinite(out["LOG"])]
    out = out.sort_values("DEPTH").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Gibbs full conditionals
# -----------------------------------------------------------------------------

def _sample_gamma1(gr, depth, theta, tau_v, tau_w, tau_z, z_dtop, z_dbot, mu_p, rng):
    m1 = depth < z_dtop
    m2 = (depth >= z_dtop) & (depth <= z_dbot)
    m3 = depth > z_dbot

    prec = tau_v
    mean_num = tau_v * mu_p

    if m1.any():
        prec += tau_v * m1.sum()
        mean_num += tau_v * gr[m1].sum()

    if m2.any() and z_dbot > z_dtop:
        frac = (depth[m2] - z_dtop) / (z_dbot - z_dtop)
        r2 = gr[m2] - theta * frac
        prec += tau_w * m2.sum()
        mean_num += tau_w * r2.sum()

    if m3.any():
        r3 = gr[m3] - theta
        prec += tau_z * m3.sum()
        mean_num += tau_z * r3.sum()

    return rng.normal(mean_num / prec, 1.0 / np.sqrt(prec))


def _sample_theta(gr, depth, gamma1, tau_w, tau_z, z_dtop, z_dbot, rng):
    m2 = (depth >= z_dtop) & (depth <= z_dbot)
    m3 = depth > z_dbot

    prec = tau_z
    mean_num = 0.0

    if m2.any() and z_dbot > z_dtop:
        frac = (depth[m2] - z_dtop) / (z_dbot - z_dtop)
        prec += tau_w * np.sum(frac ** 2)
        mean_num += tau_w * np.sum((gr[m2] - gamma1) * frac)

    if m3.any():
        prec += tau_z * m3.sum()
        mean_num += tau_z * np.sum(gr[m3] - gamma1)

    return rng.normal(mean_num / prec, 1.0 / np.sqrt(prec))


def _sample_tau_v(gr, depth, gamma1, z_dtop, mu_p, a, b, rng):
    m1 = depth < z_dtop
    n1 = m1.sum()
    ss1 = np.sum((gr[m1] - gamma1) ** 2) if n1 > 0 else 0.0
    shape = a + (n1 + 1) / 2.0
    rate = b + (ss1 + (gamma1 - mu_p) ** 2) / 2.0
    return rng.gamma(shape, 1.0 / rate)


def _sample_tau_w(gr, depth, gamma1, theta, z_dtop, z_dbot, a, b, rng):
    m2 = (depth >= z_dtop) & (depth <= z_dbot)
    n2 = m2.sum()
    if n2 > 0 and z_dbot > z_dtop:
        frac = (depth[m2] - z_dtop) / (z_dbot - z_dtop)
        mu2 = gamma1 + theta * frac
        ss2 = np.sum((gr[m2] - mu2) ** 2)
    else:
        ss2 = 0.0
    shape = max(a + n2 / 2.0, 1e-9)
    rate = max(b + ss2 / 2.0, 1e-12)
    return rng.gamma(shape, 1.0 / rate)


def _sample_tau_z(gr, depth, gamma1, theta, z_dbot, a, b, rng):
    m3 = depth > z_dbot
    n3 = m3.sum()
    ss3 = np.sum((gr[m3] - gamma1 - theta) ** 2) if n3 > 0 else 0.0
    shape = a + (n3 + 1) / 2.0
    rate = b + (ss3 + theta ** 2) / 2.0
    return rng.gamma(shape, 1.0 / rate)


def _sample_z_dtop(gr, depth, gamma1, theta, tau_v, tau_w, z_top, z_cas, d_min, z_dbot, max_candidates, rng):
    candidates = depth[(depth >= z_top + d_min) & (depth < z_cas) & (depth < z_dbot)]
    if len(candidates) == 0:
        return float(z_cas - d_min)
    if len(candidates) > max_candidates:
        idx = np.round(np.linspace(0, len(candidates) - 1, max_candidates)).astype(int)
        candidates = candidates[idx]

    r = gr - gamma1
    log_prob = np.empty(len(candidates), dtype=float)
    for k, z_dt in enumerate(candidates):
        m1 = depth < z_dt
        m2 = (depth >= z_dt) & (depth <= z_dbot)
        ll = 0.0
        if m1.any():
            ll += 0.5 * m1.sum() * np.log(tau_v) - 0.5 * tau_v * np.sum(r[m1] ** 2)
        if m2.any() and z_dbot > z_dt:
            frac = (depth[m2] - z_dt) / (z_dbot - z_dt)
            res2 = r[m2] - theta * frac
            ll += 0.5 * m2.sum() * np.log(tau_w) - 0.5 * tau_w * np.sum(res2 ** 2)
        log_prob[k] = ll

    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob /= prob.sum()
    return float(candidates[rng.choice(len(candidates), p=prob)])


def _sample_z_dbot(gr, depth, gamma1, theta, tau_w, tau_z, z_cas, z_bot, d_min, z_dtop, max_candidates, rng):
    candidates = depth[(depth > z_cas) & (depth > z_dtop) & (depth <= z_bot - d_min)]
    if len(candidates) == 0:
        return float(z_cas + d_min)
    if len(candidates) > max_candidates:
        idx = np.round(np.linspace(0, len(candidates) - 1, max_candidates)).astype(int)
        candidates = candidates[idx]

    r = gr - gamma1
    log_prob = np.empty(len(candidates), dtype=float)
    for k, z_db in enumerate(candidates):
        m2 = (depth >= z_dtop) & (depth <= z_db)
        m3 = depth > z_db
        ll = 0.0
        if m2.any() and z_db > z_dtop:
            frac = (depth[m2] - z_dtop) / (z_db - z_dtop)
            res2 = r[m2] - theta * frac
            ll += 0.5 * m2.sum() * np.log(tau_w) - 0.5 * tau_w * np.sum(res2 ** 2)
        if m3.any():
            ll += 0.5 * m3.sum() * np.log(tau_z) - 0.5 * tau_z * np.sum((r[m3] - theta) ** 2)
        log_prob[k] = ll

    log_prob -= np.max(log_prob)
    prob = np.exp(log_prob)
    prob /= prob.sum()
    return float(candidates[rng.choice(len(candidates), p=prob)])


# -----------------------------------------------------------------------------
# Fit single casing point
# -----------------------------------------------------------------------------

def run_gibbs_sampler(gr: np.ndarray, depth: np.ndarray, z_cas: float, config: GibbsConfig) -> PosteriorSamples:
    z_top = z_cas - config.d_max
    z_bot = z_cas + config.d_max
    mu_p = config.mu_p if config.mu_p is not None else float(np.mean(gr[depth < z_cas]) if np.any(depth < z_cas) else np.mean(gr))

    store: Dict[str, list] = {k: [] for k in ["gamma1", "theta", "z_dtop", "z_dbot", "tau_v", "tau_w", "tau_z"]}

    for chain in range(config.n_chains):
        rng = np.random.default_rng(config.seed + chain * 1000)

        gamma1 = float(np.mean(gr[depth < z_cas])) if np.any(depth < z_cas) else float(np.mean(gr))
        theta = 0.0
        tau_v = tau_w = tau_z = 1.0

        top_candidates = depth[(depth < z_cas) & (depth >= z_top + config.d_min)]
        bot_candidates = depth[(depth > z_cas) & (depth <= z_bot - config.d_min)]
        z_dtop = float(top_candidates[-1]) if len(top_candidates) else float(z_cas - config.d_min)
        z_dbot = float(bot_candidates[0]) if len(bot_candidates) else float(z_cas + config.d_min)

        n_total = config.n_burn + config.n_iter
        for it in range(n_total):
            gamma1 = _sample_gamma1(gr, depth, theta, tau_v, tau_w, tau_z, z_dtop, z_dbot, mu_p, rng)
            theta = _sample_theta(gr, depth, gamma1, tau_w, tau_z, z_dtop, z_dbot, rng)
            tau_v = _sample_tau_v(gr, depth, gamma1, z_dtop, mu_p, config.a, config.b, rng)
            tau_w = _sample_tau_w(gr, depth, gamma1, theta, z_dtop, z_dbot, config.a, config.b, rng)
            tau_z = _sample_tau_z(gr, depth, gamma1, theta, z_dbot, config.a, config.b, rng)
            z_dtop = _sample_z_dtop(gr, depth, gamma1, theta, tau_v, tau_w, z_top, z_cas, config.d_min, z_dbot, config.max_discrete_candidates, rng)
            z_dbot = _sample_z_dbot(gr, depth, gamma1, theta, tau_w, tau_z, z_cas, z_bot, config.d_min, z_dtop, config.max_discrete_candidates, rng)

            if it >= config.n_burn and ((it - config.n_burn) % config.thin == 0):
                store["gamma1"].append(gamma1)
                store["theta"].append(theta)
                store["z_dtop"].append(z_dtop)
                store["z_dbot"].append(z_dbot)
                store["tau_v"].append(tau_v)
                store["tau_w"].append(tau_w)
                store["tau_z"].append(tau_z)

    return PosteriorSamples(
        gamma1=np.asarray(store["gamma1"], dtype=float),
        theta=np.asarray(store["theta"], dtype=float),
        z_dtop=np.asarray(store["z_dtop"], dtype=float),
        z_dbot=np.asarray(store["z_dbot"], dtype=float),
        tau_v=np.asarray(store["tau_v"], dtype=float),
        tau_w=np.asarray(store["tau_w"], dtype=float),
        tau_z=np.asarray(store["tau_z"], dtype=float),
        casing_depth=float(z_cas),
        n_chains=config.n_chains,
    )


def fit_single_casing_point(df: pd.DataFrame, zcas: float, curve: str, config: GibbsConfig) -> Tuple[CasingFitResult, Optional[PosteriorSamples]]:
    ztop = zcas - config.d_max
    zbot = zcas + config.d_max
    window = prepare_window(df, zcas, curve, config.d_max)

    if len(window) < config.min_points:
        return CasingFitResult(
            zcas=zcas, ztop=ztop, zbot=zbot,
            zdtop_mean=np.nan, zdbot_mean=np.nan, gamma1_mean=np.nan,
            theta_mean=np.nan, theta_ci_low=np.nan, theta_ci_high=np.nan,
            tau_v_mean=np.nan, tau_w_mean=np.nan, tau_z_mean=np.nan,
            n_points=len(window), kept_points=0, status="failed",
            note="Too few points in fitting window.",
        ), None

    z = window["DEPTH"].to_numpy(dtype=float)
    y = window["LOG"].to_numpy(dtype=float)

    top_ok = np.any((z >= ztop + config.d_min) & (z < zcas))
    bot_ok = np.any((z > zcas) & (z <= zbot - config.d_min))
    if not (top_ok and bot_ok):
        return CasingFitResult(
            zcas=zcas, ztop=ztop, zbot=zbot,
            zdtop_mean=np.nan, zdbot_mean=np.nan, gamma1_mean=np.nan,
            theta_mean=np.nan, theta_ci_low=np.nan, theta_ci_high=np.nan,
            tau_v_mean=np.nan, tau_w_mean=np.nan, tau_z_mean=np.nan,
            n_points=len(window), kept_points=0, status="failed",
            note="No admissible zdtop/zdbot candidates.",
        ), None

    mu_p_local = config.mu_p
    if mu_p_local is None:
        above = y[z < zcas]
        mu_p_local = float(np.mean(above)) if len(above) else float(np.mean(y))

    local_cfg = GibbsConfig(**{**asdict(config), "mu_p": mu_p_local})
    posterior = run_gibbs_sampler(y, z, zcas, local_cfg)
    theta_low, theta_high = posterior.theta_ci_95
    kept_points = int(np.sum(~((z >= posterior.z_dtop_mean) & (z <= posterior.z_dbot_mean))))

    result = CasingFitResult(
        zcas=float(zcas),
        ztop=float(ztop),
        zbot=float(zbot),
        zdtop_mean=posterior.z_dtop_mean,
        zdbot_mean=posterior.z_dbot_mean,
        gamma1_mean=float(np.mean(posterior.gamma1)),
        theta_mean=posterior.theta_mean,
        theta_ci_low=theta_low,
        theta_ci_high=theta_high,
        tau_v_mean=float(np.mean(posterior.tau_v)),
        tau_w_mean=float(np.mean(posterior.tau_w)),
        tau_z_mean=float(np.mean(posterior.tau_z)),
        n_points=len(window),
        kept_points=kept_points,
        status="ok",
        note="Analytic Gibbs sampler.",
    )
    return result, posterior


# -----------------------------------------------------------------------------
# Correction / validation
# -----------------------------------------------------------------------------

def apply_shift_correction(df: pd.DataFrame, curve: str, fit_results: Sequence[CasingFitResult]) -> pd.DataFrame:
    out = df.copy().sort_values("DEPTH").reset_index(drop=True)
    out["GR_ORIG"] = pd.to_numeric(out[curve], errors="coerce")
    out["GR_SHIFTED"] = out["GR_ORIG"].astype(float)
    out["TRANSITION_MASK"] = False

    cumulative_shift = 0.0
    for r in sorted([r for r in fit_results if r.status == "ok"], key=lambda x: x.zcas):
        trans_mask = (out["DEPTH"] >= r.zdtop_mean) & (out["DEPTH"] <= r.zdbot_mean)
        below_mask = out["DEPTH"] > r.zdbot_mean
        out.loc[trans_mask, "TRANSITION_MASK"] = True
        gr_shifted = out.loc[below_mask, "GR_ORIG"] - (cumulative_shift + r.theta_mean)
        out.loc[below_mask, "GR_SHIFTED"] = gr_shifted
        cumulative_shift += r.theta_mean

    out.loc[out["TRANSITION_MASK"], "GR_SHIFTED"] = np.nan
    out["SGR"] = robust_rescale_01(out["GR_SHIFTED"].to_numpy())
    return out


def validate_casing_zone(df: pd.DataFrame, z_cas: float, window: float = 40.0, log_columns: Optional[List[str]] = None) -> pd.DataFrame:
    if log_columns is None:
        exclude = {"GR", "DEPTH", "MD", "TVD", "CALI", "CAL"}
        log_columns = [c for c in df.columns if str(c).upper() not in exclude]

    depth = df["DEPTH"].to_numpy(dtype=float)
    mask = (depth >= z_cas - window) & (depth <= z_cas + window)
    rows = []
    for col in log_columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df.loc[mask, col], errors="coerce").to_numpy(dtype=float)
        deps = depth[mask]
        ok = np.isfinite(vals)
        if ok.sum() < 6:
            continue
        above = vals[ok & (deps < z_cas)]
        below = vals[ok & (deps >= z_cas)]
        if len(above) < 3 or len(below) < 3:
            continue
        t_stat, p_val = stats.ttest_ind(above, below, equal_var=False)
        rows.append({
            "curve": col,
            "zcas": z_cas,
            "mean_above": float(np.mean(above)),
            "mean_below": float(np.mean(below)),
            "shift": float(np.mean(below) - np.mean(above)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant_0_05": bool(p_val < 0.05),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_single_fit(df: pd.DataFrame, zcas: float, curve: str, fit_result: CasingFitResult, posterior: Optional[PosteriorSamples], output_png: Path) -> None:
    if fit_result.status != "ok" or posterior is None:
        return
    window = prepare_window(df, zcas, curve, (fit_result.zbot - fit_result.ztop) / 2.0)
    if len(window) == 0:
        return

    z = window["DEPTH"].to_numpy(dtype=float)
    y = window["LOG"].to_numpy(dtype=float)
    z_fine = np.linspace(z.min(), z.max(), 500)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(posterior.theta), size=min(100, len(posterior.theta)), replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(z, y, s=10, alpha=0.6, label=curve)
    for idx in sample_idx:
        mu = piecewise_linear(z_fine, posterior.gamma1[idx], posterior.theta[idx], posterior.z_dtop[idx], posterior.z_dbot[idx])
        ax1.plot(z_fine, mu, alpha=0.05, linewidth=0.8)
    mu_mean = piecewise_linear(z_fine, fit_result.gamma1_mean, fit_result.theta_mean, fit_result.zdtop_mean, fit_result.zdbot_mean)
    ax1.plot(z_fine, mu_mean, linewidth=2, label="Posterior mean")
    ax1.axvline(zcas, linestyle="--", linewidth=1, label="Casing point")
    ax1.axvspan(fit_result.zdtop_mean, fit_result.zdbot_mean, alpha=0.15, label="Transition")
    ax1.set_xlabel("Depth")
    ax1.set_ylabel(curve)
    ax1.set_title(f"{curve} fit around casing {zcas:.2f}")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.hist(posterior.theta, bins=50, density=True, alpha=0.75)
    ax2.axvline(fit_result.theta_mean, linewidth=2, label=f"Mean = {fit_result.theta_mean:.2f}")
    ax2.axvline(fit_result.theta_ci_low, linestyle="--", linewidth=1, label="95% CI")
    ax2.axvline(fit_result.theta_ci_high, linestyle="--", linewidth=1)
    ax2.set_xlabel("theta")
    ax2.set_ylabel("Density")
    ax2.set_title("Posterior of theta")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def plot_full_log(df: pd.DataFrame, curve: str, output_png: Path) -> None:
    dep = df["DEPTH"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True)
    axes[0].plot(df["GR_ORIG"].to_numpy(), dep)
    axes[1].plot(df["GR_SHIFTED"].to_numpy(), dep)
    axes[2].plot(df["SGR"].to_numpy(), dep)
    axes[0].set_title("Original GR")
    axes[1].set_title("Shifted GR")
    axes[2].set_title("SGR")
    axes[0].set_xlabel(curve)
    axes[1].set_xlabel("Shifted GR")
    axes[2].set_xlabel("SGR [0,1]")
    axes[0].set_ylabel("Depth")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def plot_nphi_rhob_crossplot(df: pd.DataFrame, nphi: str, rhob: str, output_png: Path) -> None:
    sub = df[[nphi, rhob]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(sub[nphi], sub[rhob], s=8, alpha=0.5)
    plt.xlabel(nphi)
    plt.ylabel(rhob)
    plt.title("NPHI-RHOB crossplot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian shifted gamma-ray workflow around casing points.")
    parser.add_argument("--input", required=True, help="Input LAS / CSV / Parquet file.")
    parser.add_argument("--depth", default=None, help="Depth column for CSV/Parquet.")
    parser.add_argument("--gr", default="GR", help="Gamma-ray curve name.")
    parser.add_argument("--casing-points", nargs="+", type=float, required=True, help="Casing-point depths.")
    parser.add_argument("--dmax", type=float, default=40.0, help="Half-window size around casing point.")
    parser.add_argument("--dmin", type=float, default=3.0, help="Minimum segment length.")
    parser.add_argument("--a", type=float, default=1.0, help="Gamma prior alpha.")
    parser.add_argument("--b", type=float, default=1.0, help="Gamma prior beta/rate.")
    parser.add_argument("--n-iter", type=int, default=3000, help="Posterior iterations per chain.")
    parser.add_argument("--n-burn", type=int, default=1000, help="Burn-in per chain.")
    parser.add_argument("--n-chains", type=int, default=3, help="Number of chains.")
    parser.add_argument("--thin", type=int, default=1, help="Thinning interval.")
    parser.add_argument("--max-discrete-candidates", type=int, default=300, help="Max candidate depths for zdtop/zdbot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--validate-curves", nargs="*", default=["RHOB", "NPHI", "DT", "CALI"], help="Other logs to test around casing points.")
    parser.add_argument("--nphi", default=None, help="Optional NPHI curve name for crossplot.")
    parser.add_argument("--rhob", default=None, help="Optional RHOB curve name for crossplot.")
    parser.add_argument("--output-dir", default="sgr_output", help="Output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_log_data(args.input, depth_col=args.depth)
    if "DEPTH" not in df.columns:
        raise ValueError("Could not identify depth column.")
    if args.gr not in df.columns:
        raise ValueError(f"Gamma-ray curve '{args.gr}' not found in input.")

    df = df.sort_values("DEPTH").reset_index(drop=True)
    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    df[args.gr] = pd.to_numeric(df[args.gr], errors="coerce")
    df = df[np.isfinite(df["DEPTH"]) & np.isfinite(df[args.gr])].copy()

    config = GibbsConfig(
        n_iter=args.n_iter,
        n_burn=args.n_burn,
        n_chains=args.n_chains,
        thin=args.thin,
        d_max=args.dmax,
        d_min=args.dmin,
        a=args.a,
        b=args.b,
        max_discrete_candidates=args.max_discrete_candidates,
        seed=args.seed,
    )

    fit_results: List[CasingFitResult] = []
    posterior_map: Dict[float, PosteriorSamples] = {}
    posterior_summary_rows: List[pd.DataFrame] = []
    validation_rows: List[pd.DataFrame] = []

    for zcas in sorted(args.casing_points):
        print(f"Fitting casing point at {zcas} ...")
        result, posterior = fit_single_casing_point(df, zcas, args.gr, config)
        fit_results.append(result)
        if posterior is not None:
            posterior_map[zcas] = posterior
            summ = posterior.summary_frame().copy()
            summ.insert(0, "zcas", zcas)
            posterior_summary_rows.append(summ)

        plot_single_fit(
            df=df,
            zcas=zcas,
            curve=args.gr,
            fit_result=result,
            posterior=posterior,
            output_png=output_dir / f"fit_{args.gr}_{zcas:.2f}.png",
        )

        if args.validate_curves:
            validation_df = validate_casing_zone(df, zcas, window=args.dmax, log_columns=args.validate_curves)
            if not validation_df.empty:
                validation_rows.append(validation_df)

    results_df = pd.DataFrame([asdict(r) for r in fit_results])
    results_df.to_csv(output_dir / "casing_fit_summary.csv", index=False)

    if posterior_summary_rows:
        pd.concat(posterior_summary_rows, ignore_index=True).to_csv(output_dir / "posterior_parameter_summary.csv", index=False)

    if validation_rows:
        pd.concat(validation_rows, ignore_index=True).to_csv(output_dir / "validation_other_logs.csv", index=False)

    shifted = apply_shift_correction(df, curve=args.gr, fit_results=fit_results)
    shifted.to_csv(output_dir / "shifted_gamma_ray.csv", index=False)
    plot_full_log(shifted, curve=args.gr, output_png=output_dir / "shifted_gamma_ray_overview.png")

    if args.nphi and args.rhob and args.nphi in df.columns and args.rhob in df.columns:
        plot_nphi_rhob_crossplot(df, args.nphi, args.rhob, output_dir / "nphi_rhob_crossplot.png")

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
