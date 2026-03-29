#!/usr/bin/env python3
"""
shifted_gamma_ray_bayesian.py

Implementation of the workflow described in:
Oughton, R.H., Wooff, D.A. & O'Connor, S.A. (2014),
"A Bayesian shifting method for uncertainty in the open-hole gamma-ray log around casing points",
Petroleum Geoscience, 20, 375-391.

What this script does
---------------------
1. Reads well logs from LAS / CSV / Parquet.
2. Around each known casing point, extracts a symmetric depth window.
3. Fits a Bayesian piecewise-linear model with:
      - constant GR level above the casing point
      - linear transition zone
      - constant GR level below the transition zone
4. Estimates the shift parameter theta and the transition-zone limits
   zdtop and zdbot using MCMC.
5. Removes the transition-zone samples and shifts the deeper GR segment
   by the posterior mean of theta.
6. Builds a shifted gamma-ray log and rescales it to [0, 1] to obtain SGR.
7. Optionally runs the same test on other logs (RHOB, NPHI, DT, CALI, etc.)
   to assess whether casing effects are also present.
8. Produces plots and CSV outputs.

Notes
-----
- The paper's original implementation used R + JAGS/Gibbs sampling.
- Here the same statistical structure is implemented in Python with PyMC.
- Because zdtop and zdbot are discrete changepoint depths, this script uses
  Metropolis updates for those discrete indices and NUTS for continuous
  parameters. This preserves the article's Bayesian workflow while staying
  practical in Python.
- The method assumes lithology is reasonably consistent around the casing
  point, ideally massive shale. Use the optional validation plots and logs
  to assess that assumption before trusting a correction.

Typical usage
-------------
python shifted_gamma_ray_bayesian.py \
    --input well.las \
    --depth DEPT \
    --gr GR \
    --casing-points 1387 2700 4312 \
    --dmax 40 \
    --dmin 3 \
    --output-dir sgr_output

Dependencies
------------
pip install numpy pandas matplotlib scipy lasio pymc arviz pyarrow

Optional:
pip install dlisio

Author
------
Generated from the workflow in the attached article.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import lasio
except Exception:
    lasio = None

try:
    import pymc as pm
    import pytensor.tensor as pt
except Exception as exc:
    raise ImportError(
        "PyMC is required. Install with: pip install pymc arviz"
    ) from exc

import arviz as az


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class FitConfig:
    dmax: float = 40.0
    dmin: float = 3.0
    a: float = 1.0
    b: float = 1.0
    mu_p: Optional[float] = None
    draws: int = 2000
    tune: int = 1000
    chains: int = 3
    target_accept: float = 0.9
    random_seed: int = 42
    min_points: int = 40


@dataclass
class CasingFitResult:
    zcas: float
    ztop: float
    zbot: float
    zdtop_mean: float
    zdbot_mean: float
    gamma1_mean: float
    theta_mean: float
    tau_v_mean: float
    tau_w_mean: float
    tau_z_mean: float
    theta_hdi_low: float
    theta_hdi_high: float
    n_points: int
    kept_points: int
    status: str
    note: str = ""


# -----------------------------------------------------------------------------
# File readers
# -----------------------------------------------------------------------------

def read_log_data(
    path: str | Path,
    depth_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read logs from LAS / CSV / Parquet into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Input file path.
    depth_col : str, optional
        Name of depth column for text/tabular formats.

    Returns
    -------
    pd.DataFrame
        DataFrame with numeric columns only where possible.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".las", ".las2"}:
        if lasio is None:
            raise ImportError("lasio is required for LAS files. Install with: pip install lasio")
        las = lasio.read(path)
        df = las.df().reset_index()
        if depth_col is None:
            # las.df() usually indexes on depth; after reset_index() first column is depth.
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


def _guess_depth_column(columns: Iterable[str]) -> str:
    candidates = ["DEPTH", "DEPT", "MD", "TDEP", "DEPTH_M", "TVD", "RKB"]
    upper_map = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand in upper_map:
            return upper_map[cand]
    return list(columns)[0]


def _coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def robust_rescale_01(x: np.ndarray, qlow: float = 0.01, qhigh: float = 0.99) -> np.ndarray:
    """
    Rescale to [0, 1] using robust quantiles.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    y = np.full_like(x, np.nan, dtype=float)
    if finite.sum() == 0:
        return y
    lo = np.nanquantile(x[finite], qlow)
    hi = np.nanquantile(x[finite], qhigh)
    if hi <= lo:
        y[finite] = 0.5
        return y
    y[finite] = np.clip((x[finite] - lo) / (hi - lo), 0.0, 1.0)
    return y


def smooth_series(x: np.ndarray, window: int = 11) -> np.ndarray:
    """
    Light moving-average smoothing for visualization only.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def prepare_window(
    df: pd.DataFrame,
    zcas: float,
    curve: str,
    dmax: float,
) -> pd.DataFrame:
    """
    Extract one depth window around a casing point.
    """
    ztop = zcas - dmax
    zbot = zcas + dmax
    out = df[(df["DEPTH"] >= ztop) & (df["DEPTH"] <= zbot)].copy()
    out = out[["DEPTH", curve] + [c for c in df.columns if c not in {"DEPTH", curve}]].copy()
    out = out.rename(columns={curve: "LOG"})
    out = out[np.isfinite(out["DEPTH"]) & np.isfinite(out["LOG"])]
    out = out.sort_values("DEPTH").reset_index(drop=True)
    return out


def make_piecewise_mean(depth: np.ndarray, gamma1: np.ndarray, theta: np.ndarray,
                        zdtop: np.ndarray, zdbot: np.ndarray) -> np.ndarray:
    """
    Vectorized piecewise mean function.
    """
    depth = np.asarray(depth)
    out = np.empty_like(depth, dtype=float)

    mask_top = depth <= zdtop
    mask_bot = depth >= zdbot
    mask_mid = (~mask_top) & (~mask_bot)

    out[mask_top] = gamma1
    out[mask_bot] = gamma1 + theta

    if np.any(mask_mid):
        frac = (depth[mask_mid] - zdtop) / np.maximum(zdbot - zdtop, 1e-6)
        out[mask_mid] = gamma1 + theta * frac

    return out


# -----------------------------------------------------------------------------
# Bayesian fitting
# -----------------------------------------------------------------------------

def fit_single_casing_point(
    df: pd.DataFrame,
    zcas: float,
    curve: str,
    config: FitConfig,
) -> Tuple[CasingFitResult, Optional[az.InferenceData]]:
    """
    Fit the Bayesian piecewise-linear model around one casing point.
    """
    ztop = zcas - config.dmax
    zbot = zcas + config.dmax

    window = prepare_window(df, zcas=zcas, curve=curve, dmax=config.dmax)
    if len(window) < config.min_points:
        result = CasingFitResult(
            zcas=zcas, ztop=ztop, zbot=zbot,
            zdtop_mean=np.nan, zdbot_mean=np.nan,
            gamma1_mean=np.nan, theta_mean=np.nan,
            tau_v_mean=np.nan, tau_w_mean=np.nan, tau_z_mean=np.nan,
            theta_hdi_low=np.nan, theta_hdi_high=np.nan,
            n_points=len(window), kept_points=0,
            status="failed", note="Too few points in fitting window.",
        )
        return result, None

    z = window["DEPTH"].to_numpy(dtype=float)
    y = window["LOG"].to_numpy(dtype=float)

    # Allowed discrete candidate indices:
    # zdtop < zcas and zdbot > zcas, while each constant segment must be at least dmin wide.
    top_candidates = np.where((z > ztop + config.dmin) & (z < zcas))[0]
    bot_candidates = np.where((z > zcas) & (z < zbot - config.dmin))[0]

    if len(top_candidates) == 0 or len(bot_candidates) == 0:
        result = CasingFitResult(
            zcas=zcas, ztop=ztop, zbot=zbot,
            zdtop_mean=np.nan, zdbot_mean=np.nan,
            gamma1_mean=np.nan, theta_mean=np.nan,
            tau_v_mean=np.nan, tau_w_mean=np.nan, tau_z_mean=np.nan,
            theta_hdi_low=np.nan, theta_hdi_high=np.nan,
            n_points=len(window), kept_points=0,
            status="failed", note="No admissible zdtop/zdbot candidates.",
        )
        return result, None

    mu_p = config.mu_p
    if mu_p is None:
        pre = window[(window["DEPTH"] >= (zcas - config.dmin)) & (window["DEPTH"] <= zcas)]["LOG"]
        mu_p = float(pre.mean()) if len(pre) else float(np.nanmean(y))

    with pm.Model() as model:
        # Discrete changepoint indices; uniform over admissible candidates as in the paper.
        k_top = pm.DiscreteUniform("k_top", lower=int(top_candidates.min()), upper=int(top_candidates.max()))
        k_bot = pm.DiscreteUniform("k_bot", lower=int(bot_candidates.min()), upper=int(bot_candidates.max()))

        zdtop = pm.Deterministic("zdtop", z[k_top])
        zdbot = pm.Deterministic("zdbot", z[k_bot])

        # Priors matching the article's structure.
        tau_v = pm.Gamma("tau_v", alpha=config.a, beta=config.b)
        tau_w = pm.Gamma("tau_w", alpha=config.a, beta=config.b)
        tau_z = pm.Gamma("tau_z", alpha=config.a, beta=config.b)

        gamma1 = pm.Normal("gamma1", mu=mu_p, tau=tau_v)
        theta = pm.Normal("theta", mu=0.0, tau=tau_z)

        z_shared = pt.as_tensor_variable(z)

        mu_top = pt.full(z.shape, gamma1)
        frac = (z_shared - zdtop) / pt.maximum(zdbot - zdtop, 1e-6)
        mu_mid = gamma1 + theta * frac
        mu_bot = pt.full(z.shape, gamma1 + theta)

        is_top = z_shared <= zdtop
        is_bot = z_shared >= zdbot
        mu = pt.switch(is_top, mu_top, pt.switch(is_bot, mu_bot, mu_mid))

        sigma_v = 1.0 / pt.sqrt(tau_v)
        sigma_w = 1.0 / pt.sqrt(tau_w)
        sigma_z = 1.0 / pt.sqrt(tau_z)
        sigma = pt.switch(is_top, sigma_v, pt.switch(is_bot, sigma_z, sigma_w))

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        step_discrete = pm.Metropolis(vars=[k_top, k_bot])
        step_cont = pm.NUTS(vars=[gamma1, theta, tau_v, tau_w, tau_z], target_accept=config.target_accept)

        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=1,
            random_seed=config.random_seed,
            progressbar=False,
            compute_convergence_checks=True,
            step=[step_discrete, step_cont],
            return_inferencedata=True,
        )

    posterior = idata.posterior

    theta_samples = posterior["theta"].values.reshape(-1)
    zdtop_samples = posterior["zdtop"].values.reshape(-1)
    zdbot_samples = posterior["zdbot"].values.reshape(-1)

    theta_hdi = az.hdi(theta_samples, hdi_prob=0.95)

    kept_points = int(np.sum(~((z > np.mean(zdtop_samples)) & (z < np.mean(zdbot_samples)))))

    result = CasingFitResult(
        zcas=float(zcas),
        ztop=float(ztop),
        zbot=float(zbot),
        zdtop_mean=float(np.mean(zdtop_samples)),
        zdbot_mean=float(np.mean(zdbot_samples)),
        gamma1_mean=float(np.mean(posterior["gamma1"].values)),
        theta_mean=float(np.mean(theta_samples)),
        tau_v_mean=float(np.mean(posterior["tau_v"].values)),
        tau_w_mean=float(np.mean(posterior["tau_w"].values)),
        tau_z_mean=float(np.mean(posterior["tau_z"].values)),
        theta_hdi_low=float(theta_hdi[0]),
        theta_hdi_high=float(theta_hdi[1]),
        n_points=len(window),
        kept_points=kept_points,
        status="ok",
        note="Posterior mean summary.",
    )
    return result, idata


# -----------------------------------------------------------------------------
# Apply correction
# -----------------------------------------------------------------------------

def apply_shift_correction(
    df: pd.DataFrame,
    curve: str,
    fit_results: Sequence[CasingFitResult],
) -> pd.DataFrame:
    """
    Build shifted GR and SGR from posterior mean corrections.

    The workflow follows the paper:
      - remove points in transition zones
      - shift the data below each transition zone by theta
      - rescale final shifted log to [0, 1]
    """
    out = df.copy().sort_values("DEPTH").reset_index(drop=True)
    out["GR_ORIG"] = pd.to_numeric(out[curve], errors="coerce")
    out["GR_SHIFTED"] = out["GR_ORIG"].astype(float)
    out["TRANSITION_MASK"] = False

    # Apply sequentially from shallow to deep.
    valid_results = [r for r in fit_results if r.status == "ok"]
    valid_results = sorted(valid_results, key=lambda r: r.zcas)

    cumulative_shift = 0.0
    for r in valid_results:
        mid_mask = (out["DEPTH"] > r.zdtop_mean) & (out["DEPTH"] < r.zdbot_mean)
        bot_mask = out["DEPTH"] >= r.zdbot_mean

        out.loc[mid_mask, "TRANSITION_MASK"] = True
        # Following the paper's sign convention:
        # if the deeper level is gamma1 + theta, subtract theta from the deeper segment
        # to reconnect with the upper level.
        cumulative_shift += r.theta_mean
        out.loc[bot_mask, "GR_SHIFTED"] = out.loc[bot_mask, "GR_ORIG"] - cumulative_shift

    out["SGR"] = robust_rescale_01(out["GR_SHIFTED"].to_numpy())
    out.loc[out["TRANSITION_MASK"], "SGR"] = np.nan
    return out


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def run_log_validation(
    df: pd.DataFrame,
    casing_points: Sequence[float],
    curves: Sequence[str],
    config: FitConfig,
) -> pd.DataFrame:
    """
    Run the same casing-shift fit on other logs to see whether theta differs from 0.
    """
    rows = []
    for curve in curves:
        if curve not in df.columns:
            continue
        for zcas in casing_points:
            try:
                result, _ = fit_single_casing_point(df, zcas=zcas, curve=curve, config=config)
                row = asdict(result)
                row["curve"] = curve
                row["theta_significant_95"] = bool(
                    np.isfinite(result.theta_hdi_low)
                    and np.isfinite(result.theta_hdi_high)
                    and (result.theta_hdi_low > 0.0 or result.theta_hdi_high < 0.0)
                )
                rows.append(row)
            except Exception as exc:
                rows.append({
                    "curve": curve,
                    "zcas": zcas,
                    "status": "failed",
                    "note": str(exc),
                })
    return pd.DataFrame(rows)


def plot_nphi_rhob_crossplot(df: pd.DataFrame, nphi: str, rhob: str, output_png: Path) -> None:
    """
    NPHI-RHOB crossplot suggested in the paper for lithology consistency checks.
    """
    sub = df[[nphi, rhob]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
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
# Plotting
# -----------------------------------------------------------------------------

def plot_single_fit(
    df: pd.DataFrame,
    zcas: float,
    curve: str,
    fit_result: CasingFitResult,
    output_png: Path,
) -> None:
    """
    Plot data window and posterior-mean fitted curve.
    """
    window = prepare_window(df, zcas, curve, dmax=(fit_result.zbot - fit_result.ztop) / 2.0)
    if len(window) == 0 or fit_result.status != "ok":
        return

    z = window["DEPTH"].to_numpy()
    y = window["LOG"].to_numpy()
    yhat = make_piecewise_mean(
        depth=z,
        gamma1=fit_result.gamma1_mean,
        theta=fit_result.theta_mean,
        zdtop=fit_result.zdtop_mean,
        zdbot=fit_result.zdbot_mean,
    )

    plt.figure(figsize=(8, 4.5))
    plt.scatter(z, y, s=10, alpha=0.6, label=curve)
    plt.plot(z, yhat, linewidth=2, label="Posterior mean fit")
    plt.axvline(zcas, linestyle="--", linewidth=1, label="Casing point")
    plt.axvspan(fit_result.zdtop_mean, fit_result.zdbot_mean, alpha=0.15, label="Transition zone")
    plt.xlabel("Depth")
    plt.ylabel(curve)
    plt.title(f"{curve} fit around casing point {zcas:.2f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


def plot_full_log(df: pd.DataFrame, curve: str, output_png: Path) -> None:
    """
    Plot original GR, shifted GR, and SGR.
    """
    dep = df["DEPTH"].to_numpy()
    gr = df["GR_ORIG"].to_numpy()
    sgr_curve = df["GR_SHIFTED"].to_numpy()
    sgr_index = df["SGR"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 8), sharey=True)

    axes[0].plot(gr, dep)
    axes[0].set_title("Original GR")
    axes[0].set_xlabel(curve)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sgr_curve, dep)
    axes[1].set_title("Shifted GR")
    axes[1].set_xlabel("Shifted GR")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sgr_index, dep)
    axes[2].set_title("SGR")
    axes[2].set_xlabel("SGR [0, 1]")
    axes[2].grid(True, alpha=0.3)

    axes[0].set_ylabel("Depth")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian shifted gamma-ray workflow around casing points.")
    parser.add_argument("--input", required=True, help="Input LAS / CSV / Parquet file.")
    parser.add_argument("--depth", default=None, help="Depth column (for CSV/Parquet).")
    parser.add_argument("--gr", default="GR", help="Gamma-ray curve name.")
    parser.add_argument("--casing-points", nargs="+", type=float, required=True, help="Casing-point depths.")
    parser.add_argument("--dmax", type=float, default=40.0, help="Half-window size around casing point.")
    parser.add_argument("--dmin", type=float, default=3.0, help="Minimum length of constant sections.")
    parser.add_argument("--a", type=float, default=1.0, help="Gamma prior alpha for precisions.")
    parser.add_argument("--b", type=float, default=1.0, help="Gamma prior beta for precisions.")
    parser.add_argument("--draws", type=int, default=2000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning draws per chain.")
    parser.add_argument("--chains", type=int, default=3, help="Number of chains.")
    parser.add_argument("--target-accept", type=float, default=0.9, help="NUTS target_accept.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--validate-curves", nargs="*", default=["RHOB", "NPHI", "DT", "CALI"], help="Other logs to test for casing effects.")
    parser.add_argument("--nphi", default=None, help="Optional neutron porosity curve name for crossplot.")
    parser.add_argument("--rhob", default=None, help="Optional density curve name for crossplot.")
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

    config = FitConfig(
        dmax=args.dmax,
        dmin=args.dmin,
        a=args.a,
        b=args.b,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )

    fit_results: List[CasingFitResult] = []
    for zcas in args.casing_points:
        print(f"Fitting casing point at {zcas} ...")
        try:
            result, idata = fit_single_casing_point(df, zcas=zcas, curve=args.gr, config=config)
            fit_results.append(result)

            plot_single_fit(
                df=df,
                zcas=zcas,
                curve=args.gr,
                fit_result=result,
                output_png=output_dir / f"fit_{args.gr}_{zcas:.2f}.png",
            )

            if idata is not None:
                az.to_netcdf(idata, output_dir / f"idata_{args.gr}_{zcas:.2f}.nc")
        except Exception as exc:
            fit_results.append(CasingFitResult(
                zcas=zcas,
                ztop=zcas - config.dmax,
                zbot=zcas + config.dmax,
                zdtop_mean=np.nan, zdbot_mean=np.nan,
                gamma1_mean=np.nan, theta_mean=np.nan,
                tau_v_mean=np.nan, tau_w_mean=np.nan, tau_z_mean=np.nan,
                theta_hdi_low=np.nan, theta_hdi_high=np.nan,
                n_points=0, kept_points=0,
                status="failed",
                note=str(exc),
            ))

    results_df = pd.DataFrame([asdict(r) for r in fit_results])
    results_df.to_csv(output_dir / "casing_fit_summary.csv", index=False)

    shifted = apply_shift_correction(df, curve=args.gr, fit_results=fit_results)
    shifted.to_csv(output_dir / "shifted_gamma_ray.csv", index=False)
    plot_full_log(shifted, curve=args.gr, output_png=output_dir / "shifted_gamma_ray_overview.png")

    # Optional validation on other logs
    validation_df = run_log_validation(
        df=df,
        casing_points=args.casing_points,
        curves=args.validate_curves,
        config=config,
    )
    validation_df.to_csv(output_dir / "validation_other_logs.csv", index=False)

    # Optional NPHI-RHOB crossplot
    if args.nphi and args.rhob and args.nphi in df.columns and args.rhob in df.columns:
        plot_nphi_rhob_crossplot(df, args.nphi, args.rhob, output_dir / "nphi_rhob_crossplot.png")

    # Save run config
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
