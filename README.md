# BSGR – Bayesian Shifted Gamma-Ray (SGR) Correction

Python implementations of the Bayesian method for correcting open-hole gamma-ray (GR) logs across casing points (Oughton, Wooff & O'Connor, 2014). A copy of the paper is included in `doc/`.

## Quick start

### 1) Install dependencies

**Lightweight (pure Gibbs scripts):**

```bash
pip install numpy scipy matplotlib pandas lasio pyarrow
```

**Full (PyMC + ArviZ script):**

```bash
pip install numpy scipy matplotlib pandas lasio pyarrow pymc arviz
```

### 2) Run a synthetic demo (no data file required)

```bash
python sgr_bayesian1.py
```

### 3) Run on a real well (LAS)

```bash
python sgr_bayesian1.py --las well.las --casings 1000 2500 4000
```

Tip: run any script with `--help` for a complete list of options.

---

## Background

Gamma-ray logs recorded in different borehole sections (separated by casing points) are commonly affected by step-changes caused by:

- changes in borehole diameter or mud system,
- use of different logging tools between casing strings.

These artefacts make it difficult to compare or cross-calibrate GR values across a well.

The method in this repository removes those artefacts by fitting a piecewise-linear Bayesian model in a depth window around each known casing point, estimating the shift parameter **θ** using Markov Chain Monte Carlo (MCMC), and then applying the correction sequentially down the well to produce a **Shifted Gamma-Ray Index (SGR)** normalised to **[0, 1]**.

---

## Repository contents

| File | Description |
|------|-------------|
| `sgr_bayesian1.py` | Pure Gibbs sampler (NumPy / SciPy only). Minimal dependencies. Includes a built-in synthetic-well demo and optional LAS file support. |
| `sgr_bayesian2.py` | Enhanced Gibbs sampler (NumPy / SciPy only). Extends v1 with NPHI–RHOB crossplot visualisation, multi-curve validation, and structured output to a directory. |
| `shifted_gamma_ray_bayesian1.py` | Alternative Gibbs implementation (NumPy / SciPy only). Library-style structure; exports posterior JSON, multi-format input (LAS / CSV / Parquet), multi-chain convergence diagnostics. |
| `shifted_gamma_ray_bayesian2.py` | PyMC + ArviZ implementation using NUTS sampling. Richer diagnostics; requires a heavier dependency stack (`pymc`, `arviz`). |
| `doc/` | Reference paper (PDF). |

### Choosing a script

| Need | Recommended script |
|------|--------------------|
| Minimal dependencies, quick start | `sgr_bayesian1.py` |
| Minimal dependencies + crossplot / multi-curve validation | `sgr_bayesian2.py` |
| Library-style API, JSON output, multi-chain diagnostics | `shifted_gamma_ray_bayesian1.py` |
| Full Bayesian diagnostics (ArviZ, trace plots, HDI) | `shifted_gamma_ray_bayesian2.py` |

---

## Usage

### Synthetic demo (no data file required)

```bash
python sgr_bayesian1.py
```

### Real LAS file (casing points required)

```bash
python sgr_bayesian1.py --las well.las --casings 1000 2500 4000
```

### CSV or Parquet file with column names

```bash
python sgr_bayesian2.py \
  --input well.csv \
  --depth DEPT --gr-col GR \
  --casings 1000 2500 4000
```

### With crossplot and multi-curve validation

```bash
python sgr_bayesian2.py \
  --las well.las \
  --casings 1000 2500 4000 \
  --validate-curves RHOB NPHI DT \
  --nphi NPHI --rhob RHOB \
  --output-dir results/
```

### Library-style (shifted_gamma_ray_bayesian1)

```bash
python shifted_gamma_ray_bayesian1.py \
  --input well.las \
  --gr GR \
  --casing-points 1387 2700 4312 \
  --output-dir sgr_output
```

### PyMC / NUTS (shifted_gamma_ray_bayesian2)

```bash
python shifted_gamma_ray_bayesian2.py \
  --input well.las \
  --depth DEPT --gr GR \
  --casing-points 1387 2700 4312 \
  --dmax 40 --dmin 3 \
  --output-dir sgr_output
```

---

## Statistical model

For data in the window **[z_top, z_bot]** around casing point **z_cas**:

```
Zone 1  (z_top  < z < z_dtop):    GR_i ~ N(γ₁,           τ_v⁻¹)
Zone 2  (z_dtop ≤ z ≤ z_dbot):    GR_i ~ N(γ₁ + θ·fᵢ,    τ_w⁻¹)   fᵢ ∈ [0, 1]
Zone 3  (z_dbot < z < z_bot):     GR_i ~ N(γ₁ + θ,        τ_z⁻¹)
```

Priors:

```
τ_v, τ_w, τ_z  ~ Gamma(a, b)
γ₁ | τ_v       ~ N(μ_p, τ_v⁻¹)
θ  | τ_z       ~ N(0,   τ_z⁻¹)
z_dtop         ~ Uniform on depth grid in [z_top + d_min, z_cas)
z_dbot         ~ Uniform on depth grid in (z_cas, z_bot − d_min]
```

All conjugate full conditionals are derived analytically. The discrete depth parameters (`z_dtop`, `z_dbot`) are sampled by enumerating their finite support.

**Key parameters (typical defaults):**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `d_max` | 40 m | Half-window around each casing point |
| `d_min` | 3 m | Minimum length of each constant segment |
| `n_iter` | 3000 | MCMC iterations per chain (after burn-in) |
| `n_burn` | 1000 | Burn-in iterations discarded |
| `n_chains` | 3 | Independent chains |
| `a`, `b` | 1, 1 | Gamma prior hyperparameters (weakly informative / non-informative) |

---

## Outputs

All scripts produce a subset of:

- **Corrected GR log** — raw GR with the estimated shifts removed.
- **SGR** — Shifted Gamma-Ray Index, rescaled to [0, 1].
- **Posterior summary table** — mean, SD, 95% CI for γ₁, θ, z_dtop, z_dbot, τ_v, τ_w, τ_z.
- **Diagnostic plots** — GR log with fitted piecewise model, posterior histograms, trace plots (PyMC variant), NPHI–RHOB crossplot (v2 only).
- **CSV / JSON exports** — machine-readable posterior summaries and corrected log values.

---

## Reference

Oughton, R.H., Wooff, D.A. & O'Connor, S.A. (2014). *A Bayesian shifting method for uncertainty in the open-hole gamma-ray log around casing points.* **Petroleum Geoscience**, 20, 375–391. https://doi.org/10.1144/petgeo2014-006
