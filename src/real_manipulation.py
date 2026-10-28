"""
Fully-real manipulation experiment.

Instead of simulating reports as theta + alpha_i * s + noise, this runs the eight
trained SOC models on REAL LUCAS spectra that have been REALLY perturbed by the
moisture model, and feeds their ACTUAL predictions into the mechanism.

For each test sample with true OC theta and each perturbation magnitude w:
  honest report  mu_i        = model_i(R)
  manipulated    mu_i(w)     = model_i(R * exp(-w (c + k(lambda))))   (wetting)
The mechanism aggregates the manipulated reports:
  WLS uses the measured alpha_i to separate and remove the manipulation;
  the baselines (precision-weighted, naive mean, median) do not.

We report the systematic bias E[(theta_hat - theta)/theta] and the MARE versus w.
This tests whether a single measured alpha_i (a local linear sensitivity) is
enough to de-bias the real, generally nonlinear, model responses to a real
spectral perturbation. The WLS bias is not zero by construction here: it is zero
only if the real responses follow mu_i = theta + alpha_i * s closely enough.

Run on a GPU cluster via cluster/real_manip.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.spectral_models import (
    load_cache, water_kernel, _fit_torch, _TORCH, RNG_SEED, BROAD, _OSSL,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

MODELS = ["Linear", "RF", "CNN", "ViT", "SSL-SoilNet",
          "CNN-LSTM-ECA", "Spectral-Swin", "1DCNN-SAE"]


def fit_all(Xtr: np.ndarray, ytr: np.ndarray, n: int) -> dict:
    """Train all eight models on the training split; return predict closures."""
    P = {}
    rdg = Ridge(alpha=10.0).fit(Xtr, np.log1p(ytr))
    P["Linear"] = lambda X, _r=rdg: np.expm1(_r.predict(X))
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1,
                               random_state=RNG_SEED).fit(Xtr, np.log1p(ytr))
    P["RF"] = lambda X, _r=rf: np.expm1(_r.predict(X))
    for name, kind in _TORCH.items():
        P[name] = _fit_torch(kind, Xtr, ytr, Xtr[:2], n)   # returns predict closure
    return P


def run_real(w_grid=None) -> pd.DataFrame:
    if w_grid is None:
        w_grid = np.linspace(0.0, 0.5, 11)

    params = json.loads((_OSSL / "measured_params_cluster.json").read_text())
    alpha = np.array([params["alpha_norm"][m] for m in MODELS])
    sigma = np.array([params["sigma_gkg"][m] for m in MODELS])

    X, y, wl = load_cache()
    n = X.shape[1]
    k = water_kernel(wl)

    rng = np.random.default_rng(RNG_SEED)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    print(f"train {len(tr)} / test {len(te)}; {n} bands; models {MODELS}", flush=True)

    P = fit_all(X[tr], y[tr], n)
    Xte, yte = X[te], y[te]

    # Honest-prediction quality per model (sanity: each model's bias at w=0)
    mu0 = np.column_stack([P[m](Xte) for m in MODELS])
    for i, m in enumerate(MODELS):
        b = float(np.mean((mu0[:, i] - yte) / np.maximum(yte, 1e-6)))
        print(f"  {m:14s} honest bias={b*100:+.1f}%  RMSE={np.sqrt(np.mean((mu0[:,i]-yte)**2)):.2f}", flush=True)

    # WLS theta-row and precision weights from the measured parameters
    A = np.column_stack([np.ones(len(MODELS)), alpha])
    W = np.diag(1.0 / sigma ** 2)
    c_theta = (np.linalg.inv(A.T @ W @ A) @ A.T @ W)[0]    # length-8
    w_pw = (1.0 / sigma ** 2) / np.sum(1.0 / sigma ** 2)   # precision-weighted (Boomitra)

    rows = []
    for w in w_grid:
        Xw = Xte * np.exp(-w * (BROAD + k))[None, :]
        mu = np.column_stack([P[m](Xw) for m in MODELS])   # REAL predictions on perturbed spectra
        shift = float(np.mean(mu.mean(1) - mu0.mean(1)))    # mean real report shift (g/kg)
        ests = {
            "Ours (WLS)":   mu @ c_theta,
            "Boomitra":     mu @ w_pw,
            "Naive mean":   mu.mean(axis=1),
            "Median":       np.median(mu, axis=1),
        }
        for name, est in ests.items():
            bias = float(np.mean((est - yte) / np.maximum(yte, 1e-6)))
            mare = float(np.mean(np.abs(est - yte) / np.maximum(yte, 1e-6)))
            rows.append({"method": name, "w": float(w),
                         "bias": bias, "mare": mare, "report_shift_gkg": shift})

    df = pd.DataFrame(rows)
    out = _OSSL / "real_manipulation.csv"
    df.to_csv(out, index=False)

    print("\n=== REAL manipulation: bias E[(theta_hat-theta)/theta] ===", flush=True)
    print(f"{'w':>5} {'shift':>7} " + " ".join(f"{m:>11}" for m in ests))
    for w in w_grid:
        s = df[df.w == w]
        sh = float(s.report_shift_gkg.iloc[0])
        print(f"{w:>5.2f} {sh:>6.1f}g " + " ".join(f"{float(s[s.method==m].bias.iloc[0])*100:>+10.1f}%" for m in ests), flush=True)
    print(f"saved -> {out}", flush=True)
    return df


if __name__ == "__main__":
    run_real()
