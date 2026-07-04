"""
Data-driven measurement of sensor noise sigma_i and manipulation sensitivity
alpha_i from real LUCAS vis-NIR spectra (OSSL compilation).

Pipeline:
  1. Load LUCAS vis-NIR reflectance (350-2500 nm) joined to organic carbon (OC)
     ground truth via id.layer_uuid_txt; resample to a 10 nm grid.
  2. Train five model families to predict OC (g/kg) from the spectrum:
        ViT-1D, Transformer-1D, CNN-1D (torch), Random Forest, Ridge (sklearn).
     Five-fold cross-validation gives each model's RMSE = measured sigma_i.
  3. Simulate soil wetting as added water absorption at the O-H overtone bands
     (~1450 nm, ~1940 nm): R'(lambda) = R(lambda) * exp(-w * k(lambda)).
     alpha_i = mean d(OC_pred_i)/dw at small w = each model's SOC-prediction
     sensitivity to wetting.

All numbers are produced from the real spectra and real trained models. The only
modeling assumption is the shape of the wetting kernel k(lambda), which follows
the known water-absorption bands.

Run with the marble2026 conda env:
    conda run -n marble2026 python -m src.spectral_alpha
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

_OSSL = Path(__file__).parent.parent / "ossl_data"
_VISNIR = _OSSL / "ossl_visnir_L0_v1.2.csv.gz"
_LAB = _OSSL / "ossl_soillab_L1_v1.2.csv.gz"
_CACHE = _OSSL / "lucas_visnir_cache.npz"

RNG_SEED = 42
GRID_STEP = 10          # nm resampling grid
WL_LO, WL_HI = 400, 2450  # usable range (trim noisy edges of 350-2500)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_cache() -> None:
    """Parse OSSL files once, join LUCAS spectra to OC, resample, cache to npz."""
    print("Loading OC labels...")
    lab = pd.read_csv(
        _LAB, compression="gzip", low_memory=False,
        usecols=["dataset.code_ascii_txt", "id.layer_uuid_txt", "oc_usda.c729_w.pct"],
    )
    lab = lab[lab["dataset.code_ascii_txt"].astype(str).str.contains("LUCAS", case=False, na=False)]
    lab["oc"] = pd.to_numeric(lab["oc_usda.c729_w.pct"], errors="coerce") * 10.0  # % -> g/kg
    lab = lab.dropna(subset=["oc", "id.layer_uuid_txt"])[["id.layer_uuid_txt", "oc"]]

    print("Loading vis-NIR spectra (174 MB, this takes a moment)...")
    vis = pd.read_csv(_VISNIR, compression="gzip", low_memory=False)
    vis = vis[vis["dataset.code_ascii_txt"].astype(str).str.contains("LUCAS", case=False, na=False)]

    spec_cols = [c for c in vis.columns if c.startswith("scan_visnir.")]
    wl = np.array([float(c.split(".")[1].split("_")[0]) for c in spec_cols])
    order = np.argsort(wl)
    wl = wl[order]
    spec_cols = [spec_cols[i] for i in order]

    merged = vis[["id.layer_uuid_txt"] + spec_cols].merge(lab, on="id.layer_uuid_txt", how="inner")
    print(f"Joined LUCAS samples with spectra + OC: {len(merged)}")

    R = merged[spec_cols].to_numpy(dtype=np.float32)
    y = merged["oc"].to_numpy(dtype=np.float32)

    # Some leading bands (350-398 nm) are unmeasured (all-NaN); drop those bands.
    band_ok = np.isfinite(R).all(axis=0)
    R, wl = R[:, band_ok], wl[band_ok]
    print(f"Kept {band_ok.sum()} measured bands ({wl[0]:.0f}-{wl[-1]:.0f} nm), dropped {(~band_ok).sum()}")

    # Drop rows with any residual NaN in spectra or non-positive OC
    ok = np.isfinite(R).all(axis=1) & np.isfinite(y) & (y > 0)
    R, y = R[ok], y[ok]
    print(f"After cleaning: {len(y)} samples, {R.shape[1]} bands")

    # Resample to a uniform 10 nm grid over the usable range
    grid = np.arange(WL_LO, WL_HI + 1, GRID_STEP, dtype=np.float32)
    Rg = np.empty((R.shape[0], len(grid)), dtype=np.float32)
    for i in range(R.shape[0]):
        Rg[i] = np.interp(grid, wl, R[i])

    np.savez_compressed(_CACHE, X=Rg, y=y, wl=grid)
    print(f"Cached -> {_CACHE}  X={Rg.shape}  y={y.shape}  grid={grid[0]:.0f}-{grid[-1]:.0f} nm")


def load_cache():
    d = np.load(_CACHE)
    return d["X"], d["y"], d["wl"]


# ---------------------------------------------------------------------------
# Wetting perturbation (approach A: O-H water absorption bands)
# ---------------------------------------------------------------------------

def water_kernel(wl: np.ndarray) -> np.ndarray:
    """Normalized water-absorption kernel peaked at the O-H overtone bands.

    Two Gaussian bumps at ~1450 nm and ~1940 nm (first and second water overtone
    plus combination bands), the dominant moisture features in soil vis-NIR.
    """
    k = (np.exp(-0.5 * ((wl - 1450.0) / 60.0) ** 2)
         + np.exp(-0.5 * ((wl - 1940.0) / 80.0) ** 2))
    return k / k.max()


def wet(X: np.ndarray, w: float, k: np.ndarray) -> np.ndarray:
    """Apply wetting of magnitude w: reflectance attenuated at water bands."""
    return X * np.exp(-w * k)[None, :]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _torch_device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_torch_model(kind: str, n_bands: int):
    import torch.nn as nn

    if kind == "cnn":
        return nn.Sequential(
            nn.Unflatten(1, (1, n_bands)),
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32 * (n_bands // 4), 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    if kind in ("transformer", "vit"):
        import torch
        import torch.nn as nn

        class SpecTransformer(nn.Module):
            def __init__(self, n_bands, patch, d_model=64, heads=4, layers=2, vit=False):
                super().__init__()
                self.patch = patch
                self.npatch = n_bands // patch
                self.vit = vit
                self.embed = nn.Linear(patch, d_model)
                self.pos = nn.Parameter(torch.randn(1, self.npatch + (1 if vit else 0), d_model) * 0.02)
                if vit:
                    self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 2,
                                                 batch_first=True, dropout=0.1)
                self.tr = nn.TransformerEncoder(enc, layers)
                self.head = nn.Linear(d_model, 1)

            def forward(self, x):
                b = x.shape[0]
                t = x[:, : self.npatch * self.patch].reshape(b, self.npatch, self.patch)
                t = self.embed(t)
                if self.vit:
                    t = torch.cat([self.cls.expand(b, -1, -1), t], dim=1)
                t = t + self.pos
                t = self.tr(t)
                pooled = t[:, 0] if self.vit else t.mean(dim=1)
                return self.head(pooled)

        return SpecTransformer(n_bands, patch=8, vit=(kind == "vit"))

    raise ValueError(kind)


def train_torch(kind, Xtr, ytr, Xte, n_bands, epochs=40):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    dev = _torch_device()
    torch.manual_seed(RNG_SEED)
    model = make_torch_model(kind, n_bands).to(dev)
    # standardize targets in log space for stable training
    ytr_l = np.log1p(ytr)
    ymu, ysd = ytr_l.mean(), ytr_l.std()
    Xt = torch.tensor(Xtr); yt = torch.tensor(((ytr_l - ymu) / ysd).astype(np.float32)).view(-1, 1)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_l = model(torch.tensor(Xte).to(dev)).cpu().numpy().ravel() * ysd + ymu
    return np.expm1(pred_l), model, (ymu, ysd, dev)


def predict_torch(model, X, stats):
    import torch
    ymu, ysd, dev = stats
    model.eval()
    with torch.no_grad():
        pred_l = model(torch.tensor(X.astype(np.float32)).to(dev)).cpu().numpy().ravel() * ysd + ymu
    return np.expm1(pred_l)


# ---------------------------------------------------------------------------
# Main experiment: sigma_i (CV RMSE) and alpha_i (wetting sensitivity)
# ---------------------------------------------------------------------------

def run(n_folds: int = 5, wet_probe: float = 0.1) -> dict:
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    X, y, wl = load_cache()
    n_bands = X.shape[1]
    k = water_kernel(wl)
    print(f"Data: {X.shape}, grid {wl[0]:.0f}-{wl[-1]:.0f} nm; water-kernel peaks at 1450/1940 nm")

    models = ["ViT", "Transformer", "CNN", "RF", "Linear"]
    rmse = {m: [] for m in models}      # per-fold RMSE (g/kg) -> sigma_i
    sens = {m: [] for m in models}      # per-fold wetting sensitivity -> alpha_i

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    for fold, (tr, te) in enumerate(kf.split(X)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xte_wet = wet(Xte, wet_probe, k)
        print(f"  fold {fold+1}/{n_folds} ...", flush=True)

        # sklearn models
        for name, est in [("RF", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RNG_SEED)),
                          ("Linear", Ridge(alpha=10.0))]:
            ylog = np.log1p(ytr)
            est.fit(Xtr, ylog)
            pred = np.expm1(est.predict(Xte))
            pred_wet = np.expm1(est.predict(Xte_wet))
            rmse[name].append(float(np.sqrt(np.mean((pred - yte) ** 2))))
            sens[name].append(float(np.mean(pred_wet - pred) / wet_probe))

        # torch models
        for name, kind in [("ViT", "vit"), ("Transformer", "transformer"), ("CNN", "cnn")]:
            pred, model, stats = train_torch(kind, Xtr, ytr, Xte, n_bands)
            pred_wet = predict_torch(model, Xte_wet, stats)
            rmse[name].append(float(np.sqrt(np.mean((pred - yte) ** 2))))
            sens[name].append(float(np.mean(pred_wet - pred) / wet_probe))

    # Aggregate: sigma_i = mean CV RMSE; alpha_i = |mean sensitivity|, normalized to max=1
    sigma = {m: float(np.mean(rmse[m])) for m in models}
    raw_alpha = {m: abs(float(np.mean(sens[m]))) for m in models}
    amax = max(raw_alpha.values())
    alpha = {m: round(raw_alpha[m] / amax, 3) for m in models}

    result = {
        "n_samples": int(len(y)),
        "n_bands": int(n_bands),
        "sigma_gkg": {m: round(sigma[m], 2) for m in models},
        "alpha_norm": alpha,
        "alpha_raw_gkg_per_w": {m: round(raw_alpha[m], 3) for m in models},
        "rmse_folds": rmse,
        "sens_folds": sens,
    }
    out = _OSSL / "measured_params.json"
    out.write_text(json.dumps(result, indent=2))
    print("\n=== Measured parameters (real LUCAS spectra) ===")
    for m in models:
        print(f"  {m:12s}: sigma={sigma[m]:6.2f} g/kg   alpha={alpha[m]:.3f}")
    print(f"saved -> {out}")
    return result


if __name__ == "__main__":
    t0 = time.time()
    if not _CACHE.exists():
        build_cache()
    run()
    print(f"total {time.time()-t0:.0f}s")
