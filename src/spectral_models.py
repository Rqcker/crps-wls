"""
Eight-model SOC benchmark on real LUCAS vis-NIR spectra, and data-driven
measurement of sensor noise sigma_i and manipulation sensitivity alpha_i.

Models (compact 1D implementations; the deep models are simplified variants of
the cited architectures, not reproductions of them):
  Classical: Ridge (linear), Random Forest, 1D-CNN
  Deep:      ViT-1D        -- patch transformer with a class token
             SSL-SoilNet   -- the same transformer, pretrained to reconstruct
                              spectral patches (loss on a random 30% of the
                              patches), then fine-tuned; after Kakhani et al.,
                              IEEE TGRS 2024 (arXiv:2308.03586)
             CNN-LSTM-ECA  -- CNN + bidirectional LSTM + efficient channel
                              attention; after Wang et al., Eur. J. Soil Sci. 2025
             Spectral-Swin -- patch transformer with mean pooling and global
                              attention, standing in for the Swin model of
                              Jin et al., Geoderma 2023
             1DCNN-SAE     -- MLP autoencoder features with a regression head,
                              after stacked autoencoders (Vincent et al., JMLR 2010)

Setting: LUCAS mineral soils (OC <= 100 g/kg), the carbon-credit deployment
domain where vis-NIR SOC prediction is accurate.

Wetting model: wetting darkens the whole spectrum and deepens the O-H water
bands,
    R'(lambda) = R(lambda) * exp(-w * (c_broad + k_water(lambda))),
with k_water peaked at 1450 and 1940 nm. alpha_i = d(OC_pred_i)/dw at small w.

Outputs measured_params_cluster.json (sigma_i, alpha_i, R^2 benchmark), or the
path in CRPS_OUT.

Run on a GPU cluster with Slurm; see cluster/train_models.sh.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Training budget: the defaults match cluster/train_models.sh; the env
# variables override them (the smoke test uses a smaller budget).
EPOCHS = int(os.environ.get("CRPS_EPOCHS", "30"))
PRE_EPOCHS = int(os.environ.get("CRPS_PRE", "8"))

_ROOT = Path(__file__).parent.parent
_OSSL = _ROOT / "ossl_data"
_VISNIR = _OSSL / "ossl_visnir_L0_v1.2.csv.gz"
_LAB = _OSSL / "ossl_soillab_L1_v1.2.csv.gz"
_CACHE = _OSSL / "lucas_mineral_cache.npz"

RNG_SEED = 42
GRID_STEP = 10
WL_LO, WL_HI = 400, 2450
OC_MAX = 100.0          # mineral-soil ceiling (g/kg)
BROAD = 0.15            # broadband darkening weight in the wetting model
WET_PROBE = 0.1


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_cache(oc_max: float = OC_MAX) -> None:
    print("Loading OC labels...", flush=True)
    lab = pd.read_csv(_LAB, compression="gzip", low_memory=False,
                      usecols=["dataset.code_ascii_txt", "id.layer_uuid_txt", "oc_usda.c729_w.pct"])
    lab = lab[lab["dataset.code_ascii_txt"].astype(str).str.contains("LUCAS", case=False, na=False)]
    lab["oc"] = pd.to_numeric(lab["oc_usda.c729_w.pct"], errors="coerce") * 10.0
    lab = lab.dropna(subset=["oc", "id.layer_uuid_txt"])[["id.layer_uuid_txt", "oc"]]

    print("Loading vis-NIR spectra...", flush=True)
    vis = pd.read_csv(_VISNIR, compression="gzip", low_memory=False)
    vis = vis[vis["dataset.code_ascii_txt"].astype(str).str.contains("LUCAS", case=False, na=False)]
    spec_cols = [c for c in vis.columns if c.startswith("scan_visnir.")]
    wl = np.array([float(c.split(".")[1].split("_")[0]) for c in spec_cols])
    order = np.argsort(wl); wl = wl[order]; spec_cols = [spec_cols[i] for i in order]

    merged = vis[["id.layer_uuid_txt"] + spec_cols].merge(lab, on="id.layer_uuid_txt", how="inner")
    R = merged[spec_cols].to_numpy(dtype=np.float32)
    y = merged["oc"].to_numpy(dtype=np.float32)

    band_ok = np.isfinite(R).all(axis=0)
    R, wl = R[:, band_ok], wl[band_ok]

    ok = np.isfinite(R).all(axis=1) & np.isfinite(y) & (y > 0) & (y <= oc_max)
    R, y = R[ok], y[ok]
    print(f"Mineral subset (OC<= {oc_max}): {len(y)} samples", flush=True)

    grid = np.arange(WL_LO, WL_HI + 1, GRID_STEP, dtype=np.float32)
    Rg = np.empty((R.shape[0], len(grid)), dtype=np.float32)
    for i in range(R.shape[0]):
        Rg[i] = np.interp(grid, wl, R[i])
    np.savez_compressed(_CACHE, X=Rg, y=y, wl=grid)
    print(f"Cached -> {_CACHE}  X={Rg.shape}", flush=True)


def load_cache():
    d = np.load(_CACHE)
    return d["X"], d["y"], d["wl"]


def water_kernel(wl: np.ndarray) -> np.ndarray:
    k = (np.exp(-0.5 * ((wl - 1450.0) / 60.0) ** 2)
         + np.exp(-0.5 * ((wl - 1940.0) / 80.0) ** 2))
    return k / k.max()


def wet(X: np.ndarray, w: float, k: np.ndarray) -> np.ndarray:
    """Wetting: broadband darkening + water-band deepening."""
    return X * np.exp(-w * (BROAD + k))[None, :]


# ---------------------------------------------------------------------------
# Torch architectures (faithful 1D reimplementations)
# ---------------------------------------------------------------------------

def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build(kind: str, n: int):
    import torch
    import torch.nn as nn

    if kind == "cnn":
        return nn.Sequential(
            nn.Unflatten(1, (1, n)),
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Flatten(), nn.Linear(32 * (n // 4), 64), nn.ReLU(), nn.Linear(64, 1))

    if kind == "cnn_lstm_eca":
        class ECA(nn.Module):
            def __init__(self, ch, ksz=3):
                super().__init__(); self.conv = nn.Conv1d(1, 1, ksz, padding=ksz // 2); self.sig = nn.Sigmoid()
            def forward(self, x):                  # x: (B, C, L)
                w = x.mean(dim=2, keepdim=True).transpose(1, 2)   # (B,1,C)
                w = self.sig(self.conv(w)).transpose(1, 2)        # (B,C,1)
                return x * w

        class CNNLSTMECA(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.c1 = nn.Conv1d(1, 16, 7, padding=3); self.c2 = nn.Conv1d(16, 32, 5, padding=2)
                self.eca = ECA(32); self.pool = nn.MaxPool1d(2)
                self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
                self.head = nn.Linear(64, 1)
            def forward(self, x):
                x = x.unsqueeze(1)
                x = self.pool(torch.relu(self.c1(x)))
                x = self.pool(torch.relu(self.c2(x)))
                x = self.eca(x).transpose(1, 2)          # (B, L, C)
                o, _ = self.lstm(x)
                return self.head(o.mean(dim=1))
        return CNNLSTMECA(n)

    if kind in ("vit", "transformer", "swin", "ssl"):
        class SpecTr(nn.Module):
            def __init__(self, n, patch=8, d=64, heads=4, layers=2, vit=True, window=0):
                super().__init__()
                self.patch = patch; self.np = n // patch; self.vit = vit; self.window = window
                self.embed = nn.Linear(patch, d)
                extra = 1 if vit else 0
                self.pos = nn.Parameter(torch.randn(1, self.np + extra, d) * 0.02)
                if vit:
                    self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
                enc = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True, dropout=0.1)
                self.tr = nn.TransformerEncoder(enc, layers)
                self.dec = nn.Linear(d, patch)           # for SSL masked reconstruction
                self.head = nn.Linear(d, 1)
            def tokens(self, x):
                b = x.shape[0]
                t = x[:, : self.np * self.patch].reshape(b, self.np, self.patch)
                return self.embed(t)
            def forward(self, x, reconstruct=False):
                b = x.shape[0]; t = self.tokens(x)
                if reconstruct:
                    t = t + self.pos[:, (1 if self.vit else 0):, :]
                    return self.dec(self.tr(t))          # (B, np, patch)
                if self.vit:
                    t = torch.cat([self.cls.expand(b, -1, -1), t], dim=1)
                t = t + self.pos
                t = self.tr(t)
                pooled = t[:, 0] if self.vit else t.mean(dim=1)
                return self.head(pooled)
        return SpecTr(n, vit=(kind in ("vit", "ssl")), window=(8 if kind == "swin" else 0))

    if kind == "sae":
        class SAE(nn.Module):
            """Stacked autoencoder features + small MLP regressor head."""
            def __init__(self, n, code=64):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, code), nn.ReLU())
                self.dec = nn.Sequential(nn.Linear(code, 256), nn.ReLU(), nn.Linear(256, n))
                self.head = nn.Sequential(nn.Linear(code, 32), nn.ReLU(), nn.Linear(32, 1))
            def forward(self, x, reconstruct=False):
                z = self.enc(x)
                return self.dec(z) if reconstruct else self.head(z)
        return SAE(n)

    raise ValueError(kind)


def _fit_torch(kind, Xtr, ytr, Xte, n, epochs=None, pre_epochs=None):
    epochs = EPOCHS if epochs is None else epochs
    pre_epochs = PRE_EPOCHS if pre_epochs is None else pre_epochs
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    dev = _device(); torch.manual_seed(RNG_SEED)
    model = _build(kind, n).to(dev)
    yl = np.log1p(ytr); ymu, ysd = yl.mean(), yl.std()
    Xt = torch.tensor(Xtr); yt = torch.tensor(((yl - ymu) / ysd).astype(np.float32)).view(-1, 1)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=512, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mse = torch.nn.MSELoss()

    # self-supervised / autoencoder pretraining for SSL and SAE
    if kind in ("ssl", "sae"):
        for _ in range(pre_epochs):
            for xb, _yb in dl:
                xb = xb.to(dev); opt.zero_grad()
                if kind == "sae":
                    loss = mse(model(xb, reconstruct=True), xb)
                else:
                    tok = model.tokens(xb).detach()
                    mask = (torch.rand(xb.shape[0], model.np, 1, device=dev) < 0.3).float()
                    rec = model(xb, reconstruct=True)
                    tgt = xb[:, : model.np * model.patch].reshape(xb.shape[0], model.np, model.patch)
                    loss = mse(rec * mask, tgt * mask)
                loss.backward(); opt.step()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = mse(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    def predict(Xq):
        with torch.no_grad():
            p = model(torch.tensor(Xq.astype(np.float32)).to(dev)).cpu().numpy().ravel()
        return np.expm1(p * ysd + ymu)
    return predict


# ---------------------------------------------------------------------------
# Benchmark + sigma_i + alpha_i
# ---------------------------------------------------------------------------

MODELS = ["Linear", "RF", "CNN", "ViT", "SSL-SoilNet", "CNN-LSTM-ECA", "Spectral-Swin", "1DCNN-SAE"]
_TORCH = {"CNN": "cnn", "ViT": "vit", "SSL-SoilNet": "ssl",
          "CNN-LSTM-ECA": "cnn_lstm_eca", "Spectral-Swin": "swin", "1DCNN-SAE": "sae"}


def run(n_folds: int = 5, n_sub: int = 0) -> dict:
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    X, y, wl = load_cache()
    if n_sub and n_sub < len(y):
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(y), n_sub, replace=False)
        X, y = X[idx], y[idx]
    n = X.shape[1]; k = water_kernel(wl)
    print(f"Data {X.shape}; mineral OC<= {OC_MAX}; device {_device()}; "
          f"epochs={EPOCHS}/pre={PRE_EPOCHS}/folds={n_folds}", flush=True)

    rmse = {m: [] for m in MODELS}
    r2 = {m: [] for m in MODELS}
    sens = {m: [] for m in MODELS}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    for fold, (tr, te) in enumerate(kf.split(X)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Xwet = wet(Xte, WET_PROBE, k)
        print(f"fold {fold+1}/{n_folds}", flush=True)

        preds = {}
        # sklearn
        rdg = Ridge(alpha=10.0).fit(Xtr, np.log1p(ytr))
        preds["Linear"] = (np.expm1(rdg.predict(Xte)), np.expm1(rdg.predict(Xwet)))
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RNG_SEED).fit(Xtr, np.log1p(ytr))
        preds["RF"] = (np.expm1(rf.predict(Xte)), np.expm1(rf.predict(Xwet)))
        # torch
        for name, kind in _TORCH.items():
            pr = _fit_torch(kind, Xtr, ytr, Xte, n)
            preds[name] = (pr(Xte), pr(Xwet))

        for m in MODELS:
            p, pw = preds[m]
            rmse[m].append(float(np.sqrt(np.mean((p - yte) ** 2))))
            r2[m].append(float(r2_score(yte, p)))
            sens[m].append(float(np.mean(pw - p) / WET_PROBE))

    sigma = {m: float(np.mean(rmse[m])) for m in MODELS}
    r2m = {m: float(np.mean(r2[m])) for m in MODELS}
    raw = {m: float(np.mean(sens[m])) for m in MODELS}
    amax = max(abs(v) for v in raw.values())
    alpha = {m: round(abs(raw[m]) / amax, 3) for m in MODELS}

    res = {
        "setting": f"LUCAS mineral OC<={OC_MAX}", "n_samples": int(len(y)), "n_bands": int(n),
        "sigma_gkg": {m: round(sigma[m], 2) for m in MODELS},
        "r2": {m: round(r2m[m], 3) for m in MODELS},
        "alpha_norm": alpha,
        "alpha_raw_gkg_per_w": {m: round(raw[m], 2) for m in MODELS},
        "rmse_folds": rmse, "r2_folds": r2, "sens_folds": sens,
    }
    out = Path(os.environ.get("CRPS_OUT", _OSSL / "measured_params_cluster.json"))
    out.write_text(json.dumps(res, indent=2))
    print("\n=== SOC benchmark + measured params (mineral LUCAS) ===", flush=True)
    print(f"{'model':14s}{'R2':>7}{'RMSE(g/kg)':>12}{'alpha':>8}")
    for m in MODELS:
        print(f"{m:14s}{r2m[m]:7.3f}{sigma[m]:12.2f}{alpha[m]:8.3f}", flush=True)
    print(f"saved -> {out}", flush=True)
    return res


if __name__ == "__main__":
    t0 = time.time()
    if not _CACHE.exists():
        build_cache()
    folds = int(os.environ.get("CRPS_FOLDS", "5"))
    nsub = int(os.environ.get("CRPS_NSUB", "0"))
    run(n_folds=folds, n_sub=nsub)
    print(f"total {time.time()-t0:.0f}s", flush=True)
