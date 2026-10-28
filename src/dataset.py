"""
LUCAS 2018 topsoil data loader.
Returns soil organic carbon (OC) values in g/kg for the 0-20 cm depth layer.
"""

import numpy as np
import pandas as pd
from pathlib import Path

_LUCAS_CSV = Path(__file__).parent.parent / "data" / "LUCAS-SOIL-2018.csv"


def _require_csv(path: Path) -> Path:
    """Fail with a pointer to data/README.md when the LUCAS CSV is missing."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"LUCAS 2018 topsoil CSV not found at {path}. Its licence does not "
            "allow redistribution; see data/README.md for how to obtain it."
        )
    return Path(path)


def load_oc(path: Path = _LUCAS_CSV, oc_max: float = 100.0) -> np.ndarray:
    """Load and clean organic carbon values from the LUCAS 2018 topsoil CSV.

    Drops rows where OC is NaN or flagged as below the detection limit ("< LOD").
    Keeps mineral soils (OC <= oc_max g/kg) so that the ground-truth theta used by
    the mechanism experiments comes from the same domain on which the sensor
    parameters sigma_i and alpha_i were measured (the mineral subset, OC <= 100).
    Returns a float64 array of OC values in g/kg.
    """
    df = pd.read_csv(_require_csv(path), usecols=["OC"])
    df = df[df["OC"].astype(str).str.strip() != "< LOD"]
    oc = pd.to_numeric(df["OC"], errors="coerce").dropna()
    oc = oc[oc <= oc_max]
    return oc.values.astype(np.float64)


def load_oc_by_landcover(
    path: Path = _LUCAS_CSV, min_samples: int = 500, oc_max: float = 100.0
) -> dict[str, np.ndarray]:
    """Load OC values grouped by LUCAS land-cover class (LC0_Desc).

    Returns a dict {land_cover_name: oc_array}.  Only classes with at least
    min_samples valid OC observations are returned.  Applies the same mineral-soil
    filter (OC <= oc_max g/kg) as load_oc.  Used by Exp9 to test generalisation
    across soil regimes with increasing OC level and variance
    (Cropland < Grassland < Woodland).
    """
    df = pd.read_csv(_require_csv(path), usecols=["OC", "LC0_Desc"], low_memory=False)
    df = df[df["OC"].astype(str).str.strip() != "< LOD"]
    df["OC"] = pd.to_numeric(df["OC"], errors="coerce")
    df = df.dropna(subset=["OC", "LC0_Desc"])
    df = df[df["OC"] <= oc_max]

    groups: dict[str, np.ndarray] = {}
    for name, g in df.groupby("LC0_Desc"):
        if len(g) >= min_samples:
            groups[str(name)] = g["OC"].values.astype(np.float64)
    return groups


def sample_theta(oc: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n ground-truth SOC values by bootstrapping the LUCAS OC distribution."""
    return rng.choice(oc, size=n, replace=True)
