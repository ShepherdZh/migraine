"""Unified colors for subcluster_r01 and models — used across all report figures."""
from __future__ import annotations

import sys

from pathlib import Path

import matplotlib.colors as mcolors
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
PALETTE_CSV = ROOT / "results/objects/color_palette_subcluster_r01.csv"

MODEL_COLORS = {"Naive": "#4C78A8", "IS": "#F58518", "CSD": "#54A24B"}

# Merge hard-split / tiny subclusters into biologically supported groups.
# PEP_L2 (n≈53): unstable IS-acute → PEP_L1
# Satglia_L6 (n≈74), L3 (Naive-enriched) → L0 homeostatic pole
# Satglia_L4, L5 (mixed IS/CSD intermediate) → L2
SUBCLUSTER_MERGE: dict[str, str] = {
    "PEP_L2": "PEP_L1",
    "Satglia_L6": "Satglia_L0",
    "Satglia_L3": "Satglia_L0",
    "Satglia_L4": "Satglia_L2",
    "Satglia_L5": "Satglia_L2",
}

# Active subclusters after merge (Satglia×3, PEP×2, Fibroblast×4).
_SUBCLUSTER_HEX = {
    "Satglia_L0": "#08306b",
    "Satglia_L1": "#2171b5",
    "Satglia_L2": "#6baed6",
    "PEP_L0": "#00441b",
    "PEP_L1": "#74c476",
    "Fibroblast_Dcn_L0": "#7f2704",
    "Fibroblast_Dcn_L1": "#cc4c02",
    "Fibroblast_Dcn_L2": "#fd8d3c",
    "Fibroblast_Dcn_L3": "#fdbb84",
}


def apply_subcluster_merge(labels: pd.Series | list[str]) -> pd.Series:
    s = pd.Series(labels) if not isinstance(labels, pd.Series) else labels.copy()
    return s.astype(str).replace(SUBCLUSTER_MERGE)


def active_subcluster_labels() -> list[str]:
    return sorted(_SUBCLUSTER_HEX.keys(), key=_sort_key)


def _sort_key(label: str) -> tuple[int, str]:
    if label.startswith("Satglia_"):
        return (0, label)
    if label.startswith("PEP_"):
        return (1, label)
    return (2, label)


def subcluster_colors(labels: list[str] | None = None) -> dict[str, str]:
    if labels is None:
        return dict(_SUBCLUSTER_HEX)
    return {lb: _SUBCLUSTER_HEX.get(lb, "#888888") for lb in labels}


def lineage_base_color(label: str) -> str:
    if label.startswith("Satglia_"):
        return "#2171b5"
    if label.startswith("PEP_"):
        return "#238b45"
    if label.startswith("Fibroblast"):
        return "#cc4c02"
    return "#888888"


def save_palette_csv(path: Path | None = None, labels: list[str] | None = None) -> Path:
    path = path or PALETTE_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    use = labels or active_subcluster_labels()
    pal = subcluster_colors(use)
    df = pd.DataFrame([
        {"subcluster_r01": k, "hex": pal[k], "lineage": k.split("_L")[0] if "_L" in k else k}
        for k in use
    ])
    df.to_csv(path, index=False)
    return path


def color_vector_for_obs(obs_labels: pd.Series) -> list[str]:
    pal = subcluster_colors()
    return [pal.get(str(x), "#888888") for x in obs_labels]


if __name__ == "__main__":
    p = save_palette_csv()
    print(f"Wrote {p} ({len(active_subcluster_labels())} subclusters)")
