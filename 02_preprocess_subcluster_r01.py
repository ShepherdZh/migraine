#!/usr/bin/env python3
"""
Unified subcluster (leiden res=0.1 within each major lineage) for Satglia / PEP / Fibroblast_Dcn,
then merge hard-split / tiny clusters (see color_palette.SUBCLUSTER_MERGE).

Label: subcluster_r01 = {subtype}_L{leiden}  e.g. Satglia_L0, PEP_L1
Used across CellChat, Monocle3, GSEA, CellOracle, report figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from color_palette import (  # noqa: E402
    SUBCLUSTER_MERGE,
    active_subcluster_labels,
    apply_subcluster_merge,
    save_palette_csv,
    subcluster_colors,
)
SRC = ROOT / "results/objects/mouse_rna_Naive_IS_CSD_reintegrated.h5ad"
OUT_H5AD = ROOT / "results/objects/mouse_rna_focus_subcluster_r01.h5ad"
OUT_META = ROOT / "results/objects/subcluster_r01_map.csv"
OUT_FIG = ROOT / "results/reports/web/fig"

FOCUS = ["Satglia", "PEP", "Fibroblast_Dcn"]
LEIDEN_RES = 0.1


def assign_subclusters(adata) -> pd.Series:
    labels = pd.Series(index=adata.obs_names, dtype="object")
    for st in FOCUS:
        sub = adata[adata.obs["subtype"].astype(str) == st].copy()
        sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca")
        sc.tl.leiden(sub, resolution=LEIDEN_RES, key_added="leiden_sub")
        for cell, lev in zip(sub.obs_names, sub.obs["leiden_sub"].astype(str)):
            labels.loc[cell] = f"{st}_L{lev}"
    return labels


def main():
    adata = sc.read_h5ad(SRC)
    adata = adata[adata.obs["model"].isin(["Naive", "IS", "CSD"])].copy()
    focus = adata[adata.obs["subtype"].isin(FOCUS)].copy()

    raw = assign_subclusters(focus)
    focus.obs["subcluster_r01_raw"] = raw.astype("category")
    merged = apply_subcluster_merge(raw)
    focus.obs["subcluster_r01"] = merged.astype("category")

    n_before = raw.nunique()
    n_after = focus.obs["subcluster_r01"].nunique()
    print(f"Subclusters: {n_before} leiden → {n_after} after merge")
    for src, dst in SUBCLUSTER_MERGE.items():
        n = int((raw.astype(str) == src).sum())
        if n:
            print(f"  merged {src} ({n} cells) → {dst}")

    tab = (
        focus.obs.groupby(["subcluster_r01", "subtype", "model"], observed=True)
        .size()
        .reset_index(name="n_cells")
    )
    tab.to_csv(OUT_META, index=False)
    print(tab.sort_values("n_cells", ascending=False))

    OUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
    focus.write_h5ad(OUT_H5AD)

    save_palette_csv(labels=sorted(focus.obs["subcluster_r01"].astype(str).unique(), key=lambda x: (
        0 if x.startswith("Satglia") else 1 if x.startswith("PEP") else 2, x
    )))

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    pal = subcluster_colors()
    fig, ax = plt.subplots(figsize=(9, 7))
    cats = sorted(
        focus.obs["subcluster_r01"].astype(str).unique(),
        key=lambda x: (0 if x.startswith("Satglia") else 1 if x.startswith("PEP") else 2, x),
    )
    for c in cats:
        m = focus.obs["subcluster_r01"].astype(str) == c
        ax.scatter(
            focus.obsm["X_umap"][m, 0],
            focus.obsm["X_umap"][m, 1],
            s=4,
            alpha=0.7,
            c=pal.get(c, "#888"),
            label=c,
            rasterized=True,
        )
    ax.legend(fontsize=6, markerscale=2, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title(f"subcluster_r01 (leiden res={LEIDEN_RES}, merged hard splits)")
    ax.axis("off")
    plt.tight_layout()
    p = OUT_FIG / "umap_subcluster_r01.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p} ({focus.n_obs} cells, {n_after} subclusters)")

    print(f"Wrote {OUT_H5AD}")


if __name__ == "__main__":
    main()
