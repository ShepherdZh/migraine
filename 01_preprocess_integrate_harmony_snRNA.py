#!/usr/bin/env python3
import sys
"""
Re-integrate mouse snRNA FROM SCRATCH using only Naive / IS / CSD.

Unlike scripts/23_umap_paper_conditions.py (which only subset cells from the
5-condition UMAP), this script:
  1) keeps only Naive/IS/CSD cells
  2) recomputes HVG → PCA → neighbors → UMAP on that subset alone
  3) optional Harmony by sample_name for within-condition batch mixing

Outputs:
  results/objects/mouse_rna_Naive_IS_CSD_reintegrated.h5ad
  results/figures/condition_specific/reintegrated/*
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import fisher_exact

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
SRC = ROOT / "results/objects/mouse_rna_full.h5ad"
OUT_H5AD = ROOT / "results/objects/mouse_rna_Naive_IS_CSD_reintegrated.h5ad"
OUT = ROOT / "results/figures/condition_specific/reintegrated"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = ["Naive", "IS", "CSD"]
PALETTE = {"Naive": "#1f77b4", "IS": "#ff7f0e", "CSD": "#2ca02c"}


def main():
    print("Loading full object ...")
    adata = sc.read_h5ad(SRC)
    print("before", adata.shape, adata.obs["model"].value_counts().to_dict())

    keep = adata.obs["model"].astype(str).isin(PAPER)
    adata = adata[keep].copy()
    print("after Naive/IS/CSD subset (pre-reembed)", adata.shape, adata.obs["model"].value_counts().to_dict())

    # mouse_rna_full.h5ad stores counts in .raw (see DATA.md)
    X = adata.raw.X.copy()
    var = adata.raw.var.copy()
    obs = adata.obs.copy()
    adata = sc.AnnData(X=X, obs=obs, var=var)

    # Drop prior embeddings so we cannot accidentally reuse them
    for k in list(adata.obsm.keys()):
        del adata.obsm[k]
    for k in list(adata.obsp.keys()):
        del adata.obsp[k]
    for k in list(adata.uns.keys()):
        if k in ("neighbors", "umap", "pca", "leiden", "harmony"):
            del adata.uns[k]

    sc.pp.filter_genes(adata, min_cells=10)
    # seurat_v3 needs counts; ensure non-negative
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()

    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=50, svd_solver="arpack")

    import harmonypy as hm

    meta = adata_hvg.obs[["sample_name"]].copy()
    ho = hm.run_harmony(adata_hvg.obsm["X_pca"], meta, ["sample_name"])
    z_corr = np.asarray(ho.Z_corr)
    # harmonypy Z_corr is (n_pcs, n_cells); neighbors expect (n_cells, n_pcs)
    harmony = z_corr.T if z_corr.shape[1] == adata_hvg.n_obs else z_corr
    adata_hvg.obsm["X_pca_harmony"] = harmony
    use_rep = "X_pca_harmony"

    sc.pp.neighbors(adata_hvg, n_neighbors=20, n_pcs=40, use_rep=use_rep)
    sc.tl.umap(adata_hvg, min_dist=0.3)
    sc.tl.leiden(adata_hvg, resolution=1.0, key_added="leiden_r1")

    # Copy embedding back to full-gene adata for saving
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"].copy()
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"].copy()
    adata.obsm["X_pca_harmony"] = adata_hvg.obsm["X_pca_harmony"].copy()
    adata.obs["leiden_r1"] = adata_hvg.obs["leiden_r1"].values
    adata.uns["neighbors"] = adata_hvg.uns["neighbors"]
    adata.obsp["distances"] = adata_hvg.obsp["distances"]
    adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]
    adata.uns["reintegration"] = {
        "models": PAPER,
        "use_rep": use_rep,
        "note": "HVG/PCA/neighbors/UMAP recomputed after restricting to Naive/IS/CSD",
    }

    OUT_H5AD.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(OUT_H5AD)
    print("Wrote", OUT_H5AD)

    umap = np.asarray(adata.obsm["X_umap"])

    # model UMAP + composition
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    for m in PAPER:
        msk = adata.obs["model"].astype(str) == m
        axes[0].scatter(
            umap[msk, 0], umap[msk, 1], c=PALETTE[m], s=4, alpha=0.85,
            label=f"{m} (n={int(msk.sum())})", rasterized=True,
        )
    axes[0].set_title("Re-integrated UMAP (Naive/IS/CSD only)")
    axes[0].axis("off")
    axes[0].legend(markerscale=3, fontsize=9, frameon=False)

    comp = pd.crosstab(adata.obs["subtype"].astype(str), adata.obs["model"].astype(str))
    for m in PAPER:
        if m not in comp.columns:
            comp[m] = 0
    comp = comp[PAPER]
    frac = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    frac = frac.loc[comp.sum(axis=1).sort_values(ascending=False).index]
    frac.plot(kind="bar", stacked=True, ax=axes[1], color=[PALETTE[m] for m in PAPER], width=0.85)
    axes[1].set_ylabel("Fraction within subtype")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].legend(title="model", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "Fig_reintegrated_umap_model.png", dpi=200)
    fig.savefig(OUT / "Fig_reintegrated_umap_model.pdf")
    plt.close(fig)

    # subtype
    fig, ax = plt.subplots(figsize=(7, 6))
    cats = sorted(adata.obs["subtype"].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(cats):
        msk = adata.obs["subtype"].astype(str) == c
        ax.scatter(umap[msk, 0], umap[msk, 1], c=[cmap(i % 20)], s=4, alpha=0.85, label=c, rasterized=True)
    ax.axis("off")
    ax.set_title("Re-integrated UMAP — subtype")
    ax.legend(ncol=2, markerscale=3, fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    fig.savefig(OUT / "Fig_reintegrated_umap_subtype.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "Fig_reintegrated_umap_subtype.pdf", bbox_inches="tight")
    plt.close(fig)

    # leiden
    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(adata, color="leiden_r1", ax=ax, show=False, legend_loc="on data", frameon=False, title="Re-integrated Leiden")
    fig.savefig(OUT / "Fig_reintegrated_umap_leiden.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "Fig_reintegrated_umap_leiden.pdf", bbox_inches="tight")
    plt.close(fig)

    comp.to_csv(OUT / "Table_reintegrated_subtype_counts.csv")
    frac.to_csv(OUT / "Table_reintegrated_subtype_fraction.csv")

    rows = []
    for disease in ["IS", "CSD"]:
        for st in comp.index:
            a = int(comp.loc[st, disease])
            b = int(comp[disease].sum() - a)
            c = int(comp.loc[st, "Naive"])
            d = int(comp["Naive"].sum() - c)
            OR, p = fisher_exact([[a, b], [c, d]])
            rows.append(dict(disease=disease, subtype=st, n_disease=a, n_naive=c,
                             pct_disease=100 * a / max(comp[disease].sum(), 1),
                             pct_naive=100 * c / max(comp["Naive"].sum(), 1), OR=OR, p=p))
    enr = pd.DataFrame(rows)
    enr["padj"] = (enr["p"] * len(enr)).clip(upper=1.0)
    enr.sort_values(["disease", "OR"], ascending=[True, False]).to_csv(
        OUT / "Table_reintegrated_subtype_enrichment.csv", index=False
    )

    (OUT / "README.md").write_text(
        "# Re-integrated Naive / IS / CSD\n\n"
        "Embedding recomputed **after** restricting to Naive/IS/CSD (not a subset of the 5-condition UMAP).\n"
        f"- Object: `{OUT_H5AD}`\n"
        f"- Representation used for neighbors: see `adata.uns['reintegration']`\n"
        "- PBS/Control excluded from this object.\n",
        encoding="utf-8",
    )
    print("DONE →", OUT)


if __name__ == "__main__":
    main()
