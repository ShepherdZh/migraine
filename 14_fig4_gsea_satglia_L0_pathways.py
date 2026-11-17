#!/usr/bin/env python3
"""Enrichr per subcluster_r01 — one bar chart per subcluster (no heatmap)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from color_palette import lineage_base_color, subcluster_colors  # noqa: E402

H5AD = ROOT / "results/objects/mouse_rna_focus_subcluster_r01.h5ad"
OUT_DIR = ROOT / "results/enrichment/subcluster_r01"
WEB_FIG = ROOT / "results/reports/web/fig"
BARS_DIR = WEB_FIG / "gsea_bars"

GENE_SETS = ["GO_Biological_Process_2023", "KEGG_2019_Mouse", "MSigDB_Hallmark_2020"]
MIN_CELLS = 80
MIN_DEG = 15
PADJ = 0.05
LOGFC = 0.25
TOP_TERMS = 10


def safe_name(cl: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", cl)


def deg_is_vs_naive(adata, group: str) -> pd.DataFrame:
    sub = adata[(adata.obs["subcluster_r01"] == group) & (adata.obs["model"].isin(["Naive", "IS"]))].copy()
    if sub.obs["model"].nunique() < 2 or sub.n_obs < MIN_CELLS:
        return pd.DataFrame()
    sc.tl.rank_genes_groups(sub, groupby="model", groups=["IS"], reference="Naive", method="wilcoxon")
    df = sc.get.rank_genes_groups_df(sub, group="IS")
    df = df.dropna(subset=["names", "pvals_adj", "logfoldchanges"])
    return df[(df["pvals_adj"] < PADJ) & (df["logfoldchanges"].abs() >= LOGFC)]


def enrich_genes(genes: list[str]) -> pd.DataFrame:
    import gseapy as gp

    if len(genes) < MIN_DEG:
        return pd.DataFrame()
    enr = gp.enrichr(gene_list=genes, gene_sets=GENE_SETS, organism="mouse", outdir=None, cutoff=0.5)
    res = enr.results
    if res is None or res.empty:
        return pd.DataFrame()
    res = res[res["Adjusted P-value"] < 0.05].copy()
    res["neglog10_padj"] = -np.log10(res["Adjusted P-value"].clip(lower=1e-300))
    return res.sort_values("Adjusted P-value")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BARS_DIR.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(H5AD)
    clusters = sorted(
        adata.obs["subcluster_r01"].astype(str).value_counts().pipe(lambda s: s[s >= MIN_CELLS].index).tolist(),
        key=lambda x: (0 if x.startswith("Satglia_") else 1 if x.startswith("PEP_") else 2, x),
    )
    pal = subcluster_colors()

    cached = OUT_DIR / "enrichr_all_subclusters.csv"
    force = "--force" in sys.argv
    full = pd.DataFrame()
    if cached.exists() and not force:
        old = pd.read_csv(cached)
        if set(old["subcluster_r01"].unique()) == set(clusters):
            full = old[(old["direction"] == "IS_up") & old["subcluster_r01"].isin(clusters)]

    if full.empty:
        all_rows = []
        for cl in clusters:
            sig = deg_is_vs_naive(adata, cl)
            if sig.empty:
                continue
            up = sig[sig["logfoldchanges"] > 0]["names"].tolist()
            enr = enrich_genes(up)
            if enr.empty:
                continue
            enr = enr.head(TOP_TERMS).copy()
            enr["subcluster_r01"] = cl
            enr["direction"] = "IS_up"
            all_rows.append(enr)
        if not all_rows:
            print("No enrichment results")
            return
        full = pd.concat(all_rows, ignore_index=True)
        full.to_csv(cached, index=False)

    manifest = []
    # Remove bar charts for clusters no longer present
    for old_png in BARS_DIR.glob("gsea_bar_*.png"):
        cl_guess = old_png.stem.replace("gsea_bar_", "")
        if cl_guess not in clusters and cl_guess.replace("_", " ") not in clusters:
            old_png.unlink(missing_ok=True)

    for cl in clusters:
        bar = full[full["subcluster_r01"] == cl].nlargest(TOP_TERMS, "neglog10_padj")
        if bar.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        color = pal.get(cl, lineage_base_color(cl))
        ax.barh(bar["Term"].str[:55], bar["neglog10_padj"], color=color)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(adj p)")
        ax.set_title(f"{cl} — IS vs Naive (top pathways)")
        plt.tight_layout()
        fname = f"gsea_bar_{safe_name(cl)}.png"
        fig.savefig(BARS_DIR / fname, dpi=180, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"subcluster_r01": cl, "figure": f"fig/gsea_bars/{fname}"})
        print(f"Wrote {BARS_DIR / fname}")

    pd.DataFrame(manifest).to_csv(BARS_DIR / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
