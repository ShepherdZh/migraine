#!/usr/bin/env python3
"""Monocle3 pseudotime: stage-ordered heatmap (diagonal pattern) + Spearman curves."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.stats import spearmanr

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from color_palette import MODEL_COLORS  # noqa: E402

WEB_FIG = ROOT / "results/reports/web/fig"

MIGRAINE_GENES = [
    "Calca", "Calcb", "Ramp1", "Tac1", "Scn10a", "Ptn", "Cadm1", "Prdm16",
    "Atf3", "Fos", "Runx1", "Fabp7", "Dcn",
]

N_BINS = 12
GENES_PER_BIN = 5
MIN_CELLS_BIN = 8
MIN_EXPR_FRAC = 0.05
MIN_Z_PEAK = 0.4

# Genes to highlight on Satglia pseudotime heatmap (report-facing)
HEATMAP_HIGHLIGHT = {
    "Fabp7", "Ptn", "Runx1", "Prdm16", "Fos", "Atf3", "Cadm1", "Dcn", "Egr1",
}


def load_lineage(lineage: str):
    base = ROOT / f"results/monocle3/{lineage}"
    mat = mmread(base / "counts.mtx.gz").T.tocsc()
    genes = pd.read_csv(base / "gene_metadata.csv")["gene_short_name"].astype(str).tolist()
    meta = pd.read_csv(base / "cell_metadata.csv", index_col=0)
    pt = pd.read_csv(base / "pseudotime.csv").set_index("cell")["pseudotime"]
    cells = [c for c in meta.index if c in pt.index and np.isfinite(pt[c])]
    cell_idx = [list(meta.index).index(c) for c in cells]
    pt_vals = pt.loc[cells].values.astype(float)
    return mat, genes, cell_idx, pt_vals


def binned_z_matrix(mat, genes, cell_idx, pt_vals, gene_pool: list[str], n_bins: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    order = np.argsort(pt_vals)
    pt_sorted = pt_vals[order]
    bins = np.linspace(pt_sorted.min(), pt_sorted.max(), n_bins + 1)
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins)]

    rows, kept = [], []
    for gene in gene_pool:
        if gene not in genes:
            continue
        gi = genes.index(gene)
        x = np.asarray(mat[cell_idx, gi].todense()).ravel().astype(float)[order]
        row = []
        for i in range(n_bins):
            m = (pt_sorted >= bins[i]) & (pt_sorted < bins[i + 1]) if i < n_bins - 1 else (
                (pt_sorted >= bins[i]) & (pt_sorted <= bins[i + 1])
            )
            row.append(float(np.mean(x[m])) if m.sum() >= MIN_CELLS_BIN else np.nan)
        arr_row = np.array(row, dtype=float)
        ok = np.isfinite(arr_row)
        if ok.sum() >= 3:
            mu, sd = arr_row[ok].mean(), arr_row[ok].std()
            arr_row[ok] = (arr_row[ok] - mu) / (sd + 1e-9)
        rows.append(arr_row)
        kept.append(gene)

    return np.array(rows), kept, np.array(bin_centers)


def pick_stage_ordered_genes(zmat: np.ndarray, gene_list: list[str], n_bins: int) -> list[str]:
    """One block of marker genes per pseudotime bin → diagonal heatmap."""
    peak_bin = np.nanargmax(zmat, axis=1)
    peak_z = zmat.max(axis=1)
    selected: list[str] = []
    used: set[str] = set()

    # Force migraine genes if they peak clearly in one bin
    for i, g in enumerate(gene_list):
        if g in MIGRAINE_GENES and peak_z[i] >= MIN_Z_PEAK and g not in used:
            selected.append((int(peak_bin[i]), -peak_z[i], g))
            used.add(g)

    for b in range(n_bins):
        idx = np.where(peak_bin == b)[0]
        order = idx[np.argsort(-peak_z[idx])]
        n = 0
        for i in order:
            g = gene_list[i]
            if g in used or peak_z[i] < MIN_Z_PEAK:
                continue
            selected.append((b, -peak_z[i], g))
            used.add(g)
            n += 1
            if n >= GENES_PER_BIN:
                break

    selected.sort(key=lambda x: (x[0], x[1]))
    return [g for _, _, g in selected]


def rank_genes(mat, genes, cell_idx, pt_vals) -> pd.DataFrame:
    rows = []
    for gi, gene in enumerate(genes):
        x = np.asarray(mat[cell_idx, gi].todense()).ravel().astype(float)
        if np.mean(x > 0) < MIN_EXPR_FRAC:
            continue
        rho, pval = spearmanr(pt_vals, x)
        if np.isfinite(rho):
            rows.append({"gene": gene, "rho": rho, "pval": pval})
    return pd.DataFrame(rows).sort_values("rho", ascending=False)


def plot_stage_heatmap(lineage: str, zmat: np.ndarray, gene_list: list[str], centers: np.ndarray, peak_bins: list[int]):
    n_genes = len(gene_list)
    fig_h = max(8, n_genes * 0.32)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    im = ax.imshow(zmat, aspect="auto", cmap="RdYlBu_r", vmin=-2, vmax=2.5)
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([f"{c:.0f}" for c in centers], fontsize=15, rotation=0)
    ax.set_xlabel("Pseudotime bin (Naive → IS/CSD)", fontsize=16)
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_list, fontsize=16, fontstyle="italic")
    for i, g in enumerate(gene_list):
        if g in HEATMAP_HIGHLIGHT:
            ax.get_yticklabels()[i].set_color("#c0392b")
            ax.get_yticklabels()[i].set_fontweight("bold")
            ax.get_yticklabels()[i].set_fontsize(18)
    prev = -1
    for i, b in enumerate(peak_bins + [999]):
        if b != prev and i > 0:
            ax.axhline(i - 0.5, color="white", lw=0.6, alpha=0.5)
        prev = b
    hl = ", ".join(g for g in gene_list if g in HEATMAP_HIGHLIGHT)
    ax.set_title(
        f"{lineage} — stage genes along pseudotime\n(red: {hl})",
        fontsize=16,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03)
    cbar.set_label("z-score (expr in bin)", fontsize=14)
    cbar.ax.tick_params(labelsize=13)
    plt.tight_layout()
    out = WEB_FIG / f"monocle3_{lineage}_pseudotime_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_curves(lineage, mat, genes, cell_idx, pt_vals, gene_list, rank_df):
    fig, axes = plt.subplots(len(gene_list), 1, figsize=(7, 1.8 * len(gene_list)), sharex=True)
    if len(gene_list) == 1:
        axes = [axes]
    rho_map = rank_df.set_index("gene")["rho"].to_dict()
    order = np.argsort(pt_vals)
    for ax, gene in zip(axes, gene_list):
        gi = genes.index(gene)
        x = np.asarray(mat[cell_idx, gi].todense()).ravel().astype(float)
        rho = rho_map.get(gene, spearmanr(pt_vals, x)[0])
        ax.scatter(pt_vals[order], x[order], s=2, alpha=0.1, c="gray", rasterized=True)
        bins = np.linspace(pt_vals.min(), pt_vals.max(), 20)
        bx, by = [], []
        for i in range(len(bins) - 1):
            m = (pt_vals >= bins[i]) & (pt_vals < bins[i + 1])
            if m.sum() >= 5:
                bx.append((bins[i] + bins[i + 1]) / 2)
                by.append(float(np.mean(x[m])))
        if bx:
            ax.plot(bx, by, color=MODEL_COLORS["IS"], lw=2)
        ax.set_ylabel(gene, fontsize=8)
        ax.text(0.02, 0.82, f"ρ={rho:.2f}", transform=ax.transAxes, fontsize=7)
    axes[-1].set_xlabel("Pseudotime")
    fig.suptitle(f"{lineage} — top stage genes (Spearman with pseudotime)", fontsize=10)
    plt.tight_layout()
    out = WEB_FIG / f"monocle3_{lineage}_pseudotime_genes.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    WEB_FIG.mkdir(parents=True, exist_ok=True)
    (WEB_FIG.parent / "data").mkdir(parents=True, exist_ok=True)
    for lineage in ["satglia", "pep", "fibroblast_dcn"]:
        mat, genes, cell_idx, pt_vals = load_lineage(lineage)
        if len(cell_idx) < 30:
            continue
        rank_df = rank_genes(mat, genes, cell_idx, pt_vals)
        rank_df.to_csv(WEB_FIG.parent / "data" / f"monocle3_{lineage}_pseudotime_spearman.csv", index=False)
        pool = rank_df.head(400)["gene"].tolist()
        pool = list(dict.fromkeys([g for g in MIGRAINE_GENES if g in genes] + pool))
        zmat, kept, centers = binned_z_matrix(mat, genes, cell_idx, pt_vals, pool, N_BINS)
        stage_genes = pick_stage_ordered_genes(zmat, kept, N_BINS)
        if len(stage_genes) < 5:
            continue
        gi_map = {g: i for i, g in enumerate(kept)}
        sub = zmat[[gi_map[g] for g in stage_genes]]
        peak_bins = [int(np.argmax(sub[i])) for i in range(len(stage_genes))]
        plot_stage_heatmap(lineage, sub, stage_genes, centers, peak_bins)


if __name__ == "__main__":
    main()
