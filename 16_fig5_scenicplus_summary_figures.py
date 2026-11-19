#!/usr/bin/env python3
import sys
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
OUT = ROOT / "results/reports/web/fig"
R2G_FOCUS = ROOT / "results/scenicplus/mouse/run/egrn_pycistopic/region_to_gene_pycistopic.csv"
R2G_FULL = ROOT / "results/scenicplus/mouse/run/egrn_full/region_to_gene_adj.tsv"
TF_FILE = Path("/opt/NAS/RD/pinned/scenic/allTFs_mm.txt")
RANK_OUT = ROOT / "results/scenicplus/mouse/run/egrn_pycistopic/r2g_gene_rankings.csv"
P2G_MM10 = ROOT / "results/scenicplus/mouse_mm10/egrn/region_to_gene_from_archr.csv"
CALCA_SHARE = ROOT / "results/scenicplus/calca_satglia_pep/genes_sharing_calca_linked_peaks_mm10.csv"

FOCUS = ["Calca", "Calcb", "Ramp1", "Prdm16", "Ptn", "Tac1", "Runx1", "Stat6", "Mef2d"]
HIGHLIGHT = {"Calca", "Calcb", "Prdm16", "Runx1", "Ptn", "Ramp1", "Ebf1"}


def load_full_r2g(min_abs_rho: float = 0.3) -> pd.DataFrame:
    df = pd.read_csv(R2G_FULL, sep="\t")
    if min_abs_rho > 0:
        df = df[df["rho"].abs() >= min_abs_rho].copy()
    return df


def r2g_link_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("target").size().sort_values(ascending=False)


def build_rank_table(min_abs_rho: float = 0.3) -> pd.DataFrame:
    counts = r2g_link_counts(load_full_r2g(min_abs_rho))
    tfs = set(TF_FILE.read_text().split())
    rows = []
    for rank, (gene, n_links) in enumerate(counts.items(), start=1):
        rows.append(
            {
                "rank": rank,
                "gene": gene,
                "n_links": int(n_links),
                "is_tf": gene in tfs,
                "is_focus": gene in FOCUS,
            }
        )
    out = pd.DataFrame(rows)
    RANK_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RANK_OUT, index=False)
    return out


def _bar_colors(genes: pd.Index, *, universe: str) -> list[str]:
    colors = []
    for g in genes:
        if g == "Prdm16":
            colors.append("#ea580c")
        elif g in FOCUS:
            colors.append("#58a6ff")
        elif universe == "genome" and g in HIGHLIGHT:
            colors.append("#58a6ff")
        else:
            colors.append("#64748b")
    return colors


def plot_r2g_top_genome(top_n: int = 20, min_abs_rho: float = 0.3) -> pd.DataFrame:
    """Genome-wide R2G hub ranking (full SCENIC+ adjacencies, not pre-selected genes)."""
    rank = build_rank_table(min_abs_rho)
    top = rank.head(top_n).copy()
    prdm16_rank = int(rank.loc[rank["gene"] == "Prdm16", "rank"].iloc[0]) if (rank["gene"] == "Prdm16").any() else None

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(top))
    ax.barh(y, top["n_links"], color=_bar_colors(top["gene"], universe="genome"), edgecolor="#30363d")
    ax.set_yticks(y)
    ax.set_yticklabels(top["gene"], fontsize=8.5, fontstyle="italic")
    ax.invert_yaxis()
    ax.set_xlabel(f"SCENIC+ region-to-gene links (genome-wide, |ρ|≥{min_abs_rho})")
    title = f"Top {top_n} genes by enhancer linkage density"
    if prdm16_rank is not None:
        title += f"\nPrdm16 rank {prdm16_rank} (not in top {top_n})"
    ax.set_title(title, fontsize=10, fontweight="bold")
    for yi, n in zip(y, top["n_links"]):
        ax.text(n + 1.5, yi, str(int(n)), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"scenicplus_r2g_top{top_n}_genome.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rank


def plot_r2g_top_tfs(top_n: int = 20, min_abs_rho: float = 0.3) -> pd.DataFrame:
    """TF-restricted R2G hub ranking — Prdm16 emerges without pre-selection."""
    rank = build_rank_table(min_abs_rho)
    tf_rank = rank[rank["is_tf"]].reset_index(drop=True)
    tf_rank["tf_rank"] = np.arange(1, len(tf_rank) + 1)
    top = tf_rank.head(top_n).copy()
    prdm16_tf_rank = int(tf_rank.loc[tf_rank["gene"] == "Prdm16", "tf_rank"].iloc[0]) if (tf_rank["gene"] == "Prdm16").any() else None

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    y = np.arange(len(top))
    ax.barh(y, top["n_links"], color=_bar_colors(top["gene"], universe="tf"), edgecolor="#30363d")
    ax.set_yticks(y)
    ax.set_yticklabels(top["gene"], fontsize=8.5, fontstyle="italic")
    ax.invert_yaxis()
    ax.set_xlabel(f"SCENIC+ region-to-gene links (TFs only, |ρ|≥{min_abs_rho})")
    title = f"Top {top_n} transcription factors by enhancer linkage density"
    if prdm16_tf_rank is not None:
        title += f"\nPrdm16 TF rank {prdm16_tf_rank}"
    ax.set_title(title, fontsize=10, fontweight="bold")
    for yi, n in zip(y, top["n_links"]):
        ax.text(n + 1.0, yi, str(int(n)), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"scenicplus_r2g_top{top_n}_tf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tf_rank


def plot_focus_r2g_counts(r2g: pd.DataFrame) -> None:
    counts = r2g.groupby("target").size().reindex(FOCUS).fillna(0).astype(int)
    colors = ["#58a6ff" if g in {"Prdm16", "Runx1", "Ptn"} else "#3fb950" if g in {"Calca", "Calcb", "Ramp1", "Tac1"} else "#8b949e" for g in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="#30363d")
    ax.set_xlabel("SCENIC+ region-to-gene links (pycistopic, focus genes)")
    ax.set_title("Focus-gene enhancer linkage density")
    ax.invert_yaxis()
    for b, v in zip(bars, counts.values):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_focus_gene_r2g_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coreg_network(r2g: pd.DataFrame) -> None:
    reg_by = {g: set(r2g.loc[r2g["target"] == g, "region"]) for g in FOCUS if g in set(r2g["target"])}
    G = nx.Graph()
    for g in reg_by:
        G.add_node(g)
    for a, b in combinations(reg_by, 2):
        n = len(reg_by[a] & reg_by[b])
        if n:
            G.add_edge(a, b, weight=n)
    pos = nx.spring_layout(G, seed=42, k=1.4)
    fig, ax = plt.subplots(figsize=(8, 7))
    node_colors = ["#58a6ff" if n in {"Prdm16", "Runx1", "Ptn"} else "#3fb950" if n in {"Calca", "Calcb", "Ramp1", "Tac1"} else "#f0883e" for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200, ax=ax, edgecolors="#30363d")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    edges = G.edges(data=True)
    widths = [0.5 + 0.25 * d["weight"] for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.55, edge_color="#8b949e", ax=ax)
    edge_labels = {(a, b): str(d["weight"]) for a, b, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title("Shared enhancer peaks among focus genes (pycistopic R2G)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_focus_coreg_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hub_genes(p2g: pd.DataFrame, top_n: int = 20) -> None:
    hubs = p2g["gene"].value_counts().head(top_n)
    colors = ["#58a6ff" if g in HIGHLIGHT else "#484f58" for g in hubs.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(hubs.index[::-1], hubs.values[::-1], color=colors[::-1], edgecolor="#30363d")
    ax.set_xlabel("Linked ATAC peaks (ArchR P2G, mm10)")
    ax.set_title(f"Top {top_n} hub genes by peak linkage")
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_p2g_hub_genes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calca_locus_neighbors() -> None:
    raw = pd.read_csv(CALCA_SHARE)
    df = raw.rename(columns={raw.columns[0]: "gene", raw.columns[1]: "n_shared_peaks"})
    df = df.sort_values("n_shared_peaks", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#3fb950" if g in {"Calca", "Calcb"} else "#58a6ff" if g in {"Cyp2r1", "Insc", "Prdm16"} else "#484f58" for g in df["gene"]]
    ax.barh(df["gene"][::-1], df["n_shared_peaks"][::-1], color=colors[::-1], edgecolor="#30363d")
    ax.set_xlabel("Peaks co-linked with Calca (mm10 ArchR P2G)")
    ax.set_title("chr7 Calca locus — co-regulated neighbors")
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_calca_locus_neighbors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r2g = pd.read_csv(R2G_FOCUS)
    plot_focus_r2g_counts(r2g)
    plot_coreg_network(r2g)
    genome_rank = plot_r2g_top_genome(top_n=20)
    tf_rank = plot_r2g_top_tfs(top_n=20)
    prdm16 = genome_rank.loc[genome_rank["gene"] == "Prdm16"].iloc[0]
    prdm16_tf = tf_rank.loc[tf_rank["gene"] == "Prdm16"].iloc[0]
    print(
        f"Genome-wide Prdm16: rank {int(prdm16['rank'])}, {int(prdm16['n_links'])} links; "
        f"TF-only rank {int(prdm16_tf['tf_rank'])}, {int(prdm16_tf['n_links'])} links"
    )
    p2g = pd.read_csv(P2G_MM10, usecols=["gene"])
    plot_hub_genes(p2g)
    plot_calca_locus_neighbors()
    print(f"Wrote SCENIC+ report figures → {OUT}")


if __name__ == "__main__":
    main()
