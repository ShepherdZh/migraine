import sys
#!/usr/bin/env python3
"""SCENIC+ Prdm16 eRegulon + AUCell figures for web report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
OUT = ROOT / "results/reports/web/fig"
EGRN = ROOT / "results/scenicplus/mouse/run/egrn_full"


def plot_prdm16_star_eregulon(ed: pd.DataFrame, n_targets: int = 14) -> None:
    """Classic SCENIC+ style: TF center + straight lines to downstream targets."""
    sub = ed[(ed["TF"] == "Prdm16") & (ed["regulation"] == 1)].copy()
    sub = sub.sort_values("rho_TF2G", ascending=False).drop_duplicates("Gene")
    sub = sub[sub["Gene"] != "Prdm16"]
    top = sub.head(n_targets - 1)
    if "Ptn" not in top["Gene"].values:
        ptn = sub[sub["Gene"] == "Ptn"]
        if not ptn.empty:
            top = pd.concat([top, ptn]).drop_duplicates("Gene")
    genes = top["Gene"].tolist()
    n = len(genes)
    angles = np.linspace(np.pi / 2, 2.5 * np.pi, n, endpoint=False)
    radius = 1.05
    pos = {g: (radius * np.cos(a), radius * np.sin(a)) for g, a in zip(genes, angles)}

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor="white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")

    for g in genes:
        lw = 2.2 if g == "Ptn" else 1.0
        color = "#d97706" if g == "Ptn" else "#94a3b8"
        ax.plot([0, pos[g][0]], [0, pos[g][1]], color=color, lw=lw, zorder=1, solid_capstyle="round")

    ax.scatter([0], [0], s=9800, c="#2563eb", zorder=3, edgecolors="#1e293b", linewidths=1.5)
    ax.text(0, 0, "Prdm16", ha="center", va="center", fontsize=18, fontweight="bold",
            fontstyle="italic", color="white", zorder=4)

    for g in genes:
        x, y = pos[g]
        is_ptn = g == "Ptn"
        ax.scatter([x], [y], s=1400 if is_ptn else 900, c="#d97706" if is_ptn else "#64748b",
                   zorder=3, edgecolors="#1e293b", linewidths=1.0)
        label_r = 1.52
        lx, ly = label_r * np.cos(np.arctan2(y, x)), label_r * np.sin(np.arctan2(y, x))
        ax.text(lx, ly, g, ha="center", va="center", fontsize=20 if is_ptn else 18,
                fontstyle="italic", fontweight="bold" if is_ptn else "normal", color="#1e293b", zorder=4)

    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 1.95)
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_prdm16_star_network.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_prdm16_targets(ed: pd.DataFrame) -> None:
    sub = ed[(ed["TF"] == "Prdm16") & (ed["regulation"] == 1)].copy()
    sub = sub.sort_values("importance_x_abs_rho", ascending=False).drop_duplicates("Gene")
    top = sub.head(15)
    highlight = {"Ptn", "Prdm16", "Apoe", "Fabp7", "Egr1", "Ptn", "Cadm1", "Lama2"}
    colors = ["#58a6ff" if g in highlight else "#484f58" for g in top["Gene"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top["Gene"][::-1], top["importance_x_abs_rho"][::-1], color=colors[::-1], edgecolor="#30363d")
    ax.set_xlabel("Region–gene score (|importance × ρ|)")
    ax.set_title("Prdm16 direct eRegulon — activating target genes (top 15)")
    ax.axvline(0, color="#666", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_prdm16_targets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prdm16_ptn_network(ed: pd.DataFrame) -> None:
    """Prdm16 → Ptn (+ context genes) mini network."""
    genes = ["Prdm16", "Ptn", "Sdc3", "Calca", "Nfia", "Ebf1"]
    G = nx.DiGraph()
    G.add_node("Prdm16")
    G.add_node("Ptn")
    G.add_edge("Prdm16", "Ptn", label="eRegulon direct (+)")
    if not ed[(ed["TF"] == "Nfia") & (ed["Gene"] == "Ptn")].empty:
        G.add_edge("Nfia", "Ptn", label="eRegulon (+)")
    if not ed[(ed["TF"] == "Ebf1") & (ed["Gene"] == "Cadm1")].empty:
        G.add_node("Cadm1")
        G.add_edge("Ebf1", "Cadm1", label="eRegulon (+)")
    G.add_node("PEP")
    G.add_edge("Ptn", "PEP", label="CellChat\n(paracrine)")
    G.add_node("CGRP")
    G.add_edge("PEP", "CGRP", label="Calca\n(PEP identity)")
    pos = {
        "Prdm16": (0, 1),
        "Nfia": (-1, 0.5),
        "Ebf1": (-1, -0.5),
        "Ptn": (1, 0.5),
        "Cadm1": (0, -0.5),
        "PEP": (2.2, 0.5),
        "CGRP": (3.2, 0.5),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    node_colors = {
        "Prdm16": "#58a6ff", "Ptn": "#58a6ff", "Nfia": "#8b949e", "Ebf1": "#8b949e",
        "Cadm1": "#3fb950", "PEP": "#3fb950", "CGRP": "#f0883e",
    }
    nx.draw_networkx_nodes(G, pos, node_color=[node_colors.get(n, "#888") for n in G.nodes], node_size=2800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#aaa", arrows=True, arrowsize=18, ax=ax, connectionstyle="arc3,rad=0.1")
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.set_title("SCENIC+ Prdm16 → Ptn eRegulon bridged to PEP / CGRP (Satglia compartment)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_prdm16_ptn_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aucell_by_subtype() -> None:
    import mudata as md
    acc = md.read_h5mu(EGRN / "ACC_GEX.h5mu")
    auc = md.read_h5mu(EGRN / "AUCell_direct.h5mu")
    sub = acc.obs["scRNA:subtype"]
    gb = auc.mod["Gene_based"]
    cols = [c for c in gb.var_names if c.startswith("Prdm16")]
    if not cols:
        return
    X = gb[:, cols].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    # Merge +/+ and +/- modules: per-metacell max (union regulon activity)
    combined = np.max(X, axis=1)
    df = pd.DataFrame({"Prdm16": combined, "subtype": sub.values})
    means = df.groupby("subtype")["Prdm16"].mean().sort_values(ascending=False)
    order = means.index.tolist()
    colors = ["#58a6ff" if s == "Satglia" else "#484f58" for s in order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(order))
    ax.bar(x, means.values, color=colors, edgecolor="#30363d")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("AUCell score (metacell mean)")
    ax.set_title("Prdm16 eRegulon activity by RNA subtype (combined direct modules)")
    ax.axhline(0, color="#666", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "scenicplus_prdm16_aucell_subtype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ed = pd.read_csv(EGRN / "eRegulon_direct.tsv", sep="\t")
    plot_prdm16_star_eregulon(ed)
    try:
        plot_aucell_by_subtype()
    except ImportError:
        print("Skip AUCell figure (mudata not installed)")
    print(f"Wrote Prdm16 SCENIC+ figures → {OUT}")


if __name__ == "__main__":
    main()
