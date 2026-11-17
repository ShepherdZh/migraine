import sys
#!/usr/bin/env python3
"""Standalone barplots for web report: composition, IEG activation, scDRS."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
WEB_FIG = ROOT / "results/reports/web/fig"
PAPER = ["Naive", "IS", "CSD"]
PALETTE = {"Naive": "#1f77b4", "IS": "#ff7f0e", "CSD": "#2ca02c"}


def save(fig, name: str) -> None:
    WEB_FIG.mkdir(parents=True, exist_ok=True)
    p = WEB_FIG / name
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")


def composition_barplot() -> None:
    comp = pd.read_csv(
        ROOT / "results/figures/condition_specific/reintegrated/Table_reintegrated_subtype_counts.csv",
        index_col=0,
    )
    for m in PAPER:
        if m not in comp.columns:
            comp[m] = 0
    comp = comp[PAPER]
    frac = comp.div(comp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    frac = frac.loc[comp.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 5))
    frac.plot(kind="bar", stacked=True, ax=ax, color=[PALETTE[m] for m in PAPER], width=0.85)
    ax.set_ylabel("Fraction within subtype")
    ax.set_xlabel("Cell subtype")
    ax.set_title("Condition composition by cell subtype (Naive / IS / CSD)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.legend(title="model", fontsize=8)
    save(fig, "barplot_subtype_composition.png")


def ieg_barplot() -> None:
    """IEG activation fraction + Fisher OR (IS vs Naive) — paper readout is activation, not composition."""
    df = pd.read_csv(ROOT / "results/condition_composition/IEG_activation_by_subtype_model.csv")
    df = df[df["model"].isin(PAPER)].copy()
    pivot = df.pivot(index="subtype", columns="model", values="frac").fillna(0)
    for m in PAPER:
        if m not in pivot.columns:
            pivot[m] = 0.0
    pivot = pivot[PAPER]

    fisher = pd.read_csv(ROOT / "results/condition_composition/fisher_IS_vs_Naive_IEG_activation.csv")
    fisher = fisher.set_index("subtype")

    sub = pivot.sort_values("IS", ascending=False)
    fsub = fisher.reindex(sub.index).dropna(subset=["OR"])
    fsub = fsub[fsub["OR"] > 0]
    colors = ["#3fb950" if fisher.loc[x, "p"] < 0.05 else "#8b949e" for x in fsub.index]

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.barh(fsub.index, np.log10(fsub["OR"]), color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("log10(OR) IS vs Naive IEG activation", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    save(fig, "barplot_ieg_activation.png")


def scdrs_barplot() -> None:
    df = pd.read_csv(ROOT / "results/figures/paper/Table_scdrs_group_assoc.csv")
    df = df.sort_values("assoc_mcz", ascending=False)
    colors = ["#3fb950" if z > 0 else "#f85149" for z in df["assoc_mcz"]]

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.barh(df["group"].astype(str), df["assoc_mcz"], color=colors)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("scDRS association z-score (migraine GWAS)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    save(fig, "barplot_scdrs_group_assoc.png")


def main():
    ieg_barplot()
    scdrs_barplot()


if __name__ == "__main__":
    main()
