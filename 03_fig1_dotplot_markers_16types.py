import sys
#!/usr/bin/env python3
"""Marker dotplot — all subtypes; row/column order aligned (diagonal pattern)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
SRC = ROOT / "results/objects/mouse_rna_Naive_IS_CSD_reintegrated.h5ad"
OUT = ROOT / "results/figures/paper"
WEB_FIG = ROOT / "results/reports/web/fig"

# Row order = marker group order (top-left → bottom-right diagonal)
SUBTYPE_ORDER = [
    "Satglia", "PEP", "NP", "TRPM8", "cLTMR", "SST",
    "NF1", "NF2", "NF3",
    "Schwann_N", "Schwann_M",
    "Fibroblast_Dcn", "Fibroblast_Mgp",
    "Vascular", "Immune", "Injured",
]

SUBTYPE_MARKERS: dict[str, list[str]] = {
    "Satglia": ["Prdm16", "Fabp7", "Gpm6b", "Ptn"],
    "PEP": ["Calca", "Calcb", "Ramp1", "Tac1"],
    "NP": ["Mrgprd", "P2rx3", "Scn10a"],
    "TRPM8": ["Trpm8", "Tmem100", "Kcnk18"],
    "cLTMR": ["Th", "Slc17a8", "Slc6a17"],
    "SST": ["Sst", "Nts", "Calb2"],
    "NF1": ["Nefh", "Nefm", "Nefl"],
    "NF2": ["Nefm", "Nefl", "Nefh"],
    "NF3": ["Nefl", "Nefm", "Nefh"],
    "Schwann_N": ["Mpz", "Mbp", "Plp1"],
    "Schwann_M": ["Ngfr", "Cdh19", "Sox10"],
    "Fibroblast_Dcn": ["Dcn", "Col1a1", "Pdgfra"],
    "Fibroblast_Mgp": ["Mgp", "Col3a1", "Dcn"],
    "Vascular": ["Pecam1", "Cldn5", "Flt1"],
    "Immune": ["Ptprc", "Cd68", "C1qa"],
    "Injured": ["Atf3", "Gal", "Sprr1a"],
}

# Manuscript Fig. 1c (rotated): keep labels readable after composite scaling.
DOTPLOT_TICK_PT = 18
DOTPLOT_LEGEND_PT = 16


def _bump_dotplot_fonts(fig, *, tick: float = DOTPLOT_TICK_PT, legend: float = DOTPLOT_LEGEND_PT) -> None:
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=tick)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(tick)
        if ax.get_title():
            ax.set_title(ax.get_title(), fontsize=legend)
        for text in ax.texts:
            if text.get_fontsize() < tick:
                text.set_fontsize(tick)
        leg = ax.get_legend()
        if leg is not None:
            for text in leg.get_texts():
                text.set_fontsize(legend)
            if leg.get_title() is not None:
                leg.get_title().set_fontsize(legend)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WEB_FIG.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(SRC)
    adata = adata[adata.obs["model"].isin(["Naive", "IS", "CSD"])].copy()

    present = [st for st in SUBTYPE_ORDER if st in set(adata.obs["subtype"].astype(str))]
    var_names = {
        st: [g for g in SUBTYPE_MARKERS[st] if g in adata.var_names]
        for st in present
    }
    var_names = {k: v for k, v in var_names.items() if v}

    adata = adata[adata.obs["subtype"].astype(str).isin(present)].copy()
    adata.obs["subtype"] = pd.Categorical(
        adata.obs["subtype"].astype(str), categories=list(var_names.keys()), ordered=True
    )

    n_genes = sum(len(v) for v in var_names.values())
    n_rows = len(var_names)

    dp = sc.pl.dotplot(
        adata,
        var_names=var_names,
        groupby="subtype",
        categories_order=list(var_names.keys()),
        standard_scale="var",
        show=False,
        return_fig=True,
        figsize=(max(16, n_genes * 0.62), max(11, n_rows * 0.55)),
        dendrogram=False,
    )
    dp.make_figure()
    _bump_dotplot_fonts(plt.gcf())
    plt.tight_layout()
    for dest in (
        OUT / "Fig_dotplot_markers_all_subtypes.png",
        WEB_FIG / "dotplot_markers_all_subtypes.png",
    ):
        plt.savefig(dest, dpi=220, bbox_inches="tight")
        print(f"Wrote {dest}")
    plt.close()


if __name__ == "__main__":
    main()
