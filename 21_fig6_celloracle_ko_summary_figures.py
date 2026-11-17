#!/usr/bin/env python3
"""CellOracle report figures: cosine bar + thick-arrow overlay + L0-internal axis.

Monocle3 embedding is built on all Satglia, but the activation axis paired with
CellOracle is *inside Satglia_L0*: bottom ≈ Naive, top ≈ a minority of IS cells.
PT gradient and cosine are therefore fitted on L0 cells only.

Figure text is English to avoid CJK tofu in matplotlib. Captions live in the PDF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from sklearn.linear_model import LinearRegression

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from color_palette import MODEL_COLORS, apply_subcluster_merge, subcluster_colors  # noqa: E402

OUT = ROOT / "results/reports/web/fig"
KO = ROOT / "results/celloracle/satglia_pep/ko_monocle3_ranked.csv"
MONO = ROOT / "results/monocle3/satglia"


def _open_umap_axes(ax, *, equal: bool = True) -> None:
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_frame_on(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _draw_arrow(ax, x0, y0, dx, dy, color, lw=3.4, ls="solid", z=12):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x0 + dx, y0 + dy),
        arrowstyle="-|>", mutation_scale=26, lw=lw + 2.4,
        color="white", linestyle="solid", zorder=z, shrinkA=0, shrinkB=0,
        joinstyle="miter", capstyle="round",
    ))
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x0 + dx, y0 + dy),
        arrowstyle="-|>", mutation_scale=22, lw=lw,
        color=color, linestyle=ls, zorder=z + 1, shrinkA=0, shrinkB=0,
        joinstyle="miter", capstyle="round",
    ))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0 or not np.isfinite(na) or not np.isfinite(nb):
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _l0_naive_is_reference(xy: np.ndarray, l0: np.ndarray, model: np.ndarray):
    m_n = l0 & (model == "Naive")
    m_i = l0 & (model == "IS")
    c_naive = np.nanmean(xy[m_n], axis=0)
    c_is = np.nanmean(xy[m_i], axis=0)
    axis = c_is - c_naive
    return c_naive, c_is, axis


def load_satglia() -> dict:
    umap = pd.read_csv(MONO / "umap.csv").set_index("cell")
    meta = pd.read_csv(MONO / "cell_metadata.csv").set_index("cell")
    pt = pd.read_csv(MONO / "pseudotime.csv").set_index("cell")["pseudotime"]
    ko = pd.read_csv(KO)

    cells = umap.index.intersection(meta.index).intersection(pt.index)
    xy = umap.loc[cells, ["UMAP_1", "UMAP_2"]].to_numpy(float)
    sub = apply_subcluster_merge(meta.loc[cells, "subcluster_r01"]).astype(str).to_numpy()
    model = meta.loc[cells, "model"].astype(str).to_numpy()
    ptv = pt.loc[cells].to_numpy(float)
    l0 = sub == "Satglia_L0"

    c_naive, c_is, axis_ref = _l0_naive_is_reference(xy, l0, model)

    ko = ko.copy()
    cos_l0 = []
    for _, r in ko.iterrows():
        if pd.isna(r.get("shift_x")) or pd.isna(r.get("shift_y")):
            cos_l0.append(np.nan)
        else:
            cos_l0.append(cosine_sim(np.array([r["shift_x"], r["shift_y"]], float), axis_ref))
    ko["cosine_vs_l0_internal_pt"] = cos_l0
    ko["cosine_vs_satglia_internal_pt"] = cos_l0
    return {
        "xy": xy, "sub": sub, "model": model, "ptv": ptv, "l0": l0,
        "axis_ref": axis_ref, "c_naive": c_naive, "c_is": c_is,
        "grad_l0": axis_ref, "ko": ko,
    }


def plot_cosine_bar(ko: pd.DataFrame) -> None:
    df = ko.dropna(subset=["cosine_vs_l0_internal_pt"]).copy()
    df = df.sort_values("cosine_vs_l0_internal_pt", ascending=True)
    colors = []
    for _, r in df.iterrows():
        if r["TF"] == "Prdm16":
            colors.append("#1d4ed8")
        elif r["cosine_vs_l0_internal_pt"] > 0.2:
            colors.append("#059669")
        elif r["cosine_vs_l0_internal_pt"] < -0.2:
            colors.append("#dc2626")
        else:
            colors.append("#94a3b8")
    fig, ax = plt.subplots(figsize=(7.6, 4.2), facecolor="white")
    ax.barh(df["TF"], df["cosine_vs_l0_internal_pt"], color=colors, edgecolor="#1e293b", height=0.7)
    ax.axvline(0, color="#334155", lw=1)
    ax.set_xlabel("Alignment with the naïve → IS centroid axis", fontsize=7)
    ax.set_title("In silico KO versus the naïve-to-IS direction", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(
        handles=[
            Patch(facecolor="#1d4ed8", edgecolor="#1e293b", label="Prdm16"),
            Patch(facecolor="#059669", edgecolor="#1e293b", label="Same direction as Naive→IS"),
            Patch(facecolor="#dc2626", edgecolor="#1e293b", label="Opposite to Naive→IS"),
        ],
        loc="lower right", fontsize=6, frameon=True,
    )
    fig.tight_layout()
    fig.savefig(OUT / "celloracle_cosine_satglia_bar.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_l0_model_axis(d: dict) -> None:
    """L0 coloured by condition; dashed chord Naive centroid → IS centroid."""
    xy, model, l0 = d["xy"], d["model"], d["l0"]
    c_naive, c_is, axis_ref = d["c_naive"], d["c_is"], d["axis_ref"]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor="white")
    # faint L1/L2 for orientation
    ax.scatter(xy[~l0, 0], xy[~l0, 1], c="#e2e8f0", s=5, alpha=0.45, linewidths=0, rasterized=True, zorder=1)
    for lab in ["CSD", "IS", "Naive"]:
        m = l0 & (model == lab)
        if not m.any():
            continue
        ax.scatter(
            xy[m, 0], xy[m, 1],
            c=MODEL_COLORS.get(lab, "#94a3b8"), s=10,
            alpha=0.85, linewidths=0, rasterized=True, zorder=3 if lab != "Naive" else 2,
            label=f"L0 {lab} (n={int(m.sum())})",
        )
    span = float(np.percentile(np.ptp(xy[l0], axis=0), 50))
    nv = np.linalg.norm(axis_ref)
    if nv > 0:
        u = axis_ref / nv * (0.55 * span)
        _draw_arrow(ax, c_naive[0], c_naive[1], u[0], u[1], "#64748b", lw=2.8, ls="dashed")
        ax.scatter([c_naive[0], c_is[0]], [c_naive[1], c_is[1]], s=42, c="#64748b", edgecolors="#0f172a", zorder=9)
        ax.text(
            (c_naive[0] + c_is[0]) / 2, (c_naive[1] + c_is[1]) / 2,
            "Naive → IS (L0 centroids)", fontsize=16, fontweight="bold", color="#475569",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#64748b", alpha=0.95),
            zorder=15,
        )
    y_naive = xy[l0 & (model == "Naive"), 1]
    y_is = xy[l0 & (model == "IS"), 1]
    x_l0 = xy[l0, 0]
    x_right = float(np.max(x_l0)) + 0.4
    if len(y_is):
        ax.annotate(
            "top: IS-mixed",
            xy=(float(np.median(xy[l0 & (model == "IS"), 0])), float(np.median(y_is))),
            xytext=(x_right, float(np.median(y_is))),
            fontsize=16, color="#c2410c",
            arrowprops=dict(arrowstyle="-", color="#c2410c", lw=0.8),
            va="center",
        )
    if len(y_naive):
        ax.annotate(
            "bottom: Naive",
            xy=(float(np.median(xy[l0 & (model == "Naive"), 0])), float(np.percentile(y_naive, 8))),
            xytext=(x_right, float(np.percentile(y_naive, 8))),
            fontsize=16, color="#1d4ed8",
            arrowprops=dict(arrowstyle="-", color="#1d4ed8", lw=0.8),
            va="center",
        )
    ax.set_title("Naïve-to-IS axis in Satglia (grey = other subclusters)", fontsize=16)
    ax.set_xlabel("Monocle3 UMAP_1", fontsize=18)
    ax.set_ylabel("Monocle3 UMAP_2", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(loc="lower left", fontsize=16, markerscale=2.2, frameon=True)
    _open_umap_axes(ax)
    fig.tight_layout()
    dest = OUT / "monocle3_satglia_l0_model.png"
    fig.savefig(dest, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {dest}")


def _draw_l0_ko_on_ax(
    ax,
    d: dict,
    tf: str,
    *,
    style: str = "simple",
    ko_color: str | None = None,
    axis_fs: float = 9,
    tick_fs: float = 8,
    ko_fs: float = 9.5,
) -> None:
    """style=simple: uniform L0 (control TFs). style=contextual: L0 coloured by Naive/IS/CSD (Prdm16)."""
    xy, sub, l0, axis_ref, model, ko = (
        d["xy"], d["sub"], d["l0"], d["axis_ref"], d["model"], d["ko"].set_index("TF"),
    )
    c_naive, c_is = d["c_naive"], d["c_is"]
    if tf not in ko.index:
        return
    colors = {"Prdm16": "#2563eb", "Mef2d": "#7c3aed", "Zeb2": "#059669", "Pou4f1": "#dc2626"}
    color = ko_color or colors.get(tf, "#334155")
    cx, cy = np.nanmean(xy[l0], axis=0)
    span = float(np.percentile(np.ptp(xy, axis=0), 50))
    arrow_len = 0.52 * span
    vec = ko.loc[tf, ["shift_x", "shift_y"]].to_numpy(float)

    ax.scatter(xy[~l0, 0], xy[~l0, 1], c="#cbd5e1", s=5, alpha=0.4, linewidths=0, rasterized=True, zorder=1)
    if style == "contextual":
        for lab in ["Naive", "IS", "CSD"]:
            m = l0 & (model == lab)
            if not m.any():
                continue
            ax.scatter(
                xy[m, 0], xy[m, 1], c=MODEL_COLORS.get(lab, "#94a3b8"), s=10,
                alpha=0.88, linewidths=0, rasterized=True, zorder=3,
                label=f"L0 {lab}",
            )
        ax.legend(loc="lower left", fontsize=7, markerscale=2, frameon=True)
    else:
        ax.scatter(xy[l0, 0], xy[l0, 1], c="#1e3a8a", s=8, alpha=0.82, linewidths=0, rasterized=True, zorder=2)

    ax.scatter([cx], [cy], s=88, c="white", edgecolors="#0f172a", linewidths=1.6, zorder=8)
    gnv = np.linalg.norm(axis_ref)
    if gnv > 0:
        u = axis_ref / gnv * arrow_len
        _draw_arrow(ax, c_naive[0], c_naive[1], u[0], u[1], "#64748b", lw=2.6, ls="dashed")
        if style == "contextual":
            ax.scatter([c_naive[0], c_is[0]], [c_naive[1], c_is[1]], s=36, c="#64748b", edgecolors="#0f172a", zorder=9)
    nv = np.linalg.norm(vec)
    if np.isfinite(nv) and nv > 0:
        u = vec / nv * arrow_len
        _draw_arrow(ax, cx, cy, u[0], u[1], color, lw=3.2, ls="solid")
        ax.text(
            cx + u[0] * 1.08, cy + u[1] * 1.08, f"{tf} KO", fontsize=ko_fs, fontweight="bold", color=color,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, alpha=0.95), zorder=15,
        )
    ax.set_xlabel("Monocle3 UMAP_1", fontsize=axis_fs)
    ax.set_ylabel("Monocle3 UMAP_2", fontsize=axis_fs)
    ax.tick_params(labelsize=tick_fs)
    _open_umap_axes(ax)


def _s7_shared_limits(d: dict) -> tuple[float, float, float, float]:
    """Tight on left, right and top; extra room only below for the downward KO arrow."""
    xy = d["xy"]
    span = float(np.percentile(np.ptp(xy, axis=0), 50))
    arrow_len = 0.52 * span
    cx, cy = np.nanmean(xy[d["l0"]], axis=0)
    # Pou4f1-style downward tip plus its label (placed at 1.08 × arrow).
    y_lo = min(float(np.nanmin(xy[:, 1])), cy - arrow_len * 1.08) - 0.18 * arrow_len
    y_hi = float(np.nanmax(xy[:, 1])) + 0.06 * arrow_len
    x_lo = float(np.nanmin(xy[:, 0])) - 0.06 * arrow_len
    x_hi = float(np.nanmax(xy[:, 0])) + 0.16 * arrow_len
    return x_lo, x_hi, y_lo, y_hi


def plot_ko_control_panel(d: dict, tf: str, out_name: str) -> None:
    """Supplementary: simple centroid-arrow layout (positive or negative control)."""
    x_lo, x_hi, y_lo, y_hi = _s7_shared_limits(d)
    fig, ax = plt.subplots(figsize=(5.8, 5.2), facecolor="white")
    _draw_l0_ko_on_ax(ax, d, tf, style="simple", axis_fs=16, tick_fs=14, ko_fs=15)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    koi = d["ko"].set_index("TF")
    cos = float(koi.loc[tf, "cosine_vs_l0_internal_pt"])
    note = "opposite Naive→IS" if cos < 0 else "with Naive→IS"
    ax.set_title(f"{tf} KO  ·  {note}  ·  cos={cos:.2f}", fontsize=15, fontweight="bold")
    dest = OUT / out_name
    fig.savefig(dest, dpi=185, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {dest}")


def plot_ko_overlay(d: dict) -> None:
    xy, sub, l0, grad, ko = d["xy"], d["sub"], d["l0"], d["grad_l0"], d["ko"].set_index("TF")
    pal = subcluster_colors()
    cx, cy = np.nanmean(xy[l0], axis=0)
    span = float(np.percentile(np.ptp(xy, axis=0), 50))
    arrow_len = 0.52 * span

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2), facecolor="white")

    ax = axes[0]
    for lab in ["Satglia_L2", "Satglia_L1", "Satglia_L0"]:
        m = sub == lab
        if not m.any():
            continue
        ax.scatter(
            xy[m, 0], xy[m, 1],
            c=pal.get(lab, "#94a3b8"), s=8 if lab == "Satglia_L0" else 6,
            alpha=0.85 if lab == "Satglia_L0" else 0.55,
            linewidths=0, rasterized=True, zorder=2 if lab == "Satglia_L0" else 1,
            label=lab,
        )
    ax.set_title("Satglia subclusters")
    ax.set_xlabel("Monocle3 UMAP_1")
    ax.set_ylabel("Monocle3 UMAP_2")
    ax.legend(loc="lower right", fontsize=10, markerscale=3, frameon=True,
              title="Knockout arrow: one subcluster")
    _open_umap_axes(ax)

    ax = axes[1]
    _draw_l0_ko_on_ax(ax, d, "Prdm16", style="contextual")
    _open_umap_axes(ax)
    ax.set_title("Prdm16 knockout and naïve-to-IS centroids")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#64748b", lw=2.4, ls="--", label="Dashed: naïve-to-IS centroids"),
            Line2D([0], [0], color="#2563eb", lw=3.0, label="Blue: Prdm16 KO (mean shift)"),
        ],
        loc="upper right", fontsize=7.5, frameon=True,
    )

    fig.tight_layout()
    dest = OUT / "celloracle_ko_on_monocle3_satglia.png"
    fig.savefig(dest, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {dest}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_satglia()
    pr = d["ko"].loc[d["ko"]["TF"] == "Prdm16", "cosine_vs_l0_internal_pt"]
    print(f"Prdm16 cosine vs L0 Naive→IS axis: {float(pr.iloc[0]):.3f}")
    d["ko"].to_csv(OUT / "celloracle_cosine_l0_internal.csv", index=False)
    d["ko"].to_csv(KO, index=False)
    plot_cosine_bar(d["ko"])
    plot_l0_model_axis(d)
    plot_ko_overlay(d)
    for tf in ("Mef2d", "Zeb2", "Pou4f1"):
        plot_ko_control_panel(d, tf, f"celloracle_ko_control_{tf.lower()}.png")
    print(f"Wrote CellOracle summary figures → {OUT}")


if __name__ == "__main__":
    main()
