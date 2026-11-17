#!/usr/bin/env python3
"""
CellOracle in silico perturbation — Monocle3 lineage UMAP.

Satglia KO/OE and cos(Satglia internal PT) on Satglia-only + Monocle3 Satglia UMAP.
Overlay emphasizes Satglia_L0 (homeostatic) as reference; Prdm16 OE shown directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import LinearRegression

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
from color_palette import subcluster_colors  # noqa: E402

OUT = ROOT / "results/celloracle/satglia_pep"
FOCUS_H5AD = ROOT / "results/objects/mouse_rna_focus_subcluster_r01.h5ad"
TF_TABLE = ROOT / "results/celloracle/mouse_gwas_seed_tfs/Table_prioritized_TFs_for_CellOracle.csv"
SEED_TXT = ROOT / "results/celloracle/celloracle_seed_tfs.txt"
GRN = ROOT / "results/celloracle/mouse/grn_edges.csv"

LINEAGE_CFG = {
    "Satglia": {
        "mono_dir": ROOT / "results/monocle3/satglia",
        "grn_clusters": ["Satglia"],
        "cos_col": "cosine_vs_satglia_internal_pt",
    },
    "PEP": {
        "mono_dir": ROOT / "results/monocle3/pep",
        "grn_clusters": ["PEP"],
        "cos_col": "cosine_vs_pep_internal_pt",
    },
}


def load_seed_tfs(top_n: int = 6) -> list[str]:
    seeds = []
    if TF_TABLE.exists():
        df = pd.read_csv(TF_TABLE)
        seeds.extend(df.sort_values("score", ascending=False).head(top_n)["gene"].tolist())
    if SEED_TXT.exists():
        seeds.extend([x.strip() for x in SEED_TXT.read_text().splitlines() if x.strip()])
    seeds.extend(["Prdm16", "Ptn", "Runx1", "Mef2d"])
    out, seen = [], set()
    for g in seeds:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out[:top_n]


def build_tfdict(grn: pd.DataFrame, var_names: set[str], grn_clusters: list[str]) -> dict[str, list[str]]:
    focus = grn[grn["cluster"].isin(grn_clusters)].copy()
    if focus.empty:
        focus = grn.copy()
    tfdict: dict[str, list[str]] = {}
    for target, tf in focus[["target", "source"]].drop_duplicates().itertuples(index=False):
        if target not in var_names:
            continue
        tfdict.setdefault(target, [])
        if tf not in tfdict[target]:
            tfdict[target].append(tf)
    return tfdict


def supplement_tfdict_from_coexpression(
    adata, tfdict: dict[str, list[str]], seed_tfs: list[str], top_n: int = 80
) -> dict[str, list[str]]:
    missing = [tf for tf in seed_tfs if tf in adata.var_names
               and tf not in {t for regs in tfdict.values() for t in regs}]
    if not missing:
        return tfdict
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    for tf in missing:
        ti = gene_to_idx[tf]
        tf_expr = X[:, ti]
        if np.std(tf_expr) < 1e-8:
            continue
        cors = []
        for g, gi in gene_to_idx.items():
            if g == tf:
                continue
            expr = X[:, gi]
            if np.std(expr) < 1e-8:
                continue
            cors.append((g, float(np.corrcoef(tf_expr, expr)[0, 1])))
        cors.sort(key=lambda x: -abs(x[1]))
        for g, _ in cors[:top_n]:
            tfdict.setdefault(g, [])
            if tf not in tfdict[g]:
                tfdict[g].append(tf)
    return tfdict


def get_raw_counts(adata):
    if adata.raw is not None:
        X = adata.raw[:, adata.var_names].X
    elif "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        X = np.expm1(X)
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X)


def load_monocle3_embedding(mono_dir: Path, cell_names: pd.Index) -> tuple[np.ndarray, pd.Series]:
    umap = pd.read_csv(mono_dir / "umap.csv").set_index("cell")
    pt = pd.read_csv(mono_dir / "pseudotime.csv").set_index("cell")["pseudotime"]
    emb = np.full((len(cell_names), 2), np.nan)
    for i, c in enumerate(cell_names):
        if c in umap.index:
            emb[i] = umap.loc[c, ["UMAP_1", "UMAP_2"]].values
    if np.isnan(emb).any():
        n_miss = int(np.isnan(emb).any(axis=1).sum())
        raise ValueError(f"{mono_dir}: {n_miss} cells missing Monocle3 UMAP coords")
    pt = pt.reindex(cell_names)
    return emb, pt


def pt_gradient(emb: np.ndarray, pseudotime: pd.Series) -> np.ndarray | None:
    ok = pseudotime.notna().values & np.isfinite(pseudotime.values)
    if ok.sum() < 50:
        return None
    reg = LinearRegression().fit(emb[ok], pseudotime.values[ok])
    return reg.coef_.astype(float)


def l0_naive_to_is_axis(
    emb: np.ndarray, l0_mask: np.ndarray, model: np.ndarray,
) -> np.ndarray | None:
    """Reference direction: IS centroid minus Naive centroid within Satglia_L0 (UMAP)."""
    naive = l0_mask & (model.astype(str) == "Naive")
    is_m = l0_mask & (model.astype(str) == "IS")
    if naive.sum() < 30 or is_m.sum() < 10:
        return None
    chord = np.nanmean(emb[is_m], axis=0) - np.nanmean(emb[naive], axis=0)
    if not np.isfinite(chord).all() or np.linalg.norm(chord) < 1e-8:
        return None
    return chord.astype(float)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def prepare_adata_for_oracle(adata, seed_tfs: list[str]):
    adata = adata.copy()
    adata.layers["raw_count"] = get_raw_counts(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=False)
    keep = adata.var["highly_variable"].copy()
    for g in seed_tfs:
        if g in adata.var_names:
            keep[g] = True
    adata = adata[:, keep].copy()
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    return adata


def oe_perturb_value(oracle, tf: str, percentile: float = 95.0) -> float:
    """Moderate OE level (high-end WT range, within CellOracle safe bounds)."""
    import scanpy as sc

    vals = sc.get.obs_df(oracle.adata, keys=[tf], layer="imputed_count")[tf].values.astype(float)
    vmin, vmax = float(vals.min()), float(vals.max())
    target = float(np.percentile(vals, percentile))
    upper = vmax + (vmax - vmin)  # safe_range_fold=2
    return min(target, upper)


def simulate_one(
    oracle,
    tf: str,
    perturb_value: float,
    grad: np.ndarray | None,
    cos_col: str,
    lineage: str,
    perturb_type: str,
    l0_mask: np.ndarray | None = None,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)
    n_cells = oracle.adata.n_obs
    oracle.simulate_shift(perturb_condition={tf: perturb_value}, n_propagation=3, ignore_warning=True)
    oracle.estimate_transition_prob(
        n_neighbors=min(200, n_cells - 1), knn_random=True, sampled_fraction=0.3,
    )
    oracle.calculate_embedding_shift(sigma_corr=0.05)
    delta = oracle.delta_embedding
    mean_shift = np.nanmean(delta, axis=0)
    if l0_mask is not None and l0_mask.sum() >= 30:
        mean_shift_l0 = np.nanmean(delta[l0_mask], axis=0)
    else:
        mean_shift_l0 = mean_shift
    cos = cosine_sim(mean_shift, grad) if grad is not None else np.nan
    cos_l0 = cosine_sim(mean_shift_l0, grad) if grad is not None else np.nan
    return {
        "TF": tf,
        "lineage": lineage,
        "embedding": "monocle3_umap",
        "perturb_type": perturb_type,
        "perturb_value": float(perturb_value),
        "shift_x": float(mean_shift_l0[0]),
        "shift_y": float(mean_shift_l0[1]),
        cos_col: cos_l0,
        "cosine_global": cos,
        "mean_shift": mean_shift_l0,
        "error": "",
    }


def run_lineage_ko(
    adata_full,
    lineage: str,
    cfg: dict,
    grn: pd.DataFrame,
    tfs: list[str],
) -> tuple[list[dict], dict]:
    """Run CellOracle on one lineage using Monocle3 UMAP (no scanpy UMAP)."""
    import celloracle as co

    sub = adata_full[adata_full.obs["subtype"] == lineage].copy()
    if sub.n_obs < 100:
        raise ValueError(f"Too few {lineage} cells: {sub.n_obs}")

    emb, pt = load_monocle3_embedding(cfg["mono_dir"], sub.obs_names)
    sub.obsm["X_umap_monocle3"] = emb

    sub = prepare_adata_for_oracle(sub, tfs)
    sub.obsm["X_umap_monocle3"] = emb  # restore after prepare (same cell order)

    sub.X = sub.layers["raw_count"].copy()
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(
        adata=sub,
        cluster_column_name="subcluster_r01" if "subcluster_r01" in sub.obs else "subtype",
        embedding_name="X_umap_monocle3",
    )

    var_set = set(sub.var_names)
    tfdict = build_tfdict(grn, var_set, cfg["grn_clusters"])
    tfdict = supplement_tfdict_from_coexpression(sub, tfdict, tfs)
    oracle.import_TF_data(TFdict=tfdict)

    oracle.perform_PCA()
    oracle.knn_imputation(k=max(10, int(sub.n_obs * 0.025)))
    oracle.fit_GRN_for_simulation(GRN_unit="cluster", alpha=1, verbose_level=0)

    subclusters = (
        sub.obs["subcluster_r01"].astype(str).values
        if "subcluster_r01" in sub.obs.columns else np.array(["Satglia"] * sub.n_obs)
    )
    l0_mask = _l0_mask(subclusters)
    # Pair CellOracle with the L0-internal Naive (bottom) → IS (top) axis, not L0→L1.
    model_arr = sub.obs["model"].astype(str).values if "model" in sub.obs.columns else np.array(["Naive"] * sub.n_obs)
    if lineage == "Satglia" and l0_mask.sum() >= 50:
        grad = l0_naive_to_is_axis(emb, l0_mask, model_arr)
        if grad is None:
            pt_l0 = pd.Series(np.asarray(pt)[l0_mask])
            grad = pt_gradient(emb[l0_mask], pt_l0)
    else:
        grad = pt_gradient(emb, pt)
    simulatable = set(getattr(oracle, "all_regulatory_genes_in_TFdict", []))
    rows = []
    shifts: dict[str, np.ndarray] = {}
    oe_rows: list[dict] = []

    for tf in tfs:
        base = {"TF": tf, "lineage": lineage, "embedding": "monocle3_umap"}
        if tf not in sub.var_names:
            rows.append({**base, "perturb_type": "ko", "error": "not in var_names"})
            continue
        if tf not in simulatable:
            rows.append({**base, "perturb_type": "ko", "error": "not in TFdict regulators"})
            continue
        try:
            row = simulate_one(
                oracle, tf, 0.0, grad, cfg["cos_col"], lineage, "ko",
                l0_mask=l0_mask if lineage == "Satglia" else None, seed=42 + hash(tf) % 1000,
            )
            shifts[f"{tf}_KO"] = row.pop("mean_shift")
            rows.append(row)
        except Exception as e:
            rows.append({
                **base, "perturb_type": "ko", "shift_x": np.nan, "shift_y": np.nan,
                cfg["cos_col"]: np.nan, "error": str(e),
            })

    # Prdm16 overexpression (direct forward readout; no KO reverse inference)
    if "Prdm16" in simulatable and "Prdm16" in sub.var_names:
        try:
            oe_val = oe_perturb_value(oracle, "Prdm16")
            row = simulate_one(
                oracle, "Prdm16", oe_val, grad, cfg["cos_col"], lineage, "oe",
                l0_mask=l0_mask, seed=1616,
            )
            shifts["Prdm16_OE"] = row.pop("mean_shift")
            oe_rows.append(row)
        except Exception as e:
            oe_rows.append({
                "TF": "Prdm16", "lineage": lineage, "embedding": "monocle3_umap",
                "perturb_type": "oe", "perturb_value": np.nan,
                "shift_x": np.nan, "shift_y": np.nan, cfg["cos_col"]: np.nan, "error": str(e),
            })

    subclusters = (
        sub.obs["subcluster_r01"].astype(str).values
        if "subcluster_r01" in sub.obs.columns else np.array(["Satglia"] * sub.n_obs)
    )
    meta = {"lineage": lineage, "n_cells": int(sub.n_obs), "embedding": str(cfg["mono_dir"] / "umap.csv"),
            "l0_n_cells": int(l0_mask.sum()), "cosine_scope": "Satglia_L0 mean shift" if lineage == "Satglia" else "global mean shift"}
    return rows, {"shifts": shifts, "emb": emb, "pt": pt, "subclusters": subclusters, "grad": grad, "meta": meta, "oe_rows": oe_rows}


def merge_results(sat_rows: list[dict], pep_rows: list[dict], tfs: list[str], sat_oe: list[dict] | None = None) -> pd.DataFrame:
    by_tf: dict[str, dict] = {tf: {"TF": tf, "embedding": "monocle3_umap"} for tf in tfs}
    for row in sat_rows + pep_rows:
        if row.get("perturb_type") != "ko":
            continue
        tf = row["TF"]
        if tf not in by_tf:
            by_tf[tf] = {"TF": tf}
        for k, v in row.items():
            if k in {"TF", "lineage", "perturb_type", "mean_shift"}:
                continue
            if k == "error" and v:
                by_tf[tf]["error"] = by_tf[tf].get("error", "") or v
            elif k not in by_tf[tf] or pd.isna(by_tf[tf].get(k)):
                by_tf[tf][k] = v
            elif k.startswith("cosine") and pd.notna(v):
                by_tf[tf][k] = v
    for row in sat_rows:
        tf = row["TF"]
        if row.get("perturb_type") == "ko" and pd.notna(row.get("shift_x")):
            by_tf[tf]["shift_x"] = row["shift_x"]
            by_tf[tf]["shift_y"] = row["shift_y"]
    if sat_oe:
        for row in sat_oe:
            if row.get("TF") != "Prdm16":
                continue
            by_tf["Prdm16"]["prdm16_oe_shift_x"] = row.get("shift_x")
            by_tf["Prdm16"]["prdm16_oe_shift_y"] = row.get("shift_y")
            by_tf["Prdm16"]["prdm16_oe_cosine_vs_satglia_internal_pt"] = row.get("cosine_vs_satglia_internal_pt")
            by_tf["Prdm16"]["prdm16_oe_perturb_value"] = row.get("perturb_value")
            if row.get("error"):
                by_tf["Prdm16"]["prdm16_oe_error"] = row["error"]
    df = pd.DataFrame(by_tf.values())
    for c in [
        "cosine_vs_satglia_internal_pt", "cosine_vs_pep_internal_pt",
        "shift_x", "shift_y",
        "prdm16_oe_shift_x", "prdm16_oe_shift_y", "prdm16_oe_cosine_vs_satglia_internal_pt",
    ]:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _l0_mask(subclusters: np.ndarray) -> np.ndarray:
    return np.array([str(s).startswith("Satglia_L0") or str(s) == "Satglia_L0" for s in subclusters])


def save_satglia_overlay(
    emb: np.ndarray,
    pt: pd.Series,
    subclusters: np.ndarray,
    shifts: dict[str, np.ndarray],
    grad: np.ndarray | None,
    highlight: list[tuple[str, str]],
    out_path: Path,
) -> None:
    """Perturbation arrows anchored at Satglia_L0 centroid (homeostatic reference)."""
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    l0 = _l0_mask(subclusters)
    non_l0 = ~l0

    if non_l0.any():
        ax.scatter(
            emb[non_l0, 0], emb[non_l0, 1], c="#2d333b", s=3, alpha=0.12, rasterized=True, linewidths=0, zorder=1,
        )
    l0_col = subcluster_colors().get("Satglia_L0", "#08306b")
    ax.scatter(
        emb[l0, 0], emb[l0, 1], c=l0_col, s=7, alpha=0.75, rasterized=True, linewidths=0, zorder=2,
        label="Satglia_L0 (homeostatic)",
    )

    span = np.percentile(np.ptp(emb, axis=0), 50)
    arrow_scale = 0.38 * span
    if l0.any():
        cx, cy = np.nanmean(emb[l0], axis=0)
    else:
        cx, cy = np.nanmean(emb, axis=0)

    if grad is not None and np.linalg.norm(grad) > 0:
        g = grad / np.linalg.norm(grad) * arrow_scale * 0.85
        ax.annotate(
            "", xy=(cx + g[0], cy + g[1]), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", lw=1.6, color="#8b949e", ls="dashed"),
        )
        ax.text(cx + g[0] * 1.12, cy + g[1] * 1.12, "PT↑", fontsize=9, color="#8b949e", ha="center")

    styles = {
        "Prdm16_KO": ("#58a6ff", "Prdm16 KO"),
        "Runx1_KO": ("#3fb950", "Runx1 KO"),
        "Prdm16_OE": ("#484f58", "Prdm16 OE"),
    }
    for key, _ptype in highlight:
        if key not in styles or key not in shifts:
            continue
        color, label = styles[key]
        v = shifts[key]
        nv = np.linalg.norm(v)
        if not np.isfinite(nv) or nv == 0:
            continue
        u = v / nv * arrow_scale
        lw = 3.0 if key == "Prdm16_KO" else 2.2
        ax.annotate(
            "", xy=(cx + u[0], cy + u[1]), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", lw=lw, color=color),
        )
        ax.text(
            cx + u[0] * 1.18 + (0.8 if key == "Runx1_KO" else -0.8),
            cy + u[1] * 1.18 + (0.6 if key == "Prdm16_KO" else -0.4),
            label,
            fontsize=10, fontweight="bold" if key == "Prdm16_KO" else "normal",
            color=color, ha="center",
        )

    ax.scatter([cx], [cy], s=55, c="#e6edf3", edgecolors=l0_col, linewidths=2, zorder=5)
    ax.set_title("CellOracle KO mean shift from Satglia_L0 (Monocle3 UMAP)")
    ax.set_xlabel("Monocle3 UMAP_1")
    ax.set_ylabel("Monocle3 UMAP_2")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(top_n: int = 6):
    OUT.mkdir(parents=True, exist_ok=True)
    if not FOCUS_H5AD.exists():
        raise FileNotFoundError(f"Missing {FOCUS_H5AD}")
    for ln, cfg in LINEAGE_CFG.items():
        if not (cfg["mono_dir"] / "umap.csv").exists():
            raise FileNotFoundError(f"Run Monocle3 UMAP export: missing {cfg['mono_dir']}/umap.csv")

    adata = sc.read_h5ad(FOCUS_H5AD)
    tfs = load_seed_tfs(top_n=top_n)
    grn = pd.read_csv(GRN)

    sat_rows, sat_extra = run_lineage_ko(adata, "Satglia", LINEAGE_CFG["Satglia"], grn, tfs)
    pep_rows, _ = run_lineage_ko(adata, "PEP", LINEAGE_CFG["PEP"], grn, tfs)

    summary = merge_results(sat_rows, pep_rows, tfs, sat_oe=sat_extra.get("oe_rows"))
    summary.to_csv(OUT / "ko_monocle3_alignment.csv", index=False)
    summary.sort_values("cosine_vs_satglia_internal_pt", ascending=False, na_position="last").to_csv(
        OUT / "ko_monocle3_ranked.csv", index=False
    )
    if sat_extra.get("oe_rows"):
        pd.DataFrame(sat_extra["oe_rows"]).to_csv(OUT / "prdm16_oe_satglia.csv", index=False)

    overlay_path = OUT / "Fig_celloracle_ko_on_monocle3_satglia_umap.png"
    save_satglia_overlay(
        sat_extra["emb"], sat_extra["pt"], sat_extra["subclusters"], sat_extra["shifts"],
        sat_extra.get("grad"),
        highlight=[("Prdm16_KO", "ko"), ("Runx1_KO", "ko")],
        out_path=overlay_path,
    )
    import shutil
    web_fig = ROOT / "results/reports/web/fig/celloracle_ko_on_monocle3_satglia.png"
    web_fig.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(overlay_path, web_fig)

    (OUT / "embedding_meta.json").write_text(
        json.dumps({"Satglia": sat_extra["meta"], "note": "cosine computed in Monocle3 lineage UMAP"}, indent=2),
        encoding="utf-8",
    )
    print(summary)
    print(f"Wrote CellOracle KO (Monocle3 UMAP) → {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=6)
    args = p.parse_args()
    main(top_n=args.top_n)
