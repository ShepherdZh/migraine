import sys
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from pycisTopic.topic_binarization import binarize_topics
from pycistarget.motif_enrichment_cistarget import cisTarget, cisTargetDatabase
from scenicplus.data_wrangling.gene_search_space import (
    download_gene_annotation_and_chromsizes,
    get_search_space,
)
from scenicplus.enhancer_to_gene import calculate_regions_to_genes_relationships

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
OUT = ROOT / "results/scenicplus/mouse/run/egrn_pycistopic"
CISTOPIC = ROOT / "results/scenicplus/mouse/run/cistopic_obj.pkl"
RNA = ROOT / "results/scenicplus/mouse/run/rna.h5ad"
CTX_DB = Path("/opt/NAS/RD/pinned/scenic/feather_GRCM38/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather")
MOTIF_TBL = Path("/opt/NAS/RD/pinned/scenic/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl")
FOCUS_GENES = ["Calca", "Calcb", "Ramp1", "Prdm16", "Ptn", "Tac1", "Runx1", "Stat6", "Mef2d"]


def binarized_topics_to_pyranges(binarized_topics: dict) -> dict:
    import pyranges as pr

    region_sets = {}
    for topic, df in binarized_topics.items():
        rows = []
        for pk in df.index.astype(str):
            if pk.startswith("chr") and pk.count("-") >= 2:
                chrom, rest = pk.split("-", 1)
                start, end = rest.split("-", 1)
                rows.append([chrom, int(start), int(end), pk])
        if rows:
            region_sets[topic] = pr.PyRanges(
                pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "Name"])
            )
    return region_sets


def topics_from_cistopic_model(cistopic_obj) -> dict:
    import pyranges as pr

    model = cistopic_obj.selected_model
    topics = np.asarray(cistopic_obj.region_names, dtype=str)
    tr = model.topic_region
    region_sets = {}
    for topic in tr.columns:
        mask = tr[topic].values > 0
        peaks = topics[mask]
        rows = []
        for pk in peaks[:50000]:
            if pk.startswith("chr") and pk.count("-") >= 2:
                chrom, rest = pk.split("-", 1)
                start, end = rest.split("-", 1)
                rows.append([chrom, int(start), int(end), pk])
        if rows:
            df = pd.DataFrame(rows, columns=["Chromosome", "Start", "End", "Name"])
            region_sets[str(topic)] = pr.PyRanges(df)
    return region_sets


def build_region_sets(cistopic_obj, log) -> dict:
    topics_dir = OUT / "topics_binarized"
    topics_dir.mkdir(exist_ok=True)
    try:
        binarized = binarize_topics(
            cistopic_obj,
            method="otsu",
            plot=False,
            save=str(topics_dir / "binarization_plots.pdf"),
        )
        region_sets = binarized_topics_to_pyranges(binarized)
        if region_sets:
            log.info("binarize_topics → %d topic region sets", len(region_sets))
            return region_sets
    except Exception as e:
        log.warning("binarize_topics failed (%s); using topic_region fallback", e)

    region_sets = topics_from_cistopic_model(cistopic_obj)
    if not region_sets:
        raise RuntimeError("Could not build topic region sets from cisTopic object")
    log.info("topic_region fallback → %d topic region sets", len(region_sets))
    return region_sets


def run_pycistarget_topics(
    region_sets: dict,
    save_path: Path,
    log,
    n_cpu: int = 8,
    max_topics: int | None = None,
) -> int:
    save_path.mkdir(parents=True, exist_ok=True)
    keys = sorted(region_sets.keys())
    if max_topics is not None:
        keys = keys[:max_topics]
    done = 0
    for key in keys:
        regions = region_sets[key]
        out_folder = save_path / f"CTX_{key}_All"
        h5 = out_folder / f"motif_enrichment_cistarget_{key}.tsv"
        if h5.exists():
            log.info("Skip existing cisTarget result for %s", key)
            done += 1
            continue
        log.info("Loading cisTarget database for %s", key)
        ctx_db = cisTargetDatabase(fname=str(CTX_DB), region_sets=regions, name=key)
        log.info("Running cisTarget for %s", key)
        ct = cisTarget(
            region_set=regions,
            name=key,
            species="mus_musculus",
            auc_threshold=0.005,
            nes_threshold=3.0,
            rank_threshold=0.05,
            path_to_motif_annotations=str(MOTIF_TBL),
            annotation_version="v10nr_clust",
            annotation_to_use=["Direct_annot", "Orthology_annot"],
            motif_similarity_fdr=1e-6,
        )
        ct.run_ctx(ctx_db)
        out_folder.mkdir(parents=True, exist_ok=True)
        ct.motif_enrichment.to_csv(out_folder / f"motif_enrichment_cistarget_{key}.tsv", sep="\t")
        done += 1
    return done


def to_scenicplus_region_name(pk: str) -> str:
    pk = str(pk)
    if ":" in pk:
        return pk
    if pk.startswith("chr") and pk.count("-") >= 2:
        chrom, rest = pk.split("-", 1)
        start, end = rest.split("-", 1)
        return f"{chrom}:{start}-{end}"
    return pk


def chromsizes_from_regions(region_names) -> pd.DataFrame:
    max_end: dict[str, int] = {}
    for pk in map(str, region_names):
        scenic = to_scenicplus_region_name(pk)
        if scenic.count(":") == 1 and scenic.count("-") >= 1:
            chrom, rest = scenic.split(":", 1)
            _, end = rest.split("-", 1)
            max_end[chrom] = max(max_end.get(chrom, 0), int(end))
    return pd.DataFrame(
        {"Chromosome": list(max_end.keys()), "Start": 0, "End": list(max_end.values())}
    )


def normalize_chromosomes(df: pd.DataFrame, col: str = "Chromosome") -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str)
    df.loc[~df[col].str.startswith("chr"), col] = "chr" + df[col]
    return df


def load_gene_annotation_and_chromsizes(cistopic_obj, log):
    annot_path = OUT / "gene_annotation.csv"
    chrom_path = OUT / "chromsizes.csv"
    if annot_path.exists() and chrom_path.exists():
        return pd.read_csv(annot_path), pd.read_csv(chrom_path)

    log.info("Downloading mm10/GRCm39 gene annotation via biomart ...")
    result = download_gene_annotation_and_chromsizes(
        species="mmusculus",
        biomart_host="http://www.ensembl.org",
    )
    if isinstance(result, tuple):
        gene_annot, chromsizes = result
    else:
        gene_annot = result
        chromsizes = chromsizes_from_regions(cistopic_obj.region_names)
        log.warning("NCBI chromsizes unavailable; inferred from peak coordinates")

    gene_annot = normalize_chromosomes(gene_annot)
    chromsizes = normalize_chromosomes(chromsizes)
    gene_annot.to_csv(annot_path, index=False)
    chromsizes.to_csv(chrom_path, index=False)
    return gene_annot, chromsizes


def build_pseudobulk_mats(cistopic_obj, rna, log):
    """Aggregate RNA/ATAC to subtype pseudobulk (RNA+ATAC barcodes are not paired)."""
    cd = cistopic_obj.cell_data.copy()
    cd.index = cd.index.astype(str)
    subtype_col = "plot_subtype" if "plot_subtype" in cd.columns else "subtype"
    frag = cistopic_obj.fragment_matrix
    region_cols = [to_scenicplus_region_name(x) for x in cistopic_obj.region_names]
    acc = pd.DataFrame(
        frag.T.toarray() if hasattr(frag.T, "toarray") else np.asarray(frag.T),
        index=cd.index.astype(str),
        columns=region_cols,
    )
    acc["__subtype__"] = cd[subtype_col].astype(str).values
    acc_pb = acc.groupby("__subtype__", observed=True).mean(numeric_only=True)

    rna_sub = rna.copy()
    rna_sub.obs["__subtype__"] = rna_sub.obs["subtype"].astype(str).values
    if sparse.issparse(rna_sub.X):
        exp = pd.DataFrame(
            rna_sub.X.toarray(),
            index=rna_sub.obs_names.astype(str),
            columns=rna_sub.var_names.astype(str),
        )
    else:
        exp = pd.DataFrame(
            np.asarray(rna_sub.X),
            index=rna_sub.obs_names.astype(str),
            columns=rna_sub.var_names.astype(str),
        )
    exp["__subtype__"] = rna_sub.obs["__subtype__"].values
    exp_pb = exp.groupby("__subtype__", observed=True).mean(numeric_only=True)

    common_sub = sorted(set(acc_pb.index) & set(exp_pb.index))
    log.info(
        "Pseudobulk R2G: %d shared subtypes (%s)",
        len(common_sub),
        ", ".join(common_sub[:8]) + ("..." if len(common_sub) > 8 else ""),
    )
    return exp_pb.loc[common_sub], acc_pb.loc[common_sub]


def run_region_to_gene(cistopic_obj, rna, log) -> pd.DataFrame:
    gene_annot, chromsizes = load_gene_annotation_and_chromsizes(cistopic_obj, log)
    scplus_regions = {to_scenicplus_region_name(x) for x in cistopic_obj.region_names}
    genes = set(FOCUS_GENES) & set(rna.var_names.astype(str))
    search_space = get_search_space(
        scplus_region=scplus_regions,
        scplus_genes=genes,
        gene_annotation=gene_annot,
        chromsizes=chromsizes,
        upstream=(1000, 150000),
        downstream=(1000, 150000),
        extend_tss=(10, 10),
    )
    search_space.to_csv(OUT / "search_space.csv", index=False)

    exp_df, acc_df = build_pseudobulk_mats(cistopic_obj, rna, log)
    genes = set(FOCUS_GENES) & set(exp_df.columns)
    ss = search_space[search_space["Gene"].isin(genes)]
    if ss.empty:
        ss = search_space[search_space["Gene"].isin(FOCUS_GENES)]

    log.info("Running SCENIC+ region-to-gene on %d gene search-space rows ...", len(ss))
    rows = []
    for gene in sorted(ss["Gene"].unique()):
        ss_g = ss[ss["Gene"] == gene]
        if gene not in exp_df.columns:
            log.warning("Skip R2G for %s: not in expression matrix", gene)
            continue
        try:
            r2g = calculate_regions_to_genes_relationships(
                df_exp_mtx=exp_df,
                df_acc_mtx=acc_df,
                search_space=ss_g,
                temp_dir=OUT / "tmp_r2g",
                importance_scoring_method="GBM",
                correlation_scoring_method="SR",
                n_cpu=4,
            )
            if isinstance(r2g, pd.DataFrame):
                df = r2g.copy()
            else:
                df = pd.DataFrame(r2g)
            if len(df):
                df["gene"] = gene
                rows.append(df)
        except Exception as e:
            log.warning("R2G skipped %s: %s", gene, e)
    r2g_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    r2g_df.to_csv(OUT / "region_to_gene_pycistopic.csv", index=False)

    calca = r2g_df[r2g_df["gene"].isin(["Calca", "Calcb"])] if len(r2g_df) else pd.DataFrame()
    calca.to_csv(OUT / "region_to_gene_calca_focus.csv", index=False)
    log.info("R2G complete: %d total links, %d Calca/Calcb links", len(r2g_df), len(calca))
    return r2g_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pycistarget", action="store_true")
    parser.add_argument("--skip-r2g", action="store_true")
    parser.add_argument("--max-topics", type=int, default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("pycistopic_scenicplus")

    log.info("Loading cisTopic object ...")
    with open(CISTOPIC, "rb") as f:
        cistopic_obj = dill.load(f)
    log.info("Loading RNA ...")
    rna = sc.read_h5ad(RNA)

    region_sets = build_region_sets(cistopic_obj, log)
    n_pycistarget = 0

    if not args.skip_r2g:
        run_region_to_gene(cistopic_obj, rna, log)

    if not args.skip_pycistarget:
        log.info("Running pycistarget on up to %d topic region sets ...",
                 args.max_topics or len(region_sets))
        n_pycistarget = run_pycistarget_topics(
            region_sets,
            OUT / "pycistarget",
            log,
            n_cpu=8,
            max_topics=args.max_topics,
        )

    r2g_path = OUT / "region_to_gene_pycistopic.csv"
    n_r2g = 0
    if r2g_path.exists() and r2g_path.stat().st_size > 1:
        n_r2g = len(pd.read_csv(r2g_path))
    calca_path = OUT / "region_to_gene_calca_focus.csv"
    n_calca = 0
    if calca_path.exists() and calca_path.stat().st_size > 1:
        n_calca = len(pd.read_csv(calca_path))

    status = {
        "cistopic": str(CISTOPIC),
        "rna_h5ad": str(RNA),
        "n_topics": len(region_sets),
        "n_pycistarget_topics_done": n_pycistarget,
        "n_r2g": n_r2g,
        "n_calca_links": n_calca,
        "motif_db": str(CTX_DB),
        "note": "GRCm39 peaks; R2G uses subtype pseudobulk (RNA/ATAC barcodes not paired).",
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
    if n_r2g > 0:
        (OUT / "SCENICPLUS_PYCISTOPIC_DONE.txt").write_text("ok\n")
    log.info("Done → %s", OUT)


if __name__ == "__main__":
    main()
