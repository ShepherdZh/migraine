#!/usr/bin/env python3
"""
Full SCENIC+ eRegulon + AUCell pipeline.

Prerequisites: cistopic_obj.pkl, rna.h5ad, pinned mm10 cisTarget feather DB.
Outputs: results/scenicplus/mouse/run/egrn_full/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import dill
import mudata as md
import pandas as pd
from pycisTopic.topic_binarization import binarize_topics

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
OUT = ROOT / "results/scenicplus/mouse/run/egrn_full"
CISTOPIC = ROOT / "results/scenicplus/mouse/run/cistopic_obj.pkl"
RNA = ROOT / "results/scenicplus/mouse/run/rna.h5ad"
PYCIS = ROOT / "results/scenicplus/mouse/run/egrn_pycistopic"
CTX_DB = Path("/opt/NAS/RD/pinned/scenic/feather_GRCM38/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather")
MOTIF_TBL = Path("/opt/NAS/RD/pinned/scenic/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl")
STATUS = OUT / "pipeline_status.json"
N_CPU = int(os.environ.get("SCENICPLUS_N_CPU", "8"))


def setup_log() -> logging.Logger:
    OUT.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("scenicplus_full")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
        fh = logging.FileHandler(OUT / "pipeline.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    return log


def load_status() -> dict:
    if STATUS.exists():
        return json.loads(STATUS.read_text())
    return {"steps_done": []}


def mark(step: str, log: logging.Logger, extra: dict | None = None) -> None:
    st = load_status()
    if step not in st["steps_done"]:
        st["steps_done"].append(step)
    if extra:
        st.update(extra)
    STATUS.write_text(json.dumps(st, indent=2))
    log.info("DONE step: %s", step)


def done(step: str) -> bool:
    return step in load_status().get("steps_done", [])


def run_cmd(cmd: list[str], log: logging.Logger) -> None:
    log.info("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def peak_dash_to_colon(name: str) -> str:
    if ":" in name:
        return name
    chrom, rest = name.split("-", 1)
    return f"{chrom}:{rest}"


def peak_colon_to_bed(name: str) -> tuple[str, int, int, str] | None:
    if ":" not in name:
        return None
    chrom, rest = name.split(":", 1)
    start, end = rest.split("-", 1)
    return chrom, int(start), int(end), name


def fix_mudata_peak_names(h5mu: Path, log: logging.Logger) -> None:
    m = md.read_h5mu(h5mu)
    old = m.mod["scATAC"].var_names.tolist()
    new = [peak_dash_to_colon(x) for x in old]
    if old == new:
        log.info("MuData peak names already colon-format")
        return
    m.mod["scATAC"].var_names = new
    # sync coordinate columns if present
    if "Chromosome" in m.mod["scATAC"].var.columns:
        for i, nm in enumerate(new):
            parsed = peak_colon_to_bed(nm)
            if parsed:
                chrom, start, end, _ = parsed
                m.mod["scATAC"].var.at[nm, "Chromosome"] = chrom
                m.mod["scATAC"].var.at[nm, "Start"] = start
                m.mod["scATAC"].var.at[nm, "End"] = end
    m.write(h5mu)
    log.info("Converted %d ATAC peak names to chr:start-end format", len(new))


def prepare_genome_tables(log: logging.Logger) -> None:
    ga_out = OUT / "genome_annotation.tsv"
    cs_out = OUT / "chromsizes.tsv"
    if not ga_out.exists():
        cols = ["Chromosome", "Start", "End", "Strand", "Gene", "Transcription_Start_Site"]
        pd.read_csv(PYCIS / "gene_annotation.csv")[cols].to_csv(ga_out, sep="\t", index=False)
    if not cs_out.exists():
        pd.read_csv(PYCIS / "chromsizes.csv").to_csv(cs_out, sep="\t", index=False)
    log.info("Genome annotation + chromsizes ready")


def build_region_set_folder(log: logging.Logger) -> Path:
    region_root = OUT / "region_sets" / "topics"
    if region_root.exists() and any(region_root.glob("*.bed")):
        log.info("Region set beds exist → %s", region_root)
        return OUT / "region_sets"

    region_root.mkdir(parents=True, exist_ok=True)
    ct = dill.load(open(CISTOPIC, "rb"))
    topics_dir = OUT / "topics_binarized"
    topics_dir.mkdir(exist_ok=True)
    region_sets: dict[str, list[str]] = {}
    try:
        binarized = binarize_topics(ct, method="otsu", plot=False, save=str(topics_dir / "binarization.pdf"))
        for topic, df in binarized.items():
            region_sets[str(topic)] = [peak_dash_to_colon(str(x)) for x in df.index.astype(str)]
    except Exception as e:
        log.warning("binarize_topics failed (%s); fallback topic_region", e)
        model = ct.selected_model
        tr = model.topic_region
        topics = pd.Index(ct.region_names).astype(str)
        for topic in tr.columns:
            peaks = topics[tr[topic].values > 0]
            region_sets[str(topic)] = [peak_dash_to_colon(p) for p in peaks[:50000]]

    n_written = 0
    for topic, peaks in sorted(region_sets.items()):
        rows = []
        for pk in peaks:
            bed = peak_colon_to_bed(pk)
            if bed:
                chrom, start, end, name = bed
                rows.append((chrom, start, end, name))
        if not rows:
            continue
        bed_path = region_root / f"{topic}.bed"
        pd.DataFrame(rows, columns=["chrom", "start", "end", "name"]).to_csv(
            bed_path, sep="\t", header=False, index=False
        )
        n_written += 1
    log.info("Wrote %d topic bed files", n_written)
    return OUT / "region_sets"


def ctx_db_path(log: logging.Logger) -> Path:
    shm = Path("/dev/shm/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather")
    if shm.exists() and shm.stat().st_size > 10_000_000_000:
        log.info("Using cached feather in /dev/shm")
        return shm
    if not CTX_DB.exists():
        raise FileNotFoundError(f"Missing cisTarget DB: {CTX_DB}")
    log.info("Copying 17GB feather to /dev/shm (one-time)...")
    shutil.copy2(CTX_DB, shm)
    return shm


def main(force: bool = False) -> None:
    log = setup_log()
    tmp = OUT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    h5mu = OUT / "ACC_GEX.h5mu"

    # 1 MuData link
    if force or not done("prepare_GEX_ACC"):
        if not h5mu.exists():
            run_cmd([
                "scenicplus", "prepare_data", "prepare_GEX_ACC",
                "--cisTopic_obj_fname", str(CISTOPIC),
                "--GEX_anndata_fname", str(RNA),
                "--out_file", str(h5mu),
                "--is_not_multiome",
                "--key_to_group_by", "subtype",
                "--nr_cells_per_metacells", "10",
                "--do_not_use_raw_for_GEX_anndata",
            ], log)
        mark("prepare_GEX_ACC", log)

    # 1b fix peak naming
    if force or not done("fix_peak_names"):
        fix_mudata_peak_names(h5mu, log)
        mark("fix_peak_names", log)

    # 2 genome tables
    if force or not done("genome_tables"):
        prepare_genome_tables(log)
        mark("genome_tables", log)

    # 3 search space
    search_space = OUT / "search_space.tsv"
    if force or not done("search_space"):
        run_cmd([
            "scenicplus", "prepare_data", "search_spance",
            "--multiome_mudata_fname", str(h5mu),
            "--gene_annotation_fname", str(OUT / "genome_annotation.tsv"),
            "--chromsizes_fname", str(OUT / "chromsizes.tsv"),
            "--out_fname", str(search_space),
        ], log)
        mark("search_space", log)

    # 4 region sets + motif enrichment HDF5
    ctx_h5 = OUT / "ctx_results.hdf5"
    if force or not done("motif_enrichment"):
        region_sets = build_region_set_folder(log)
        db = ctx_db_path(log)
        if ctx_h5.exists():
            ctx_h5.unlink()
        run_cmd([
            "scenicplus", "grn_inference", "motif_enrichment_cistarget",
            "--region_set_folder", str(region_sets),
            "--cistarget_db_fname", str(db),
            "--output_fname_cistarget_result", str(ctx_h5),
            "--temp_dir", str(tmp),
            "--species", "mus_musculus",
            "--path_to_motif_annotations", str(MOTIF_TBL),
            "--write_html",
            "--output_fname_cistarget_html", str(OUT / "ctx_results.html"),
            "--n_cpu", str(N_CPU),
        ], log)
        mark("motif_enrichment", log)

    # 5 prepare_menr
    tf_names = OUT / "tf_names.txt"
    cistromes_d = OUT / "cistromes_direct.h5ad"
    cistromes_e = OUT / "cistromes_extended.h5ad"
    if force or not done("prepare_menr"):
        run_cmd([
            "scenicplus", "prepare_data", "prepare_menr",
            "--paths_to_motif_enrichment_results", str(ctx_h5),
            "--multiome_mudata_fname", str(h5mu),
            "--out_file_tf_names", str(tf_names),
            "--out_file_direct_annotation", str(cistromes_d),
            "--out_file_extended_annotation", str(cistromes_e),
        ], log)
        mark("prepare_menr", log)

    # 6 TF_to_gene
    tf2g = OUT / "tf_to_gene_adj.tsv"
    if force or not done("TF_to_gene"):
        run_cmd([
            "scenicplus", "grn_inference", "TF_to_gene",
            "--multiome_mudata_fname", str(h5mu),
            "--tf_names", str(tf_names),
            "--temp_dir", str(tmp),
            "--out_tf_to_gene_adjacencies", str(tf2g),
            "--n_cpu", str(N_CPU),
        ], log)
        mark("TF_to_gene", log)

    # 7 region_to_gene
    r2g = OUT / "region_to_gene_adj.tsv"
    if force or not done("region_to_gene"):
        run_cmd([
            "scenicplus", "grn_inference", "region_to_gene",
            "--multiome_mudata_fname", str(h5mu),
            "--search_space_fname", str(search_space),
            "--temp_dir", str(tmp),
            "--out_region_to_gene_adjacencies", str(r2g),
            "--n_cpu", str(N_CPU),
        ], log)
        mark("region_to_gene", log)

    db = ctx_db_path(log)

    # 8 eRegulon direct
    ere_d = OUT / "eRegulon_direct.tsv"
    if force or not done("eGRN_direct"):
        run_cmd([
            "scenicplus", "grn_inference", "eGRN",
            "--TF_to_gene_adj_fname", str(tf2g),
            "--region_to_gene_adj_fname", str(r2g),
            "--cistromes_fname", str(cistromes_d),
            "--ranking_db_fname", str(db),
            "--eRegulon_out_fname", str(ere_d),
            "--temp_dir", str(tmp),
            "--n_cpu", str(N_CPU),
        ], log)
        mark("eGRN_direct", log)

    # 9 eRegulon extended
    ere_e = OUT / "eRegulon_extended.tsv"
    if force or not done("eGRN_extended"):
        run_cmd([
            "scenicplus", "grn_inference", "eGRN",
            "--TF_to_gene_adj_fname", str(tf2g),
            "--region_to_gene_adj_fname", str(r2g),
            "--cistromes_fname", str(cistromes_e),
            "--ranking_db_fname", str(db),
            "--eRegulon_out_fname", str(ere_e),
            "--temp_dir", str(tmp),
            "--is_extended",
            "--n_cpu", str(N_CPU),
        ], log)
        mark("eGRN_extended", log)

    # 10 AUCell
    auc_d = OUT / "AUCell_direct.h5mu"
    auc_e = OUT / "AUCell_extended.h5mu"
    if force or not done("AUCell_direct"):
        run_cmd([
            "scenicplus", "grn_inference", "AUCell",
            "--eRegulon_fname", str(ere_d),
            "--multiome_mudata_fname", str(h5mu),
            "--aucell_out_fname", str(auc_d),
            "--n_cpu", str(N_CPU),
        ], log)
        mark("AUCell_direct", log)
    if force or not done("AUCell_extended"):
        run_cmd([
            "scenicplus", "grn_inference", "AUCell",
            "--eRegulon_fname", str(ere_e),
            "--multiome_mudata_fname", str(h5mu),
            "--aucell_out_fname", str(auc_e),
            "--n_cpu", str(N_CPU),
        ], log)
        mark("AUCell_extended", log)

    # 11 final object
    scplus = OUT / "scplusmdata.h5mu"
    if force or not done("create_scplus_mudata"):
        run_cmd([
            "scenicplus", "grn_inference", "create_scplus_mudata",
            "--multiome_mudata_fname", str(h5mu),
            "--e_regulon_auc_direct_mudata_fname", str(auc_d),
            "--e_regulon_auc_extended_mudata_fname", str(auc_e),
            "--e_regulon_metadata_direct_fname", str(ere_d),
            "--e_regulon_metadata_extended_fname", str(ere_e),
            "--out_file", str(scplus),
        ], log)
        mark("create_scplus_mudata", log, {"status": "complete"})

    log.info("SCENIC+ eRegulon + AUCell pipeline complete → %s", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Re-run all steps")
    main(force=ap.parse_args().force)
