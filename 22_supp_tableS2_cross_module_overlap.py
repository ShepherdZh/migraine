#!/usr/bin/env python3
"""Cross-module gene overlap + pathway migraine biology for web report."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
sys.path.insert(0, str(ROOT / "scripts"))
WEB_DATA = ROOT / "results/reports/web/data"

# --- pathway → migraine biology (curated) ---
PATHWAY_MIGRAINE = {
    "TNF-alpha Signaling via NF-kB": "Neuroinflammation hub in migraine; TNF-α levels rise during attacks and anti-TNF strategies modulate trigeminal activation. Satglia IS activation fits Yang et al. glial IEG story.",
    "TNF signaling pathway": "Same axis at KEGG level — links IS/CSD model to pro-inflammatory glial/fibroblast states that sensitize trigeminal circuits.",
    "Inflammatory Response": "Hallmark of migraine chronification; non-neuronal compartments (Satglia, fibroblast) carry IS-induced inflammatory programs per scDRS/GWAS enrichment.",
    "Interferon Gamma Response": "Immune-inflammatory crosstalk; may reflect meningeal/immune adjacency in TG single-cell despite limited Immune OR in IEG barplot.",
    "IL-17 signaling pathway": "Autoimmune/inflammatory axis implicated in chronic migraine comorbidity; fibroblast ECM-immune niche.",
    "Cholesterol Homeostasis": "Lipid metabolism GWAS hits (e.g. near HMGCR/LDLR); Satglia IS-up cholesterol genes connect metabolic stress to glial activation.",
    "Cholesterol Biosynthetic Process": "Biosynthesis arm of lipid program; statins/metabolic comorbidity literature overlaps with migraine risk loci.",
    "Sterol Biosynthetic Process": "Broader sterol pathway; supports metabolic reprogramming under IS rather than pure neuronal CGRP change.",
    "mTORC1 Signaling": "Growth/metabolic sensor; ties headache model activation to anabolic/stress programs in Satglia.",
    "Hypoxia": "Vascular/metabolic stress readout; relevant to CSD/spreading depression physiology in migraine models.",
    "Epithelial Mesenchymal Transition": "MSigDB Hallmark EMT gene set — in TG snRNA-seq this usually reflects **mesenchymal-like ECM/fibroblast activation** (Col1a1, Mmp, integrin modules), not literal epithelial→mesenchymal transition (TG 中无上皮细胞群). Interpret as stromal remodeling, not classical EMT.",
    "MAPK signaling pathway": "Stress/growth-factor kinase cascade; can be driven by inflammatory cytokines and **Ptn–receptor tyrosine kinase** modules in stromal cells. In Fibroblast_Dcn it supports matrix/inflammatory coupling to headache models — Ptn here is largely **Satglia-secreted**, fibroblast enrichment reflects downstream/responder state rather than Ptn source.",
    "Focal adhesion": "Cell–matrix coupling; Lama2/Cd44/Itgb pathways bridge fibroblast activation to Satglia–PEP spatial organization.",
    "MAPK signaling pathway": "Stress kinase cascade downstream of growth factors (Ptn) and inflammatory cues; common pain sensitization mechanism.",
    "Apoptosis": "Stress-response in activated glia/fibroblast; not necessarily cell death but pro-apoptotic signaling modules under IS.",
    "Androgen Response": "Sex-hormone-linked transcription; migraine prevalence sex bias — Satglia_L0 IS module.",
    "Nervous System Development": "Satglia identity/development genes (Ptn, Fabp7, Sox10) down in IS — glial dedifferentiation or state shift away from homeostatic Satglia.",
    "Dendrite Morphogenesis": "Structural plasticity genes (Ptn, Ctnnd2); Satglia support of neurite/PEP microenvironment.",
    "Hedgehog Signaling": "Developmental morphogen; glial–neuron patterning, indirect link to trigeminal organization.",
    "Adherens junction": "Cadherin junction biology — overlaps CellChat CADM axis conceptually.",
    "Cytoplasmic Translation": "PEP_L0 IS-up — translational control in peptidergic neurons; can affect neuropeptide (CGRP) production capacity.",
    "Mitotic Spindle": "Cell-cycle/cytoskeletal module in Satglia IS-down — homeostatic Satglia exit from proliferative/stress state.",
    "UV Response Dn": "Stress-down module; inverse of inflammatory activation signature.",
    "Leukocyte transendothelial migration": "Vascular–immune interface; fibroblast/perivascular niche in TG.",
    "Apical Junction": "Polarity/junction remodeling with focal adhesion program in fibroblasts.",
    "Salmonella infection": "KEGG immune pathway artifact — often maps to generic NF-κB/cytokine genes (same drivers as TNF axis).",
    "I-kappaB kinase/NF-kappaB Signaling": "Core NF-κB activation; merges with TNF-inflammatory migraine narrative.",
    "Positive Regulation Of Transcription By RNA Polymerase II": "Broad TF activation; includes Prdm16/Runx1-class regulators in Satglia GRN.",
    "Regulation Of Transcription By RNA Polymerase II": "Same — GWAS/scDRS TFs (Runx1, Tcf4, Mef2d) often in this module.",
    "Positive Regulation Of DNA-templated Transcription": "Satglia IS-down — loss of developmental TF program under activation.",
    "Regulation Of Metal Ion Transport": "Ion homeostasis (ANK3, PER1); links to excitability and migraine chronification loci.",
    "Regulation Of Sodium Ion Transport": "Nav channel regulation context — adjacent to PEP Scn10a/Scn9a biology.",
    "Secondary Alcohol Biosynthetic Process": "Cholesterol/sterol branch in Satglia IS-up.",
    "Cholesterol Import": "LDLR/STARD4 — lipid uptake under inflammatory glial activation.",
}


def parse_gs_genes(path: Path, top_n: int = 200) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text()
    m = re.search(r"MIGRAINE_IHGC\t(.+)", text) or re.search(r"MIGRAINE[^\t]*\t(.+)", text)
    if not m:
        return set()
    pairs = m.group(1).split(",")
    genes = []
    for p in pairs:
        if ":" in p:
            g, w = p.split(":", 1)
            try:
                genes.append((g.strip(), float(w)))
            except ValueError:
                genes.append((g.strip(), 1.0))
    genes.sort(key=lambda x: -x[1])
    return {g for g, _ in genes[:top_n]}


def scdrs_top(path: Path, top_n: int = 100) -> set[str]:
    df = pd.read_csv(path, sep="\t")
    df = df.sort_values("CORR", ascending=False)
    return set(df.iloc[:top_n, 0].astype(str))


def gwas_tf_loci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["gene", "best_peak", "best_subtype", "score"]
    return df[[c for c in cols if c in df.columns]].head(20)


def enrich_deg_genes() -> set[str]:
    csv = ROOT / "results/enrichment/subcluster_r01/enrichr_all_subclusters.csv"
    if not csv.exists():
        return set()
    df = pd.read_csv(csv)
    genes = set()
    for s in df[df["direction"] == "IS_up"]["Genes"].dropna():
        genes.update(x.strip() for x in str(s).split(";") if x.strip())
    return genes


def enrich_input_degs(h5ad: Path) -> set[str]:
    adata = sc.read_h5ad(h5ad)
    out = set()
    for cl in adata.obs["subcluster_r01"].astype(str).unique():
        sub = adata[(adata.obs["subcluster_r01"] == cl) & (adata.obs["model"].isin(["Naive", "IS"]))].copy()
        if sub.n_obs < 80 or sub.obs["model"].nunique() < 2:
            continue
        sc.tl.rank_genes_groups(sub, groupby="model", groups=["IS"], reference="Naive", method="wilcoxon")
        df = sc.get.rank_genes_groups_df(sub, group="IS")
        sig = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"].abs() >= 0.25)]
        out.update(sig["names"].astype(str))
    return out


def monocle_heatmap_genes() -> set[str]:
    genes = set()
    for lineage in ["satglia", "pep", "fibroblast_dcn"]:
        csv = WEB_DATA / f"monocle3_{lineage}_pseudotime_spearman.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        # proxy: top |rho| + migraine list from 36
        top = df.reindex(df["rho"].abs().sort_values(ascending=False).index).head(30)["gene"]
        genes.update(top.astype(str))
    mig = ["Calca", "Calcb", "Ptn", "Ramp1", "Prdm16", "Runx1", "Fabp7", "Dcn", "Cadm1", "Lama2"]
    genes.update(mig)
    return genes


def pycistarget_tfs() -> set[str]:
    tfs = set()
    pycis = ROOT / "results/scenicplus/mouse/run/egrn_pycistopic/pycistarget"
    for topic in [9, 1]:
        tsv = pycis / f"CTX_Topic{topic}_All/motif_enrichment_cistarget_Topic{topic}.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t").sort_values("NES", ascending=False).head(30)
        for _, r in df.iterrows():
            for col in ("Direct_annot", "Orthology_annot"):
                v = str(r.get(col, ""))
                if v and v != "nan":
                    tfs.add(v.split(",")[0].strip())
    return {t for t in tfs if t}


def cellchat_genes() -> set[str]:
    path = ROOT / "results/cellchat/satglia_pep_subcluster/lr_pairs_merged.csv"
    if not path.exists():
        path = ROOT / "results/cellchat/satglia_pep_subcluster/lr_pairs_all.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    sat_pep = df[df["source"].astype(str).str.startswith("Satglia_") & df["target"].astype(str).str.startswith("PEP_")]
    sat_pep = sat_pep[~sat_pep["pathway_name"].astype(str).str.contains("BMP", case=False)]
    top = sat_pep.sort_values("prob", ascending=False).head(30)
    genes = set()
    for col in ("ligand", "receptor"):
        for v in top[col].astype(str):
            genes.update(re.split(r"[_+]", v))
    # named pairs
    genes.update(top["ligand"].astype(str).str.split("_").str[0])
    genes.update(["Ptn", "Sdc3", "Ncl", "Cadm1", "Lama2", "Lamc1", "Cd44", "Ncam1", "Ncam2", "L1cam"])
    return {g for g in genes if g and g != "nan" and len(g) > 1}


def celloracle_ko_tfs() -> set[str]:
    path = ROOT / "results/celloracle/satglia_pep/ko_monocle3_ranked.csv"
    if path.exists():
        return set(pd.read_csv(path)["TF"].astype(str))
    return set()


def overlap_table(sets: dict[str, set[str]]) -> pd.DataFrame:
    names = list(sets.keys())
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = sets[a] & sets[b]
            if not inter:
                continue
            rows.append({
                "set_a": a, "set_b": b, "n_intersection": len(inter),
                "genes": "; ".join(sorted(inter)[:40]) + ("…" if len(inter) > 40 else ""),
            })
    return pd.DataFrame(rows)


def main():
    WEB_DATA.mkdir(parents=True, exist_ok=True)

    gs_path = ROOT / "results/archive/20260727T013911/mouse_gwas_ihgc/migraine_ihgc_from_snps.gs"
    scdrs_path = ROOT / "results/scdrs/mouse/downstream_provisional/MIGRAINE_PUBLISHED_LOCI.scdrs_gene"
    gwas_tf_path = ROOT / "results/celloracle/mouse_gwas_seed_tfs/Table_prioritized_TFs_for_CellOracle.csv"

    sets = {
        "scDRS_top100": scdrs_top(scdrs_path, 100),
        "GWAS_gene_set": parse_gs_genes(gs_path, 150),
        "enrichr_pathway_genes": enrich_deg_genes(),
        "enrich_IS_DEGs": enrich_input_degs(ROOT / "results/objects/mouse_rna_focus_subcluster_r01.h5ad"),
        "monocle3_heatmap": monocle_heatmap_genes(),
        "pycistarget_TF_motifs": pycistarget_tfs(),
        "cellchat_LR": cellchat_genes(),
        "celloracle_KO_TF": celloracle_ko_tfs(),
    }

    # GWAS TF peaks (for report)
    if gwas_tf_path.exists():
        gwas_tf_loci(gwas_tf_path).to_csv(WEB_DATA / "gwas_tf_peak_loci.csv", index=False)

    ov = overlap_table(sets)
    ov.to_csv(WEB_DATA / "cross_module_gene_overlap.csv", index=False)

    summary = {}
    for k, v in sets.items():
        summary[k] = {"n": len(v), "sample": sorted(v)[:25]}
    (WEB_DATA / "cross_module_gene_sets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # pathway biology table
    enr = pd.read_csv(ROOT / "results/enrichment/subcluster_r01/enrichr_all_subclusters.csv")
    enr_up = enr[enr["direction"] == "IS_up"].drop_duplicates("Term")
    bio_rows = []
    for _, r in enr_up.iterrows():
        term = r["Term"]
        note = PATHWAY_MIGRAINE.get(term, "Inflammatory/metabolic/stress module under IS headache model; cross-check with subcluster cell type.")
        bio_rows.append({
            "subcluster_r01": r["subcluster_r01"],
            "pathway": term,
            "neglog10_padj": r["neglog10_padj"],
            "migraine_biology": note,
        })
    bio_df = pd.DataFrame(bio_rows).sort_values(["subcluster_r01", "neglog10_padj"], ascending=[True, False])
    bio_df.to_csv(WEB_DATA / "pathway_migraine_biology.csv", index=False)
    print(f"Wrote overlap ({len(ov)} pairs), pathway biology ({len(bio_df)} rows)")
    print(ov.head(15).to_string())


if __name__ == "__main__":
    main()
