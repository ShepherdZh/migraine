import sys
#!/usr/bin/env python3
"""Export Satglia + PEP subset for monocle3 (Naive/IS/CSD)."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
SRC = ROOT / "results/objects/mouse_rna_Naive_IS_CSD_reintegrated.h5ad"
if not SRC.exists():
    SRC = ROOT / "results/objects/mouse_rna_full.h5ad"
OUT = ROOT / "results/monocle3/satglia_pep"
SUBTYPES = ["Satglia", "PEP"]
MODELS = ["Naive", "IS", "CSD"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(SRC)
    adata = adata[adata.obs["model"].isin(MODELS) & adata.obs["subtype"].isin(SUBTYPES)].copy()
    print(f"Subset: {adata.n_obs} cells ({adata.obs['subtype'].value_counts().to_dict()})")

    # monocle3 expects non-log counts; use raw if present else reverse log1p approx
    if adata.raw is not None:
        X = adata.raw[:, adata.var_names].X
    elif "counts" in adata.layers:
        X = adata.layers["counts"]
    else:
        X = np.expm1(adata.X)
        X = sparse.csr_matrix(X) if sparse.issparse(adata.X) else np.asarray(X)

    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = sparse.csr_matrix(X)

    mmwrite(OUT / "counts.mtx", X.T)  # genes x cells
    with gzip.open(OUT / "counts.mtx.gz", "wb") as fo, open(OUT / "counts.mtx", "rb") as fi:
        fo.write(fi.read())
    (OUT / "counts.mtx").unlink()

    meta = adata.obs[["subtype", "model"]].copy()
    meta.index.name = "cell"
    meta.to_csv(OUT / "cell_metadata.csv")
    pd.DataFrame({"gene_short_name": adata.var_names}).to_csv(OUT / "gene_metadata.csv", index=False)
    adata.write_h5ad(OUT / "satglia_pep.h5ad")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
