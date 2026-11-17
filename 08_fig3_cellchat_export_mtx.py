import sys
#!/usr/bin/env python3
"""Export CellChat using unified subcluster_r01 labels."""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import scanpy as sc
from scipy import sparse
from scipy.io import mmwrite

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
SRC = ROOT / "results/objects/mouse_rna_focus_subcluster_r01.h5ad"
OUT = ROOT / "results/cellchat/satglia_pep_subcluster"
CAP = 4000


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(SRC)
    adata.obs["labels"] = adata.obs["subcluster_r01"].astype(str)

    rng = np.random.default_rng(42)
    cells = []
    for lab in sorted(adata.obs["labels"].unique()):
        cids = adata.obs_names[adata.obs["labels"] == lab]
        if len(cids) > CAP:
            cids = rng.choice(cids, size=CAP, replace=False)
        cells.extend(cids)
    adata = adata[cells].copy()
    print(adata.obs.groupby("labels", observed=True).size().sort_values(ascending=False))

    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    mmwrite(OUT / "matrix.mtx", X.T)
    with gzip.open(OUT / "matrix.mtx.gz", "wb") as fo, open(OUT / "matrix.mtx", "rb") as fi:
        fo.write(fi.read())
    (OUT / "matrix.mtx").unlink()

    meta = adata.obs[["labels", "subtype", "model", "subcluster_r01"]].copy()
    meta.index.name = "cell_barcode"
    meta.to_csv(OUT / "meta.csv", index=True)
    import pandas as pd
    pd.DataFrame({"gene": adata.var_names}).to_csv(OUT / "features.csv", index=False)
    adata.write_h5ad(OUT / "cellchat_subcluster.h5ad")
    print(f"Wrote {OUT} ({adata.n_obs} cells, {adata.obs['labels'].nunique()} subclusters)")


if __name__ == "__main__":
    main()
