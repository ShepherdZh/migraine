import sys
#!/usr/bin/env python3
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from project_paths import ROOT
WEB_FIG = ROOT / "results/reports/web/fig"
PYCIS = ROOT / "results/scenicplus/mouse/run/egrn_pycistopic/pycistarget"

# Topics linked to Satglia/PEP story
PANELS = {
    "pycistarget_motif_logos_topic9.png": (9, ["Prdm16", "Isl1", "Ebf1"]),
    "pycistarget_motif_logos_topic3.png": (3, ["Runx1", "Runx2", "Runx3", "Isl1"]),
    "pycistarget_motif_logos_topic1.png": (1, ["Nfia", "Nfib", "Nfix", "Ctcf"]),
}


def logo_url(html: str) -> str | None:
    m = re.search(r'src="(https://motifcollections[^"]+\.png)"', str(html))
    return m.group(1) if m else None


def tf_label(row) -> str:
    for col in ("Direct_annot", "Orthology_annot"):
        v = str(row.get(col, ""))
        if v and v != "nan":
            return v.split(",")[0].strip()
    return str(row.name)[:16]


def load_logos(topic: int, prefer_tfs: list[str]) -> list[tuple[str, Image.Image, float]]:
    tsv = PYCIS / f"CTX_Topic{topic}_All/motif_enrichment_cistarget_Topic{topic}.tsv"
    df = pd.read_csv(tsv, sep="\t").sort_values("NES", ascending=False)
    chosen = []
    used = set()
    # prefer TF names of interest
    for tf in prefer_tfs:
        hit = df[
            df["Direct_annot"].astype(str).str.contains(tf, case=False, na=False)
            | df["Orthology_annot"].astype(str).str.contains(tf, case=False, na=False)
        ]
        for _, r in hit.head(1).iterrows():
            url = logo_url(r.get("Logo", ""))
            if url and tf not in used:
                chosen.append((tf, url, float(r["NES"])))
                used.add(tf)
    # fill up to 6 with top NES
    for _, r in df.iterrows():
        if len(chosen) >= 6:
            break
        lab = tf_label(r)
        if lab in used:
            continue
        url = logo_url(r.get("Logo", ""))
        if url:
            chosen.append((lab, url, float(r["NES"])))
            used.add(lab)
    imgs = []
    for lab, url, nes in chosen[:6]:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            imgs.append((lab, Image.open(BytesIO(resp.content)), nes))
        except Exception as e:
            print(f"Skip logo {lab}: {e}")
    return imgs


def compose_panel(imgs: list[tuple[str, Image.Image, float]], title: str, out_name: str) -> None:
    if not imgs:
        return
    n = len(imgs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = list(axes.flatten())
    for ax, (lab, img, nes) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_title(f"{lab}\nNES={nes:.1f}", fontsize=12, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    WEB_FIG.mkdir(parents=True, exist_ok=True)
    dest = WEB_FIG / out_name
    fig.savefig(dest, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {dest}")


def main():
    for out_name, (topic, tfs) in PANELS.items():
        imgs = load_logos(topic, tfs)
        compose_panel(
            imgs,
            f"pycistarget Topic{topic} — top motif logos",
            out_name,
        )


if __name__ == "__main__":
    main()
