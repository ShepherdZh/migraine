# Paper analysis code (GitHub release)

Scripts are **renamed in narrative order** (preprocess → Figures 1–6 → supplementary tables).
The original server filename is listed in [`FIGURE_TABLE_MAP.md`](FIGURE_TABLE_MAP.md).

Regenerate from the full tree on the analysis server:

```bash
python3 scripts/build_paper_code.py
```

See [../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) and [../DATA.md](../DATA.md).

## Dependencies

| File | Use |
|------|-----|
| `requirements.txt` | pip: Scanpy track (scripts 01–14, 22) |
| `requirements-scenicplus.txt` | pip add-on: Fig 5–6 (SCENIC+/CellOracle) |
| `requirements-r.txt` | R package list (ArchR, CellChat, Monocle3) |
| `environment.yml` | conda env `satglia-scanpy` (core Python stack) |
| `pyproject.toml` | PEP 621 metadata + same Python deps |

Full versions on the authors' server: Supplementary **Table S6**.
