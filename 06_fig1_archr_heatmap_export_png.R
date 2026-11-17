#!/usr/bin/env Rscript
# Regenerate Fig. 1d PNG only (grcm39, single-column groupBy legend).
suppressPackageStartupMessages(library(ArchR))
addArchRThreads(threads = 8)
ROOT <- "/opt/NAS/RD/zym/test/migraine"
OUT <- file.path(ROOT, "results/figures/paper")
source(file.path(ROOT, "scripts/25c_archr_heatmap_onecol.R"), local = TRUE)
proj_path <- file.path(ROOT, "results/archr/mouse/ArchR_TG_mouse")
png_path <- file.path(OUT, "Fig_archr_Peak2GeneHeatmap_grcm39_bySubtype.png")
proj <- loadArchRProject(proj_path, showLogo = FALSE)
plot_peak2gene_manuscript_png(proj, png_path, groupBy = "subtype")
message("Wrote ", png_path)
