#!/usr/bin/env Rscript
# Regenerate Fig. 1d PNG only (grcm39, single-column groupBy legend).
suppressPackageStartupMessages(library(ArchR))
addArchRThreads(threads = 8)
ROOT <- Sys.getenv("MIGRAINE_REPO_ROOT", "/opt/NAS/RD/zym/test/migraine")
OUT <- file.path(ROOT, "results/figures/paper")
source(file.path(ROOT, "paper_code/05_fig1_archr_heatmap_onecol_legend.R"), local = TRUE)
proj <- loadArchRProject(file.path(ROOT, "results/archr/mouse/ArchR_TG_mouse"), showLogo = FALSE)
png_path <- file.path(OUT, "Fig_archr_Peak2GeneHeatmap_grcm39_bySubtype.png")
plot_peak2gene_manuscript_png(proj, png_path, groupBy = "subtype")
message("Wrote ", png_path)
