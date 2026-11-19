#!/usr/bin/env Rscript
# Fig. 1d — ArchR Peak2Gene heatmap (GRCm39 project, 16 subtypes on columns).
suppressPackageStartupMessages(library(ArchR))

addArchRThreads(threads = 8)
addArchRGenome("mm10")

ROOT <- Sys.getenv("MIGRAINE_REPO_ROOT", "/pathto/migraine")
OUT <- file.path(ROOT, "results/figures/paper")
PROJ <- file.path(ROOT, "results/archr/mouse/ArchR_TG_mouse")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
source(file.path(ROOT, "paper_code/05_fig1_archr_heatmap_onecol_legend.R"), local = TRUE)

proj <- loadArchRProject(PROJ, showLogo = FALSE)

pdf_path <- file.path(OUT, "Fig_archr_Peak2GeneHeatmap_grcm39_bySubtype.pdf")
pdf(pdf_path, width = 10, height = 10)
print(plotPeak2GeneHeatmap(
  ArchRProj = proj,
  corCutOff = 0.45,
  FDRCutOff = 1e-4,
  groupBy = "subtype"
))
dev.off()

png_path <- file.path(OUT, "Fig_archr_Peak2GeneHeatmap_grcm39_bySubtype.png")
plot_peak2gene_manuscript_png(proj, png_path, groupBy = "subtype")
message("Wrote ", pdf_path, "\nWrote ", png_path)
