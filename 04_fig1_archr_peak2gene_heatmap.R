#!/usr/bin/env Rscript
# Plot ArchR Peak2Gene heatmaps for GRCm39 and mm10 projects.
suppressPackageStartupMessages({
  library(ArchR)
  library(ggplot2)
})

addArchRThreads(threads = 8)
addArchRGenome("mm10")  # overridden per-project via geneAnnotation if needed

ROOT <- "/opt/NAS/RD/zym/test/migraine"
OUT <- file.path(ROOT, "results/figures/paper")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
source(file.path(ROOT, "scripts/25c_archr_heatmap_onecol.R"), local = TRUE)

# Ensure subtype labels exist on ArchR projects (for groupBy)
source(file.path(ROOT, "scripts/25b_archr_add_subtype.R"), local = TRUE)
for (pp in c(
  file.path(ROOT, "results/archr/mouse/ArchR_TG_mouse"),
  file.path(ROOT, "results/archr/mouse_mm10/ArchR_TG_mouse_mm10")
)) {
  if (dir.exists(pp)) tryCatch({
    proj_tmp <- annotate_subtype(pp)
    saveArchRProject(ArchRProj = proj_tmp, load = FALSE)
  }, error = function(e) message("subtype annotate warn: ", conditionMessage(e)))
}

configs <- list(
  list(
    name = "grcm39",
    proj = file.path(ROOT, "results/archr/mouse/ArchR_TG_mouse"),
    genome = "mm10"  # ArchR often still uses mm10 annotation style; project already built
  ),
  list(
    name = "mm10",
    proj = file.path(ROOT, "results/archr/mouse_mm10/ArchR_TG_mouse_mm10"),
    genome = "mm10"
  )
)

add_subtype_major <- function(proj) {
  cd <- getCellColData(proj, select = c("subtype"))
  st <- as.character(cd$subtype)
  major <- st
  major[st %in% c("NP", "TRPM8", "cLTMR", "SST", "NF1", "NF2", "NF3")] <- "Other neuron"
  major[st %in% c("Schwann_N", "Schwann_M")] <- "Schwann"
  major[st %in% c("Fibroblast_Dcn", "Fibroblast_Mgp")] <- "Fibroblast"
  major[st %in% c("Vascular", "Immune", "Injured")] <- "Vascular / immune"
  major[st == "PEP"] <- "PEP"
  major[st == "Satglia"] <- "Satglia"
  addCellColData(
    ArchRProj = proj,
    data = major,
    cells = rownames(cd),
    name = "subtype_major",
    force = TRUE
  )
}

plot_one <- function(proj_path, label) {
  message("=== Loading ", label, " : ", proj_path)
  if (!dir.exists(proj_path)) {
    message("SKIP missing project: ", proj_path)
    return(invisible(NULL))
  }
  proj <- loadArchRProject(path = proj_path, showLogo = FALSE)
  proj <- add_subtype_major(proj)

  # Ensure Peak2GeneLinks exist
  p2g <- tryCatch(getPeak2GeneLinks(proj), error = function(e) NULL)
  if (is.null(p2g) || length(p2g) == 0) {
    message("No Peak2GeneLinks in ", label, " — skip heatmap")
    return(invisible(NULL))
  }

  # PDF (full 16-type groupBy)
  pdf_path <- file.path(OUT, paste0("Fig_archr_Peak2GeneHeatmap_", label, "_bySubtype.pdf"))
  message("Writing ", pdf_path)
  p_full <- plotPeak2GeneHeatmap(
    ArchRProj = proj,
    corCutOff = 0.45,
    FDRCutOff = 1e-4,
    groupBy = "subtype"
  )
  pdf(pdf_path, width = 10, height = 10)
  print(p_full)
  dev.off()

  # PNG for manuscript Fig. 1d: all subtypes; groupBy legend in one column
  png_path <- file.path(OUT, paste0("Fig_archr_Peak2GeneHeatmap_", label, "_bySubtype.png"))
  message("Writing ", png_path, " (groupBy = subtype, legend ncol = 1)")
  tryCatch(
    plot_peak2gene_manuscript_png(proj, png_path, groupBy = "subtype"),
    error = function(e) {
      message("PNG failed for ", label, ": ", conditionMessage(e))
      if (dev.cur() > 1) dev.off()
    }
  )

  message("DONE ", label)
  invisible(NULL)
}

for (cfg in configs) {
  tryCatch(
    plot_one(cfg$proj, cfg$name),
    error = function(e) message("ERROR ", cfg$name, ": ", conditionMessage(e))
  )
}

message("All Peak2Gene heatmap jobs finished → ", OUT)
