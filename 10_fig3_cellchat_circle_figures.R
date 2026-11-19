#!/usr/bin/env Rscript
# CellChat circle figures — filtered edges, migraine-relevant pathways only.
suppressPackageStartupMessages({
  library(CellChat)
  suppressPackageStartupMessages(library(ComplexHeatmap))
})

ROOT <- Sys.getenv("MIGRAINE_REPO_ROOT", "/opt/NAS/RD/zym/test/migraine")
OUT <- file.path(ROOT, "results/cellchat/satglia_pep_subcluster")
FIG <- file.path(OUT, "figures_merged")
RDS <- file.path(OUT, "cellchat_subcluster_merged.rds")
LR_CSV <- file.path(OUT, "lr_pairs_merged.csv")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)

cellchat <- readRDS(RDS)
df.net <- read.csv(LR_CSV, stringsAsFactors = FALSE)
groupSize <- as.numeric(table(cellchat@idents))
prob_arr <- cellchat@netP$prob
pathway_names <- dimnames(prob_arr)[[3]]
labels <- levels(cellchat@idents)

save_png <- function(path, w, h, plot_fun) {
  png(path, width = w * 200, height = h * 200, res = 200)
  plot_fun()
  dev.off()
}

# Keep Satglia + PEP subclusters only for focused circle plots
focus_idx <- which(grepl("^(Satglia_|PEP_)", labels))
focus_labels <- labels[focus_idx]

filter_mat <- function(mat, top_n = 12, min_prob = 0) {
  m <- mat[focus_idx, focus_idx, drop = FALSE]
  if (length(dim(m)) == 3) m <- drop(m)
  dimnames(m) <- list(focus_labels, focus_labels)
  trim_edges(m, top_n, min_prob)
}

trim_edges <- function(m, top_n = 12, min_prob = 0) {
  m[m < min_prob] <- 0
  vals <- sort(m[m > 0], decreasing = TRUE)
  if (length(vals) > top_n) {
    thr <- vals[top_n]
    m[m < thr] <- 0
  }
  m
}

save_png(file.path(FIG, "Fig_cellchat_subcluster_outgoing.png"), 10, 8, function() {
  par(mar = c(1.2, 1.2, 2.0, 1.2), oma = c(0, 0, 0, 0))
  ht <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "outgoing", font.size = 9)
  draw(ht, padding = grid::unit(c(1, 1, 1, 1), "mm"))
})
save_png(file.path(FIG, "Fig_cellchat_subcluster_incoming.png"), 10, 8, function() {
  par(mar = c(1.2, 1.2, 2.0, 1.2), oma = c(0, 0, 0, 0))
  ht <- netAnalysis_signalingRole_heatmap(cellchat, pattern = "incoming", font.size = 9)
  draw(ht, padding = grid::unit(c(1, 1, 1, 1), "mm"))
})

# Pathways: migraine-relevant, exclude BMP (fibroblast baseline)
pathways <- c("PTN", "CADM", "LAMININ", "NCAM")
for (pw in pathways) {
  mat <- filter_mat(prob_arr[, , pw], top_n = 10)
  outp <- file.path(FIG, paste0("Fig_cellchat_circle_pathway_", pw, ".png"))
  save_png(outp, 10, 10, function(m = mat, p = pw) {
    par(mar = c(0.2, 0.2, 2.6, 0.2))
    netVisual_circle(m, vertex.weight = groupSize[focus_idx], weight.scale = TRUE,
                     label.edge = FALSE, title.name = NULL)
    title(main = paste0(p, " pathway · Satglia ↔ PEP"), cex.main = 1.15, font.main = 2, line = 0.2)
  })
}

# Top Satglia→PEP L-R pairs only (prob-ranked, max 5)
sat_pep <- df.net[grepl("^Satglia_", df.net$source) & grepl("^PEP_", df.net$target), ]
sat_pep <- sat_pep[order(-sat_pep$prob), ]
sat_pep <- sat_pep[!grepl("BMP", sat_pep$pathway_name, ignore.case = TRUE), ]
top_pairs <- head(unique(sat_pep$interaction_name), 5)

for (nm in top_pairs) {
  pw <- sat_pep[sat_pep$interaction_name == nm, "pathway_name"][1]
  mat <- filter_mat(prob_arr[, , pw], top_n = 8)
  safe <- gsub("[^A-Za-z0-9_+-]", "_", nm)
  outp <- file.path(FIG, paste0("Fig_cellchat_circle_LR_", safe, ".png"))
  save_png(outp, 10, 10, function(m = mat, n = nm, p = pw) {
    netVisual_circle(m, vertex.weight = groupSize[focus_idx], weight.scale = TRUE,
                     label.edge = FALSE, title.name = paste0("L-R: ", n, " (", p, ")"))
  })
}

# Aggregated Satglia↔PEP net (filtered)
agg <- matrix(0, length(focus_labels), length(focus_labels), dimnames = list(focus_labels, focus_labels))
for (pw in pathways) {
  m <- prob_arr[focus_idx, focus_idx, pw]
  agg <- agg + m
}
agg <- trim_edges(agg, top_n = 15)
save_png(file.path(FIG, "Fig_cellchat_subcluster_circle.png"), 10, 10, function() {
  netVisual_circle(agg, vertex.weight = groupSize[focus_idx], weight.scale = TRUE,
                   label.edge = FALSE, title.name = "Satglia↔PEP top interactions")
})

WEB <- file.path(ROOT, "results/reports/web/fig")
dir.create(WEB, recursive = TRUE, showWarnings = FALSE)
for (nm in c(
  "Fig_cellchat_subcluster_outgoing.png",
  "Fig_cellchat_subcluster_incoming.png"
)) {
  file.copy(file.path(FIG, nm), file.path(WEB, sub("^Fig_", "", nm)), overwrite = TRUE)
}

message("CellChat circle figures done → ", FIG)
