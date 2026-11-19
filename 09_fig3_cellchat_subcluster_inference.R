#!/usr/bin/env Rscript
# CellChat on leiden subclusters; one PNG per top L-R pair / pathway.
suppressPackageStartupMessages({
  library(CellChat)
  library(Matrix)
})

ROOT <- Sys.getenv("MIGRAINE_REPO_ROOT", "/opt/NAS/RD/zym/test/migraine")
OUT <- file.path(ROOT, "results/cellchat/satglia_pep_subcluster")
FIG <- file.path(OUT, "figures_merged")
RDS <- file.path(OUT, "cellchat_subcluster_merged.rds")
LR_CSV <- file.path(OUT, "lr_pairs_merged.csv")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)

mat <- readMM(file.path(OUT, "matrix.mtx.gz"))
meta <- read.csv(file.path(OUT, "meta.csv"), row.names = 1, stringsAsFactors = FALSE)
features <- read.csv(file.path(OUT, "features.csv"), stringsAsFactors = FALSE)

rownames(mat) <- features$gene
colnames(mat) <- rownames(meta)
mat <- as(mat, "CsparseMatrix")

meta$labels <- factor(meta$labels, levels = unique(meta$labels))
cellchat <- createCellChat(object = mat, meta = meta, group.by = "labels")
cellchat@DB <- CellChatDB.mouse
cellchat <- subsetData(cellchat)
cellchat <- identifyOverExpressedGenes(cellchat)
cellchat <- identifyOverExpressedInteractions(cellchat)
cellchat <- computeCommunProb(cellchat, type = "triMean")
cellchat <- filterCommunication(cellchat, min.cells = 10)
cellchat <- computeCommunProbPathway(cellchat)
cellchat <- aggregateNet(cellchat)
cellchat <- netAnalysis_computeCentrality(cellchat, slot.name = "netP")

saveRDS(cellchat, RDS)
df.net <- subsetCommunication(cellchat)
write.csv(df.net, LR_CSV, row.names = FALSE)

save_png <- function(path, w, h, plot_fun) {
  png(path, width = w * 200, height = h * 200, res = 200)
  plot_fun()
  dev.off()
}

# Incoming / outgoing heatmaps by subcluster
save_png(
  file.path(FIG, "Fig_cellchat_subcluster_outgoing.png"), 12, 9,
  function() netAnalysis_signalingRole_heatmap(cellchat, pattern = "outgoing", font.size = 7)
)
save_png(
  file.path(FIG, "Fig_cellchat_subcluster_incoming.png"), 12, 9,
  function() netAnalysis_signalingRole_heatmap(cellchat, pattern = "incoming", font.size = 7)
)

# Top Satglia-subcluster → PEP-subcluster pairs: one bubble per interaction
sat_pep <- df.net[grepl("^Satglia_", df.net$source) & grepl("^PEP_", df.net$target), ]
sat_pep <- sat_pep[order(-sat_pep$prob), ]
top_pairs <- head(unique(sat_pep$interaction_name), 8)

for (nm in top_pairs) {
  pairLR <- data.frame(interaction_name = nm, stringsAsFactors = FALSE)
  safe <- gsub("[^A-Za-z0-9_+-]", "_", nm)
  save_png(
    file.path(FIG, paste0("Fig_cellchat_LR_", safe, ".png")), 10, 7,
    function() netVisual_bubble(cellchat, pairLR.use = pairLR, remove.isolate = TRUE)
  )
}

# One pathway per figure (PTN, CADM, LAMININ, COLLAGEN)
pathways <- c("PTN", "CADM", "LAMININ", "COLLAGEN")
for (pw in pathways) {
  save_png(
    file.path(FIG, paste0("Fig_cellchat_pathway_", pw, ".png")), 11, 8,
    function() netVisual_bubble(cellchat, signaling = pw, remove.isolate = TRUE)
  )
}

# Subcluster-level circle (all subclusters)
groupSize <- as.numeric(table(cellchat@idents))
save_png(
  file.path(FIG, "Fig_cellchat_subcluster_circle.png"), 10, 10,
  function() netVisual_circle(
    cellchat@net$weight, vertex.weight = groupSize,
    weight.scale = TRUE, label.edge = FALSE,
    title.name = "Subcluster interaction network"
  )
)

message("CellChat subcluster done → ", OUT)
