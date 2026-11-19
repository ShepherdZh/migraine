#!/usr/bin/env Rscript
# Monocle3 trajectory on Satglia + PEP (Naive/IS/CSD).
suppressPackageStartupMessages({
  library(monocle3)
  library(Matrix)
  library(ggplot2)
  library(dplyr)
})

ROOT <- Sys.getenv("MIGRAINE_REPO_ROOT", "/opt/NAS/RD/zym/test/migraine")
OUT <- file.path(ROOT, "results/monocle3/satglia_pep")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

mat <- readMM(file.path(OUT, "counts.mtx.gz"))
genes <- read.csv(file.path(OUT, "gene_metadata.csv"), stringsAsFactors = FALSE)
meta <- read.csv(file.path(OUT, "cell_metadata.csv"), row.names = 1, stringsAsFactors = FALSE)

rownames(mat) <- genes$gene_short_name
colnames(mat) <- rownames(meta)

cds <- new_cell_data_set(
  mat,
  cell_metadata = meta,
  gene_metadata = data.frame(gene_short_name = rownames(mat), row.names = rownames(mat))
)

cds <- preprocess_cds(cds, num_dim = 50, verbose = FALSE)
cds <- reduce_dimension(cds, verbose = FALSE)
cds <- cluster_cells(cds, verbose = FALSE)
cds <- learn_graph(cds, verbose = FALSE)

# Root pseudotime at Naive Satglia (GSE197289 TG baseline glia)
root_cells <- rownames(meta)[meta$subtype == "Satglia" & meta$model == "Naive"]
cds <- order_cells(cds, root_cells = root_cells, verbose = FALSE)

saveRDS(cds, file.path(OUT, "cds_satglia_pep.rds"))

pt <- pseudotime(cds)
write.csv(data.frame(cell = names(pt), pseudotime = pt), file.path(OUT, "pseudotime.csv"), row.names = FALSE)

# Figures
p1 <- plot_cells(cds, color_cells_by = "subtype", label_cell_groups = FALSE,
                 graph_label_size = 3) + ggtitle("Satglia-PEP monocle3")
ggsave(file.path(OUT, "Fig_monocle3_subtype.pdf"), p1, width = 7, height = 6)
ggsave(file.path(OUT, "Fig_monocle3_subtype.png"), p1, width = 7, height = 6, dpi = 200)

p2 <- plot_cells(cds, color_cells_by = "pseudotime", label_cell_groups = FALSE) +
  ggtitle("Pseudotime (root = Satglia)")
ggsave(file.path(OUT, "Fig_monocle3_pseudotime.pdf"), p2, width = 7, height = 6)

p3 <- plot_cells(cds, color_cells_by = "model", label_cell_groups = FALSE) +
  ggtitle("Condition")
ggsave(file.path(OUT, "Fig_monocle3_model.pdf"), p3, width = 7, height = 6)

message("Monocle3 done → ", OUT)
