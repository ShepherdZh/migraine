# ArchR Peak2Gene manuscript heatmap: groupBy annotation legend in a single column.
# Expects plotPeak2GeneHeatmap(..., returnMatrices = TRUE) objects with ATAC/RNA
# matrices, colData, and kmeansId splits (inspect once with str() on the server).
suppressPackageStartupMessages({
  library(ArchR)
  library(ComplexHeatmap)
  library(grid)
})

p2g_clip_rescale <- function(mat, limits = c(-2, 2)) {
  mat[mat > limits[2]] <- limits[2]
  mat[mat < limits[1]] <- limits[1]
  (mat - limits[1]) / (limits[2] - limits[1])
}

p2g_heatmap_block <- function(mat, colData, colorMap, kmeans_split, heatmap_name, anno_leg) {
  limits <- c(-2, 2)
  mat <- p2g_clip_rescale(mat, limits)
  heat_colors <- circlize::colorRamp2(
    seq(0, 1, length.out = 100),
    ArchR::paletteContinuous(set = "solarExtra", n = 100)
  )
  col_df <- data.frame(colData)
  color_map <- ArchR:::.colorMapForCH(colorMap, col_df)
  show_legend <- ArchR:::.checkShowLegend(
    color_map[match(names(color_map), colnames(col_df))]
  )
  top_anno <- HeatmapAnnotation(
    df = col_df,
    col = color_map,
    show_legend = show_legend,
    show_annotation_name = TRUE,
    gp = gpar(col = "NA"),
    annotation_legend_param = anno_leg
  )
  Heatmap(
    matrix = as.matrix(mat),
    name = heatmap_name,
    col = heat_colors,
    heatmap_legend_param = list(
      at = c(0, 1),
      labels = c(limits[1], limits[2]),
      color_bar = "continuous",
      legend_direction = "horizontal",
      legend_width = unit(3, "cm"),
      labels_gp = gpar(fontsize = 11)
    ),
    show_column_names = FALSE,
    cluster_columns = FALSE,
    show_column_dend = FALSE,
    show_row_names = FALSE,
    cluster_rows = FALSE,
    show_row_dend = FALSE,
    split = kmeans_split,
    top_annotation = top_anno,
    use_raster = TRUE,
    raster_device = "png",
    raster_quality = 5
  )
}

plot_peak2gene_manuscript_png <- function(
    proj,
    out_png,
    groupBy = "subtype",
    width = 2000,
    height = 2000,
    res = 220
) {
  mats <- plotPeak2GeneHeatmap(
    ArchRProj = proj,
    corCutOff = 0.45,
    FDRCutOff = 1e-4,
    groupBy = groupBy,
    returnMatrices = TRUE
  )
  vals <- gtools::mixedsort(unique(as.character(mats$ATAC$colData[[1]])))
  pal <- ArchR::paletteDiscrete(values = vals)
  colorMap <- list(groupBy = pal)
  attr(colorMap[[1]], "discrete") <- TRUE
  anno_leg <- list(
    ncol = 1,
    labels_gp = gpar(fontsize = 12),
    title_gp = gpar(fontsize = 13, fontface = "bold")
  )

  htATAC <- p2g_heatmap_block(
    mats$ATAC$matrix,
    mats$ATAC$colData,
    colorMap,
    mats$ATAC$kmeansId,
    paste0("ATAC Z-Scores\n", nrow(mats$ATAC$matrix), " P2GLinks"),
    anno_leg
  )
  htRNA <- p2g_heatmap_block(
    mats$RNA$matrix,
    mats$RNA$colData,
    colorMap,
    mats$RNA$kmeansId,
    paste0("RNA Z-Scores\n", nrow(mats$RNA$matrix), " P2GLinks"),
    anno_leg
  )

  png(out_png, width = width, height = height, res = res)
  draw(
    htATAC + htRNA,
    padding = unit(c(45, 45, 45, 45), "mm"),
    heatmap_legend_side = "bottom",
    annotation_legend_side = "right"
  )
  dev.off()
  invisible(out_png)
}
