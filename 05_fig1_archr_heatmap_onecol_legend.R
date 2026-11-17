# ArchR Peak2Gene manuscript heatmap: groupBy annotation legend in a single column.
suppressPackageStartupMessages({
  library(ArchR)
  library(ComplexHeatmap)
  library(grid)
})

`.ArchRHeatmap1Col` <- function(
    mat = NULL,
    scale = FALSE,
    limits = c(min(mat), max(mat)),
    colData = NULL,
    color = ArchR::paletteContinuous(set = "solarExtra", n = 100),
    clusterCols = TRUE,
    clusterRows = FALSE,
    labelCols = FALSE,
    labelRows = FALSE,
    colorMap = NULL,
    useRaster = TRUE,
    rasterQuality = 5,
    split = NULL,
    fontSizeRows = 10,
    fontSizeCols = 10,
    fontSizeLabels = 12,
    showRowDendrogram = FALSE,
    showColDendrogram = FALSE,
    customColLabel = NULL,
    customColLabelIDs = NULL,
    customLabelWidth = 0.75,
    rasterDevice = "png",
    padding = 45,
    borderColor = NA,
    draw = TRUE,
    name = "Heatmap",
    annotation_legend_param = list(
      ncol = 1,
      labels_gp = gpar(fontsize = 12),
      title_gp = gpar(fontsize = 13, fontface = "bold")
    )
) {
  if (scale) {
    mat <- ArchR:::.rowZscores(mat, limit = FALSE)
    name <- paste0(name, " Z-Scores")
  }
  if (is.null(colorMap)) {
    colorMap <- ArchR:::.colorMapAnno(colData)
  }
  if (!is.null(colData)) {
    colData <- data.frame(colData)
    colorMap <- ArchR:::.colorMapForCH(colorMap, colData)
    showLegend <- ArchR:::.checkShowLegend(
      colorMap[match(names(colorMap), colnames(colData))]
    )
  } else {
    colorMap <- NULL
    showLegend <- NULL
  }
  if (!is.null(limits)) {
    mat[mat > max(limits)] <- max(limits)
    mat[mat < min(limits)] <- min(limits)
  } else {
    limits <- c(round(min(mat), 2), round(max(mat), 2))
  }
  mat <- (mat - min(limits)) / (max(limits) - min(limits))
  breaks <- seq(0, 1, length.out = length(color))
  color <- circlize::colorRamp2(breaks, color)
  if (exists("anno_mark", where = "package:ComplexHeatmap", mode = "function")) {
    anno_cols <- ComplexHeatmap::anno_mark
  } else {
    anno_cols <- ComplexHeatmap::column_anno_link
  }
  if (!is.null(colData) && !is.null(customColLabel)) {
    if (is.null(customColLabelIDs)) {
      customColLabelIDs <- colnames(mat)[customColLabel]
    }
    ht1Anno <- HeatmapAnnotation(
      df = colData, col = colorMap,
      show_legend = showLegend, show_annotation_name = TRUE,
      gp = gpar(col = "NA"),
      annotation_legend_param = annotation_legend_param,
      foo = anno_cols(
        at = customColLabel,
        labels = customColLabelIDs,
        labels_gp = gpar(fontsize = fontSizeLabels)
      )
    )
  } else if (!is.null(colData)) {
    ht1Anno <- HeatmapAnnotation(
      df = colData, col = colorMap,
      show_legend = showLegend, show_annotation_name = TRUE,
      gp = gpar(col = "NA"),
      annotation_legend_param = annotation_legend_param
    )
  } else if (is.null(colData) && !is.null(customColLabel)) {
    if (is.null(customColLabelIDs)) {
      customColLabelIDs <- colnames(mat)[customColLabel]
    }
    ht1Anno <- HeatmapAnnotation(
      foo = anno_cols(
        at = customColLabel,
        labels = customColLabelIDs,
        labels_gp = gpar(fontsize = fontSizeLabels)
      )
    )
  } else {
    ht1Anno <- NULL
  }
  ht1 <- Heatmap(
    matrix = as.matrix(mat), name = name, col = color,
    heatmap_legend_param = list(
      at = c(0, 1),
      labels = c(round(min(limits), 2), round(max(limits), 2)),
      color_bar = "continuous",
      legend_direction = "horizontal",
      legend_width = unit(3, "cm"),
      labels_gp = gpar(fontsize = 11)
    ),
    rect_gp = gpar(col = borderColor),
    show_column_names = labelCols,
    cluster_columns = clusterCols,
    show_column_dend = showColDendrogram,
    clustering_method_columns = "ward.D2",
    column_names_gp = gpar(fontsize = fontSizeCols),
    column_names_max_height = unit(100, "mm"),
    show_row_names = labelRows,
    row_names_gp = gpar(fontsize = fontSizeRows),
    cluster_rows = clusterRows,
    show_row_dend = showRowDendrogram,
    clustering_method_rows = "ward.D2",
    split = split,
    top_annotation = ht1Anno,
    use_raster = useRaster,
    raster_device = rasterDevice,
    raster_quality = rasterQuality
  )
  if (draw) {
    draw(
      ht1,
      padding = unit(c(padding, padding, padding, padding), "mm"),
      heatmap_legend_side = "bottom",
      annotation_legend_side = "right"
    )
  } else {
    ht1
  }
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
  htATAC <- `.ArchRHeatmap1Col`(
    mat = mats$ATAC$matrix,
    scale = FALSE,
    limits = c(-2, 2),
    colData = mats$ATAC$colData,
    colorMap = colorMap,
    clusterCols = FALSE,
    clusterRows = FALSE,
    split = mats$ATAC$kmeansId,
    labelRows = FALSE,
    labelCols = FALSE,
    draw = FALSE,
    name = paste0("ATAC Z-Scores\n", nrow(mats$ATAC$matrix), " P2GLinks"),
    annotation_legend_param = anno_leg
  )
  htRNA <- `.ArchRHeatmap1Col`(
    mat = mats$RNA$matrix,
    scale = FALSE,
    limits = c(-2, 2),
    colData = mats$RNA$colData,
    colorMap = colorMap,
    clusterCols = FALSE,
    clusterRows = FALSE,
    split = mats$RNA$kmeansId,
    labelRows = FALSE,
    labelCols = FALSE,
    draw = FALSE,
    name = paste0("RNA Z-Scores\n", nrow(mats$RNA$matrix), " P2GLinks"),
    annotation_legend_param = anno_leg
  )
  png(out_png, width = width, height = height, res = res)
  # Match default ArchR layout: expression colour bars below; groupBy legend on the right.
  draw(
    htATAC + htRNA,
    padding = unit(c(45, 45, 45, 45), "mm"),
    heatmap_legend_side = "bottom",
    annotation_legend_side = "right"
  )
  dev.off()
  invisible(out_png)
}
