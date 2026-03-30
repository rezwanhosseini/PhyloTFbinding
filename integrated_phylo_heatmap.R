#!/usr/bin/env Rscript
setwd("/home/seh197/rezwan/research/Maria")

# Integrated phylogenetic heatmap generator
# Combines the logic from phylo_heatmap.R and phylo_heatmap_visionloss.R
# and uses the Zoonomia taxonomy CSV to assign clades.

suppressPackageStartupMessages({
  library(ape)
  library(phytools)
  library(ComplexHeatmap)
  library(circlize)
  library(glue)
  library(stringr)
  library(dplyr)
  library(rhdf5)
  library(tools)
})

# =========================
# User inputs
# =========================
cons_pwm_file    <- "AFconverge_folder/conservation_pwm_analysis/cons_pwm_results.h5"
cons_pwm_summary <- "AFconverge_folder/conservation_pwm_analysis/cons_pwm_summary.csv"
phenotype_name   <- "MeanBodyMass"   # e.g. "MeanBodyMass" or "VisionLoss_CC"

# Optional paths
zoonomia_taxonomy_file <- "/mnt/data/zoonomia_241way_ncbi_taxonomy.csv"
tree_file              <- "AFconverge_folder/AFconverge/tree/tree_241way_mammals_zoonomia.nh"
vgp_map_file           <- "AFconverge_folder/VGP_Species_withZoonomiaNames.txt"

# How to choose promoters:
# 1) If top_genes is not NULL, use those directly.
# 2) Else if promoter_source == "permscores", derive from PermScores.
# 3) Else if promoter_source == "file", read them from promoter_list_file.
promoter_source  <- "permscores"   # "permscores" or "file"
promoter_list_file <- NULL          # e.g. "MotifTSSvalidation/afconverge/eye-specific-genes.txt"
top_genes <- NULL                   # e.g. c("LIM2", "CRYAA")

# Row ordering mode:
# "Clade" = phylogenetic tree order with dendrogram
# phenotype_name = sort by phenotype value and suppress dendrogram
order_rows_by <- phenotype_name     # or "Clade"

# Filter summary rows by data_type when present.
# For the original body-mass code this was "pwm_prob".
# For the vision-loss script it effectively allowed all non-empty motif paths.
required_data_type <- "pwm_prob"   # set to NULL to disable filtering

# Output directory
outdir <- glue("AFconverge_folder/convergeAF_puffin/by_kristina/plots_{phenotype_name}/pwm_hits_clustered_by_{order_rows_by}")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# =========================
# Helpers
# =========================

load_phenotype <- function(phenotype_name, vgp_map_file) {
  phenotype_base <- gsub("_CC$", "", phenotype_name)
  phenotype_vec <- readRDS(glue("AFconverge_folder/AFconverge/phenotype-values/{phenotype_base}_phenotype.RDS"))

  phenotype <- data.frame(
    Zoonomia_name = tolower(names(phenotype_vec)),
    stringsAsFactors = FALSE
  )

  if (phenotype_name == "MeanBodyMass") {
    phenotype$MeanBodyMass <- as.numeric(phenotype_vec)
    phenotype$logMeanBodyMass <- log10(as.numeric(phenotype_vec))
    phenotype$plot_value <- phenotype$logMeanBodyMass
    phenotype$plot_label <- "logMeanBodyMass"
  } else {
    phenotype[[phenotype_base]] <- as.numeric(phenotype_vec)
    phenotype$plot_value <- as.numeric(phenotype_vec)
    phenotype$plot_label <- phenotype_base
  }

  vgp <- read.table(vgp_map_file, sep = "\t", header = TRUE)
  vgp$VGP_name <- sapply(strsplit(vgp$Directory.Name, "__"), `[`, 3)

  phenotype <- phenotype %>%
    inner_join(vgp[, c("Zoonomia_name", "VGP_name", "Common.name")], by = "Zoonomia_name")

  phenotype
}

build_tip_lookup <- function(tree) {
  tips <- tree$tip.label
  lookup <- list()
  for (tip in tips) {
    parts <- strsplit(tip, "_")[[1]]
    if (length(parts) >= 2) {
      key <- paste0(tolower(substr(parts[1], 1, 3)), tolower(substr(parts[2], 1, 3)))
      lookup[[key]] <- tip
    }
  }
  lookup
}

manual_overrides <- c(
  "canFam"      = "Canis_lupus_familiaris",
  "rheMac"      = "Macaca_mulatta",
  "HLcapHir"    = "Capra_hircus",
  "HLcapAeg"    = "Capra_aegagrus",
  "HLcalJacc"   = "Callithrix_jacchus",
  "HLcteGun"    = "Ctenodactylus_gundi",
  "HLequAsin"   = "Equus_asinus",
  "HLequCaba"   = "Equus_caballus",
  "HLictTrid"   = "Ictidomys_tridecemlineatus",
  "HLlycPict"   = "Lycaon_pictus",
  "HLmacNeme"   = "Macaca_nemestrina",
  "HLoryCuni"   = "Oryctolagus_cuniculus",
  "HLoviArie"   = "Ovis_aries",
  "HLoviCana"   = "Ovis_canadensis",
  "HLpanOnca"   = "Panthera_onca",
  "HLrhiBiet"   = "Rhinopithecus_bieti",
  "HLsaiBoli"   = "Saimiri_boliviensis",
  "HLsusScro"   = "Sus_scrofa",
  "HLtapIndi"   = "Tapirus_indicus",
  "HLvicPaco"   = "Vicugna_pacos",
  "HLzalCali"   = "Zalophus_californianus",
  "HLdesRot"    = "Desmodus_rotundus",
  "HLiniGeo"    = "Inia_geoffrensis",
  "HLmesBid"    = "Mesoplodon_bidens",
  "HLminSch"    = "Miniopterus_schreibersii",
  "HLcerSimCot" = "Ceratotherium_simum_cottoni",
  "HLcamDro"    = "Camelus_dromedarius",
  "HLcanLupFam" = "Canis_lupus_familiaris",
  "HLcerSim"    = "Cercocebus_atys",
  "mm"          = "Mus_musculus",
  "rn"          = "Rattus_norvegicus",
  "hg"          = "Homo_sapiens"
)

assembly_to_tip_factory <- function(tips_set, tip_lookup) {
  function(code) {
    prefixes <- names(manual_overrides)[order(nchar(names(manual_overrides)), decreasing = TRUE)]
    for (pfx in prefixes) {
      if (startsWith(code, pfx)) {
        tip <- manual_overrides[[pfx]]
        if (!is.na(tip) && tip %in% tips_set) return(tip)
        return(NA_character_)
      }
    }

    m <- regmatches(code, regexpr("^([a-z]{3})([A-Z][a-z]{2})", code))
    if (length(m) > 0 && nzchar(m)) {
      key <- paste0(tolower(substr(m, 1, 3)), tolower(substr(m, 4, 6)))
      if (!is.null(tip_lookup[[key]])) return(tip_lookup[[key]])
    }

    m <- regmatches(code, regexpr("^HL([a-z]{3})([A-Z][a-z]{2})", code))
    if (length(m) > 0 && nzchar(m)) {
      inner <- sub("^HL", "", m)
      key <- paste0(tolower(substr(inner, 1, 3)), tolower(substr(inner, 4, 6)))
      if (!is.null(tip_lookup[[key]])) return(tip_lookup[[key]])
    }

    m <- regmatches(code, regexpr("^HL([a-z]{3})([A-Z][a-z]{2})[A-Z][a-z]{2}", code))
    if (length(m) > 0 && nzchar(m)) {
      inner <- sub("^HL", "", m)
      key <- paste0(tolower(substr(inner, 1, 3)), tolower(substr(inner, 4, 6)))
      if (!is.null(tip_lookup[[key]])) return(tip_lookup[[key]])
    }

    NA_character_
  }
}

phylo_to_hclust <- function(tree) {
  if (!is.ultrametric(tree, tol = 1e-6)) {
    tree <- force.ultrametric(tree, method = "extend")
  }
  as.hclust.phylo(tree)
}

load_clade_table <- function(zoonomia_taxonomy_file) {
  tax <- read.csv(zoonomia_taxonomy_file, stringsAsFactors = FALSE)
  names(tax) <- tolower(names(tax))

  required <- c("tip", "display_clade")
  missing_cols <- setdiff(required, names(tax))
  if (length(missing_cols) > 0) {
    stop("Taxonomy file is missing required columns: ", paste(missing_cols, collapse = ", "))
  }

  tax$tip <- tolower(tax$tip)
  tax$display_clade[is.na(tax$display_clade) | tax$display_clade == ""] <- "Other"
  tax
}

make_clade_colors <- function(clades) {
  base_cols <- c(
    Primates       = "#E41A1C",
    Rodentia       = "#FF7F00",
    Carnivora      = "#4DAF4A",
    Artiodactyla   = "#377EB8",
    Perissodactyla = "#984EA3",
    Chiroptera     = "#A65628",
    Eulipotyphla   = "#F781BF",
    Lagomorpha     = "#999999",
    Afrotheria     = "#66C2A5",
    Xenarthra      = "#FC8D62",
    Marsupial      = "#8DA0CB",
    Monotreme      = "#E78AC3",
    Pholidota      = "#B15928",
    Other          = "#CCCCCC"
  )

  missing <- setdiff(unique(clades), names(base_cols))
  if (length(missing) > 0) {
    extra <- structure(
      grDevices::hcl.colors(length(missing), palette = "Dynamic"),
      names = missing
    )
    base_cols <- c(base_cols, extra)
  }
  base_cols
}

get_top_promoters <- function(phenotype_name, promoter_source, promoter_list_file, top_genes) {
  if (!is.null(top_genes)) {
    return(unique(top_genes))
  }

  if (promoter_source == "file") {
    if (is.null(promoter_list_file)) stop("promoter_list_file must be set when promoter_source='file'")
    x <- read.table(promoter_list_file, header = FALSE, stringsAsFactors = FALSE)
    return(unique(x[[1]]))
  }

  permscores_file <- glue("AFconverge_folder/AFconverge/PermScores_{phenotype_name}_puffin/PermScores_matrix_puffin_NAsreplacedwith0.RData")
  permscores <- readRDS(permscores_file)
  permscores$promoter <- sub("_TSS.*", "", rownames(permscores))

  score_col <- NULL
  if ("FANTOM CAGE assay +" %in% colnames(permscores)) {
    score_col <- "FANTOM CAGE assay +"
  } else if ("FANTOM_CAGE_assay_fwd_original" %in% colnames(permscores)) {
    score_col <- "FANTOM_CAGE_assay_fwd_original"
  } else {
    stop("Could not find a promoter-ranking column in the PermScores matrix.")
  }

  permscores <- permscores %>%
    group_by(promoter) %>%
    slice_max(abs(.data[[score_col]]), n = 1, with_ties = FALSE) %>%
    ungroup()

  unique(unlist(permscores[order(abs(permscores[[score_col]]), decreasing = TRUE)[1:30], "promoter"]))
}

make_right_annotation <- function(pheno_final, phenotype_name) {
  if (phenotype_name == "MeanBodyMass") {
    rowAnnotation(
      logMeanBodyMass = anno_barplot(
        pheno_final$logMeanBodyMass,
        border = FALSE,
        gp = gpar(fill = "grey70", col = NA),
        axis_param = list(side = "bottom")
      ),
      annotation_name_gp = gpar(fontsize = 7),
      width = unit(3, "cm")
    )
  } else {
    rowAnnotation(
      value = anno_barplot(
        pheno_final$plot_value,
        border = FALSE,
        gp = gpar(fill = "grey70", col = NA),
        axis_param = list(side = "bottom")
      ),
      annotation_name_gp = gpar(fontsize = 7),
      width = unit(3, "cm")
    )
  }
}

# =========================
# Main
# =========================

tree_241 <- read.tree(tree_file)
cat("Tree loaded:", length(tree_241$tip.label), "tips\n")

tip_lookup <- build_tip_lookup(tree_241)
tips_set <- tree_241$tip.label
assembly_to_tip <- assembly_to_tip_factory(tips_set = tips_set, tip_lookup = tip_lookup)

clade_table <- load_clade_table(zoonomia_taxonomy_file)
phenotype <- load_phenotype(phenotype_name, vgp_map_file)
summary_df <- read.csv(cons_pwm_summary, header = TRUE, stringsAsFactors = FALSE)
top_promo <- get_top_promoters(phenotype_name, promoter_source, promoter_list_file, top_genes)

cat("Using", length(top_promo), "promoters\n")

for (gene in top_promo) {
  gene_rows <- summary_df[grep(gene, summary_df$gene), , drop = FALSE]
  gene_data <- gene_rows[gene_rows$gene == gene, , drop = FALSE]
  if (nrow(gene_data) == 0) next

  motif_keep <- gene_data$motif_id != ""
  if (!is.null(required_data_type) && "data_type" %in% colnames(gene_data)) {
    motif_keep <- motif_keep & gene_data$data_type == required_data_type
  }

  motifs <- unique(gene_data$motif_id[motif_keep])
  if (length(motifs) == 0) next

  for (motif in motifs) {
    keep_rows <- gene_data$motif_id == motif
    if (!is.null(required_data_type) && "data_type" %in% colnames(gene_data)) {
      keep_rows <- keep_rows & gene_data$data_type == required_data_type
    }

    pwm_path <- gene_data[keep_rows, "hdf5_path"]
    pwm_path <- pwm_path[!is.na(pwm_path) & pwm_path != ""]
    if (length(pwm_path) == 0) next
    pwm_path <- pwm_path[1]

    mat_raw <- t(h5read(cons_pwm_file, pwm_path))
    species_ids <- as.character(h5read(cons_pwm_file, gsub("/scores", "/species", pwm_path)))
    assembly_codes <- str_remove(species_ids, "_\\[.*$")

    if (length(intersect(assembly_codes, tree_241$tip.label)) > 0) {
      tip_names <- assembly_codes
    } else {
      tip_names <- sapply(assembly_codes, assembly_to_tip)
    }

    keep <- !is.na(tip_names)
    mat_raw <- mat_raw[keep, , drop = FALSE]
    tip_names <- tip_names[keep]
    if (nrow(mat_raw) < 3) next

    dup_mask <- duplicated(tip_names)
    mat_raw <- mat_raw[!dup_mask, , drop = FALSE]
    tip_names <- tip_names[!dup_mask]
    rownames(mat_raw) <- tolower(tip_names)

    pruned_tree <- keep.tip(tree_241, tip_names)
    pruned_tree$tip.label <- tolower(pruned_tree$tip.label)

    tip_df <- data.frame(Zoonomia_name = tolower(tip_names), stringsAsFactors = FALSE)
    pheno_sub <- inner_join(tip_df, phenotype, by = "Zoonomia_name") %>%
      group_by(Zoonomia_name) %>%
      summarise(
        MeanBodyMass = if ("MeanBodyMass" %in% colnames(cur_data())) mean(MeanBodyMass, na.rm = TRUE) else NA_real_,
        logMeanBodyMass = if ("logMeanBodyMass" %in% colnames(cur_data())) mean(logMeanBodyMass, na.rm = TRUE) else NA_real_,
        plot_value = mean(plot_value, na.rm = TRUE),
        Common.name = dplyr::first(Common.name),
        .groups = "drop"
      )

    if (nrow(pheno_sub) < 3) next

    if (order_rows_by == phenotype_name) {
      pheno_sub <- pheno_sub %>% arrange(desc(plot_value))
      show_row_dend <- FALSE
      cluster_rows <- FALSE
    } else if (order_rows_by == "Clade") {
      pruned_tree2 <- keep.tip(pruned_tree, pheno_sub$Zoonomia_name)
      hc2 <- phylo_to_hclust(pruned_tree2)
      tree_order <- hc2$labels[hc2$order]
      pheno_sub <- pheno_sub[match(tree_order, pheno_sub$Zoonomia_name), , drop = FALSE]
      show_row_dend <- TRUE
      cluster_rows <- hc2
    } else {
      stop("order_rows_by must be 'Clade' or exactly phenotype_name")
    }

    ordered_species <- pheno_sub$Zoonomia_name
    mat_final <- mat_raw[ordered_species, , drop = FALSE]
    pheno_final <- pheno_sub[match(rownames(mat_final), pheno_sub$Zoonomia_name), , drop = FALSE]

    mat_final[is.nan(mat_final)] <- 0
    if (nrow(mat_final) < 3) next

    clade_df <- left_join(
      data.frame(tip = rownames(mat_final), stringsAsFactors = FALSE),
      clade_table[, c("tip", "display_clade")],
      by = "tip"
    )
    clades <- clade_df$display_clade
    clades[is.na(clades) | clades == ""] <- "Other"
    clade_colors <- make_clade_colors(clades)

    left_ha <- rowAnnotation(
      Clade = clades,
      col = list(Clade = clade_colors),
      annotation_name_gp = gpar(fontsize = 7),
      width = unit(0.4, "cm"),
      annotation_legend_param = list(Clade = list(title = "Clade"))
    )

    right_ha <- make_right_annotation(pheno_final, phenotype_name)

    npos <- ncol(mat_final)
    col_labs <- rep("", npos)
    idx <- unique(round(seq(1, npos, length.out = 5)))
    col_labs[idx] <- as.character(idx - 1)

    ht <- Heatmap(
      mat_final,
      name = "Binding\nprobability",
      cluster_rows = cluster_rows,
      cluster_columns = FALSE,
      show_row_dend = show_row_dend,
      row_dend_side = "left",
      row_dend_width = unit(3, "cm"),
      left_annotation = left_ha,
      right_annotation = right_ha,
      column_title = glue("{motif} binding probability — {gene} promoter\n({nrow(mat_final)} mammals)"),
      column_title_gp = gpar(fontsize = 15),
      row_names_side = "right",
      row_names_gp = gpar(fontsize = 7),
      column_labels = col_labs,
      column_names_rot = 0,
      column_names_gp = gpar(fontsize = 15),
      col = colorRamp2(c(0, 1), c("#f7f7f7", "black")),
      use_raster = TRUE,
      raster_quality = 2
    )

    out_file <- glue("{outdir}/probNorm_{motif}_{gene}_phylo_heatmap.png")
    png(
      filename = out_file,
      width = 15,
      height = max(10, nrow(mat_final) * 0.13),
      units = "in",
      res = 300
    )
    draw(ht, merge_legend = TRUE)
    dev.off()

    cat(glue("Done: {gene} / {motif} ({nrow(mat_final)} species) -> {out_file}\n"))
  }
}
