#setwd("/home/seh197/rezwan/research/Maria")
####################################################################################################################
# ============================================================
# Phylogenetic tree-ordered heatmap of TF binding probabilities
# ============================================================
#
# Approach:
# 1. Load the 241-way Zoonomia tree (ape::read.tree)
# 2. Map assembly codes (e.g. "aotNan1") -> tree tip names (e.g. "Aotus_nancymaae")
# 3. Deduplicate: keep first assembly per species
# 4. Prune tree to present species (ape::keep.tip)
# 5. Make ultrametric if needed (phytools::force.ultrametric)
# 6. Convert pruned tree -> hclust (ape::as.hclust.phylo)
# 7. Pass hclust to ComplexHeatmap cluster_rows -> left-side dendrogram
# 8. Add clade color bar (left annotation) + body mass bar (right annotation)

suppressPackageStartupMessages({
  library(ape)
  library(phytools)
  library(ComplexHeatmap)
  library(circlize)
  library(glue)
  library(stringr)
  library(dplyr)
  library(rhdf5)
  library(optparse)
})
print("check1")

# ── 1. Load tree ──────────────────────────────────────────────────────────────
tree_241 <- read.tree("tree/tree_241way_mammals_zoonomia.nh")
cat("Tree loaded:", length(tree_241$tip.label), "tips\n")
print("check2")

# ── 2. Assembly code -> tree tip name mapping ─────────────────────────────────
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

tip_lookup <- build_tip_lookup(tree_241)
tips_set   <- tree_241$tip.label

# Manual overrides for assembly codes that don't follow the standard 3+3 pattern
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

assembly_to_tip <- function(code) {
  # Try manual overrides (longest prefix first to avoid partial matches)
  prefixes <- names(manual_overrides)[order(nchar(names(manual_overrides)), decreasing = TRUE)]
  for (pfx in prefixes) {
    if (startsWith(code, pfx)) {
      tip <- manual_overrides[[pfx]]
      if (!is.na(tip) && tip %in% tips_set) return(tip)
      else return(NA_character_)
    }
  }
  # Auto-parse standard codes: aotNan1 -> (aot, nan)
  m <- regmatches(code, regexpr("^([a-z]{3})([A-Z][a-z]{2})", code))
  if (length(m) > 0) {
    key <- paste0(tolower(substr(m, 1, 3)), tolower(substr(m, 4, 6)))
    if (!is.null(tip_lookup[[key]])) return(tip_lookup[[key]])
  }
  # HL + 3+3: HLaciJub2 -> (aci, jub)
  m <- regmatches(code, regexpr("^HL([a-z]{3})([A-Z][a-z]{2})", code))
  if (length(m) > 0) {
    inner <- sub("^HL", "", m)
    key   <- paste0(tolower(substr(inner, 1, 3)), tolower(substr(inner, 4, 6)))
    if (!is.null(tip_lookup[[key]])) return(tip_lookup[[key]])
  }
  # HL + 3+3+3: HLcanLupFam7 -> (can, lup)
  m <- regmatches(code, regexpr("^HL([a-z]{3})([A-Z][a-z]{2})[A-Z][a-z]{2}", code))
  if (length(m) > 0) {
    inner <- sub("^HL", "", m)
    key   <- paste0(tolower(substr(inner, 1, 3)), tolower(substr(inner, 4, 6)))
    if (!is.null(tip_lookup[[key]])) return(tip_lookup[[key]])
  }
  return(NA_character_)
}

# ── 3. Clade color map ────────────────────────────────────────────────────────
clade_map <- c(
  Homo="Primates", Pan="Primates", Gorilla="Primates", Pongo="Primates",
  Nomascus="Primates", Symphalangus="Primates", Macaca="Primates",
  Papio="Primates", Mandrillus="Primates", Cercocebus="Primates",
  Cercopithecus="Primates", Erythrocebus="Primates", Chlorocebus="Primates",
  Rhinopithecus="Primates", Colobus="Primates", Piliocolobus="Primates",
  Aotus="Primates", Callithrix="Primates", Saimiri="Primates",
  Cebus="Primates", Alouatta="Primates", Ateles="Primates",
  Microcebus="Primates", Mirza="Primates", Lemur="Primates",
  Eulemur="Primates", Propithecus="Primates", Indri="Primates",
  Daubentonia="Primates", Cheirogaleus="Primates", Nycticebus="Primates",
  Otolemur="Primates", Nasalis="Primates", Pygathrix="Primates",
  Semnopithecus="Primates", Pithecia="Primates", Saguinus="Primates",
  Tupaia="Primates",
  Mus="Rodentia", Rattus="Rodentia", Peromyscus="Rodentia",
  Cricetulus="Rodentia", Mesocricetus="Rodentia", Microtus="Rodentia",
  Ondatra="Rodentia", Castor="Rodentia", Marmota="Rodentia",
  Ictidomys="Rodentia", Urocitellus="Rodentia", Cynomys="Rodentia",
  Sciurus="Rodentia", Heterocephalus="Rodentia", Fukomys="Rodentia",
  Cavia="Rodentia", Chinchilla="Rodentia", Octodon="Rodentia",
  Ctenodactylus="Rodentia", Jaculus="Rodentia", Nannospalax="Rodentia",
  Dipodomys="Rodentia", Perognathus="Rodentia", Aplodontia="Rodentia",
  Glis="Rodentia", Graphiurus="Rodentia", Hydrochoerus="Rodentia",
  Spermophilus="Rodentia", Xerus="Rodentia",
  Canis="Carnivora", Lycaon="Carnivora", Vulpes="Carnivora",
  Felis="Carnivora", Panthera="Carnivora", Acinonyx="Carnivora",
  Puma="Carnivora", Lynx="Carnivora", Neofelis="Carnivora",
  Ursus="Carnivora", Ailuropoda="Carnivora", Mustela="Carnivora",
  Neovison="Carnivora", Enhydra="Carnivora", Lutra="Carnivora",
  Zalophus="Carnivora", Leptonychotes="Carnivora", Neomonachus="Carnivora",
  Odobenus="Carnivora", Ailurus="Carnivora", Cryptoprocta="Carnivora",
  Hyaena="Carnivora", Mellivora="Carnivora", Mirounga="Carnivora",
  Paradoxurus="Carnivora", Pteronura="Carnivora", Spilogale="Carnivora",
  Suricata="Carnivora",
  Bos="Artiodactyla", Bison="Artiodactyla", Bubalus="Artiodactyla",
  Ovis="Artiodactyla", Capra="Artiodactyla", Oryx="Artiodactyla",
  Cervus="Artiodactyla", Odocoileus="Artiodactyla", Rangifer="Artiodactyla",
  Giraffa="Artiodactyla", Sus="Artiodactyla", Phacochoerus="Artiodactyla",
  Vicugna="Artiodactyla", Camelus="Artiodactyla", Tursiops="Artiodactyla",
  Orcinus="Artiodactyla", Physeter="Artiodactyla", Balaenoptera="Artiodactyla",
  Lipotes="Artiodactyla", Inia="Artiodactyla", Mesoplodon="Artiodactyla",
  Delphinapterus="Artiodactyla", Monodon="Artiodactyla",
  Ammotragus="Artiodactyla", Antilocapra="Artiodactyla",
  Beatragus="Artiodactyla", Elaphurus="Artiodactyla",
  Eschrichtius="Artiodactyla", Eubalaena="Artiodactyla",
  Hemitragus="Artiodactyla", Kogia="Artiodactyla", Moschus="Artiodactyla",
  Okapia="Artiodactyla", Phocoena="Artiodactyla", Saiga="Artiodactyla",
  Ziphius="Artiodactyla",
  Equus="Perissodactyla", Tapirus="Perissodactyla",
  Ceratotherium="Perissodactyla", Dicerorhinus="Perissodactyla",
  Pteropus="Chiroptera", Rhinolophus="Chiroptera", Myotis="Chiroptera",
  Eptesicus="Chiroptera", Miniopterus="Chiroptera", Desmodus="Chiroptera",
  Rousettus="Chiroptera", Molossus="Chiroptera", Tadarida="Chiroptera",
  Artibeus="Chiroptera", Carollia="Chiroptera", Craseonycteris="Chiroptera",
  Eidolon="Chiroptera", Hipposideros="Chiroptera", Macroglossus="Chiroptera",
  Megaderma="Chiroptera", Micronycteris="Chiroptera", Mormoops="Chiroptera",
  Noctilio="Chiroptera", Pteronotus="Chiroptera", Tonatia="Chiroptera",
  Sorex="Eulipotyphla", Condylura="Eulipotyphla", Erinaceus="Eulipotyphla",
  Solenodon="Eulipotyphla",
  Oryctolagus="Lagomorpha", Ochotona="Lagomorpha", Lepus="Lagomorpha",
  Loxodonta="Afrotheria", Trichechus="Afrotheria", Procavia="Afrotheria",
  Chrysochloris="Afrotheria", Echinops="Afrotheria", Tenrec="Afrotheria",
  Dasypus="Xenarthra", Choloepus="Xenarthra", Bradypus="Xenarthra",
  Monodelphis="Marsupial", Sarcophilus="Marsupial", Macropus="Marsupial",
  Ornithorhynchus="Monotreme",
  Manis="Pholidota"
)

clade_colors <- c(
  Primates      = "#E41A1C",
  Rodentia      = "#FF7F00",
  Carnivora     = "#4DAF4A",
  Artiodactyla  = "#377EB8",
  Perissodactyla= "#984EA3",
  Chiroptera    = "#A65628",
  Eulipotyphla  = "#F781BF",
  Lagomorpha    = "#999999",
  Afrotheria    = "#66C2A5",
  Xenarthra     = "#FC8D62",
  Marsupial     = "#8DA0CB",
  Monotreme     = "#E78AC3",
  Pholidota     = "#B15928",
  Other         = "#CCCCCC"
)

get_clade <- function(tip) {
  genus <- strsplit(tip, "_")[[1]][1]
  clade <- clade_map[genus]
  if (is.na(clade)) "Other" else clade
}

# ── 4. Tree -> hclust helper ──────────────────────────────────────────────────
phylo_to_hclust <- function(tree) {
    if (!is.ultrametric(tree, tol = 1e-6)) {
    tree <- force.ultrametric(tree, method = "extend")
  }
  as.hclust.phylo(tree)
}

# ── 5. Main plotting loop ─────────────────────────────────────────────────────
option_list <- list(
  make_option("--gene", type = "character", default ="",
              help = "name of the gene"),
  make_option("--motif", type = "character", default ="",
              help = "name of the motif"),
  make_option("--runid", type = "character", default ="",
              help = "runid"),
  make_option("--summary", type = "character", default = "",
              help = "Path to the summary file."),
  make_option("--h5", type = "character", default = "",
              help = "Path to the h5 file"),
  make_option("--orderby", type = "character", default = "clade",
              help = "whether to order species by clade or phenotype value. (options: 'phenotype', 'clade')"),
  make_option("--out", type = "character", default = "motifhit_plots",
              help = "Path to put the plots in.")
)
parser <- OptionParser(option_list = option_list)
args <- parse_args(parser)

ht_opt$message = FALSE

gene  <- args$gene
motif <- args$motif
run_id <- args$runid
summary <- args$summary
h5 <- args$h5
out <- args$out
orderby <- args$orderby

print("check3")

cons_pwm_summary <- read.csv(glue("conservation_pwm_analysis/{gene}/{motif}/{run_id}/pwm_summary.csv"), header=T) # cons_pwm_results_EyeSpecificGenes_relaxed_summary.csv
cons_pwm_file <- glue("conservation_pwm_analysis/{gene}/{motif}/{run_id}/pwm_results.h5") #cons_pwm_results_EyeSpecificGenes_relaxed.h5"


phenotype_name = 'VisionLoss_CC'
phenotype <- readRDS(glue("phenotype-values/{gsub('_CC', '', phenotype_name)}_phenotype.RDS"))
phenotype <- data.frame(Zoonomia_name = tolower(names(phenotype)),
                        visionloss = phenotype)

VGP_assemblies <- read.table("VGP_Species_withZoonomiaNames.txt", sep = "\t", header = T)
VGP_assemblies$VGP_name <- sapply(strsplit(VGP_assemblies$Directory.Name, "__"), `[`, 3)
phenotype <- inner_join(phenotype, VGP_assemblies[,c("Zoonomia_name", "VGP_name", "Common.name")], by = "Zoonomia_name")
print("check4")

#eye_specific <- read.table("MotifTSSvalidation/afconverge/eye-specific-genes.txt")

#permscores_puffin <- readRDS(glue("AFconverge_folder/AFconverge/PermScores_{phenotype_name}_puffin/PermScores_matrix_puffin_NAsreplacedwith0.RData")) # NAsreplacedwith0, withNAs
#permscores_puffin$promoter <- sub("_TSS.*", "", rownames(permscores_puffin))
# permscores_puffin <- permscores_puffin%>%
#   group_by(promoter) %>%
#   slice_max(abs(`FANTOM CAGE assay +`), n = 1, with_ties = FALSE) %>%
#   ungroup()
#top_promo = eye_specific[,1] #unlist(permscores_puffin[order(abs(permscores_puffin$FANTOM_CAGE_assay_fwd_original), decreasing = T)[1:10],"promoter"])

print(orderby)
if (orderby=='clade'){
  order_rows_by = "Clade"
}else{
  order_rows_by = phenotype_name 
}
print("Check5")
print(order_rows_by)

plot_dir = glue("{out}/plots_{phenotype_name}/pwm_hits_clustered_by_{order_rows_by}")

print(plot_dir)

dir.create(plot_dir, showWarnings = F, recursive = T)

for (gene in gene) { ## change to top_promot if ou want to make all the plots again!
  # gene = top_promo[1]
  gene_rows <- cons_pwm_summary[grep(gene, cons_pwm_summary$gene),]
  gene_data <- gene_rows[gene_rows$gene == gene, ]
  motifs    <- unique(gene_data$motif_id[
    gene_data$motif_id != "" #& gene_data$data_type == "pwm_prob"
  ])
  
  for (motif in motifs) {
    # motif=motifs[65]
    pwm_path <- gene_data[
      gene_data$motif_id == motif, # & gene_data$data_type == "pwm_prob",
      "hdf5_path"
    ]
    if (length(pwm_path) == 0 | pwm_path=="") next
    
    # ── Load matrix (n_species x 500) and species IDs ──
    mat_raw     <- t(h5read(cons_pwm_file, pwm_path))
    species_ids <- as.character(h5read(
      cons_pwm_file, gsub("/scores", "/species", pwm_path)
    ))
    
    # Strip "_[ENSG...|GENE]" suffix to get assembly code
    assembly_codes <- str_remove(species_ids, "_\\[.*$")
    
    if (length(intersect(assembly_codes, tree_241$tip.label))>0){
      tip_names <- assembly_codes
    }else{
      # ── Map assembly codes -> tree tip names ──
      tip_names <- sapply(assembly_codes, assembly_to_tip)
    }
    
    # Keep only rows that map to tree tips
    keep      <- !is.na(tip_names)
    mat_raw   <- mat_raw[keep, , drop = FALSE]
    tip_names <- tip_names[keep]
    
    if (nrow(mat_raw) < 3) next
    
    # ── Deduplicate: keep first assembly per species ──
    dup_mask  <- duplicated(tip_names)
    mat_raw   <- mat_raw[!dup_mask, , drop = FALSE]
    tip_names <- tip_names[!dup_mask]
    rownames(mat_raw) <- tolower(tip_names)
    
    # ── Prune tree to present species ──
    pruned_tree <- keep.tip(tree_241, tip_names)
    pruned_tree$tip.label <- tolower(pruned_tree$tip.label)
    
    # ── Build hclust from pruned tree ──
    hc <- phylo_to_hclust(pruned_tree)
    
    # ── Join with phenotype (keeps only species with phenotype data) ──
    tip_df    <- data.frame(Zoonomia_name = tolower(tip_names), stringsAsFactors = FALSE)
    pheno_sub <- inner_join(tip_df, phenotype, by = "Zoonomia_name")%>%
      group_by(Zoonomia_name) %>%
      summarise(
        visionloss = mean(visionloss),
        Common.name = dplyr::first(Common.name),
        .groups = "drop"
      )
    if (order_rows_by==phenotype_name){
      pheno_sub <- pheno_sub%>%
        arrange(desc(visionloss))
      show_row_dend = F
      cluster_rows = F
    }
    
    ordered_species <- pheno_sub$Zoonomia_name
    mat_final <- mat_raw[tolower(ordered_species), , drop = FALSE]
    pheno_final <- pheno_sub[match(tolower(rownames(mat_final)), pheno_sub$Zoonomia_name), ]
    
    # keep_pheno <- tolower(tip_names) %in% pheno_sub$Zoonomia_name
    # mat_final  <- mat_raw[keep_pheno, , drop = FALSE]
    # pheno_final <- pheno_sub[match(tolower(rownames(mat_final)), pheno_sub$Zoonomia_name), ]
    
    mat_final[is.nan(mat_final)] <- 0
    
    if (nrow(mat_final) < 3) next
    
    # ── Rebuild hclust for phenotype-filtered subset ──
    pruned_tree2 <- keep.tip(pruned_tree, rownames(mat_final))
    hc2          <- phylo_to_hclust(pruned_tree2)
    
    if (order_rows_by=="Clade"){
      tree_order <- hc2$labels[hc2$order]
      mat_final <- mat_final[tree_order, , drop = FALSE]
      pheno_final <- pheno_final[match(tree_order, pheno_final$Zoonomia_name), , drop = FALSE]
      show_row_dend = T
      cluster_rows = hc2
    }else{
      show_row_dend = F
      cluster_rows = F
    }
    
    
    # ── Clade annotation (left side) ──
    clades      <- sapply(tools::toTitleCase(rownames(mat_final)), get_clade)
    clade_cols  <- clade_colors[clades]
    left_ha <- rowAnnotation(
      Clade = clades,
      col = list(Clade = clade_colors),
      annotation_name_gp = gpar(fontsize = 7),
      width = unit(0.4, "cm"),
      annotation_legend_param = list(
        Clade = list(title = "Clade")
      )
    )
    
    # ── Bphenotype annotation (right side) ──
    if (phenotype_name=="MeanBodyMass"){
      right_ha <- rowAnnotation(
        logMeanBodyMass = anno_barplot(
          pheno_final$logMeanBodyMass,
          border = FALSE,
          gp     = gpar(color = "black", fill = "grey80", col = NA),
          axis_param = list(side = "bottom")
        ),
        annotation_name_gp = gpar(fontsize = 7),
        width = unit(3, "cm")
      )
      
    }else if (grepl("VisionLoss", phenotype_name)){
      right_ha <- rowAnnotation(
        visionloss = anno_barplot(
          pheno_final$visionloss,
          border = FALSE,
          gp     = gpar(color = "black", fill = "grey80", col = NA),
          axis_param = list(side = "bottom")
        ),
        annotation_name_gp = gpar(fontsize = 7),
        width = unit(3, "cm")
      )
    }
    
    
    # ── Column labels (sparse: only 0, 125, 250, 375, 499) ──
    col_labs <- rep("", 500)
    col_labs[c(1, 126, 251, 376, 500)] <- c("0", "125", "250", "375", "499")
    
    # ── Heatmap ──
    ht <- Heatmap(
      mat_final,
      name              = "Binding\nprobability",
      # Phylogenetic dendrogram on rows
      cluster_rows      = cluster_rows,          # hclust derived from pruned tree
      cluster_columns   = FALSE,
      show_row_dend     = show_row_dend,
      row_dend_side     = "left",
      row_dend_width    = unit(3, "cm"),
      # Annotations
      left_annotation   = left_ha,
      right_annotation  = right_ha,
      # Labels
      column_title      = glue("{motif} binding probability — {gene} promoter\n({nrow(mat_final)} mammals, phylogenetic dendrogram)"),
      column_title_gp   = gpar(fontsize = 15),
      row_names_side    = "right",
      row_names_gp      = gpar(fontsize = 7),
      column_labels     = col_labs,
      column_names_rot  = 0,
      column_names_gp   = gpar(fontsize = 15),
      # Color scale
      col               = colorRamp2(c(0, 1), c("#f7f7f7", "black")),
      # Performance
      use_raster        = TRUE,
      raster_quality    = 2
    )
    # draw(ht, merge_legend = TRUE)
    # ── Save ──
    out_file <- glue("{out}/plots_{phenotype_name}/pwm_hits_clustered_by_{order_rows_by}/probNorm_{motif}_{gene}_phylo_heatmap.png")
    png(
      filename = out_file,
      width    = 15,
      height   = max(10, nrow(mat_final) * 0.13),
      units    = "in",
      res      = 300
    )
    draw(ht, merge_legend = TRUE)
    dev.off()
    
    cat(glue("Done: {gene} / {motif} ({nrow(mat_final)} species) -> {out_file}\n"))
  }
}
################################################################################################################################################

