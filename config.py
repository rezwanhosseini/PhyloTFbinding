from pathlib import Path

BASE_DIR = Path("/home/seh197/rezwan/research/Maria/phyloTFbinding/")  # change this
SCRIPT1_PATH = BASE_DIR / "MotifHit_loc.py"
PFM_PATH = BASE_DIR / "gimme.vertebrate.v5.0.pfm"

FASTA_DIR = Path("/home/seh197/rezwan/research/Maria/AFconverge_folder/promoterSeqs_byGene_combined/") # these should be loaded from a database: cluster, onedrive or sth for now: /home/seh197/rezwan/research/Maria/AFconverge_folder/promoterSeqs_byGene_combined/
#PAIRS_DIR = BASE_DIR / "AFconverge_folder" / "gene_motif_pairs" # this should be made or the pair given directly to the code
OUTPUT_BASE_DIR = BASE_DIR / "conservation_pwm_analysis"

SCRIPT2_PATH = BASE_DIR / "phylo_heatmap_visionloss.R"
