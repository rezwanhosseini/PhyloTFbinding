# PhyloTFbinding
This repository is to make predictions for TF binding on a given set of sequence from different species and cluster them by their clade/phylogenetic distance, or a chosen phenotype

**1. get PWM hits**

inputs:
```
fasta file including orthologouse promoter sequences with headers named by the species (either VGP or Zoonomia names)
motif PWMs, could be from GIMME, HOCOMOCO, JASPAR, cisBP
```
outputs:
```
a matrix (species x position) containing the binding score at each position on each sequence, per motif.
```

**2.plot TFbinding Heatmap**

inputs:

```
the matrix from last step.
VGP_Species_withZoonomiaNames.txt to map/match vgp names with zoonomia names
zoonomia_241way_ncbi_taxonomy.csv to map each species to its clade
```

output:

```
the heatmap shown as an example.
```
