#!/usr/bin/env python
import os
from torch.nn import functional as F
import pandas as pd
from util import SegmentDataBed, SegmentDataSeq, MEME_probNorm, MEME_FABIAN, MCspline_fitting, mc_spline, kmers
import torch
import numpy as np
from enum import Enum
import typer
import pickle
import time
import resource
import matplotlib.pyplot as plt
from pysam import FastaFile

#import pyarrow.feather as feather


app = typer.Typer()

torch.backends.cudnn.deterministic = True

def write_output_motif_features(filename, mat, names, index = None):
    if index is None:
        df = pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()])#.to_csv(filename, sep = "\t", float_format = '%.3f', index = False)
        print("writing to csv with no index...")
        df.to_csv(filename, sep = "\t", float_format = '%.3f', index = False)
        #feather.write_feather(pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()]), f"{filename}.feather")
        print("writing to hdf5 with no index...")
        df.to_hdf(filename+'.h5', key='df', mode='w', format='table', complib='zlib', complevel=9)
    else:
        df = pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()], index=index)#.to_csv(filename, sep = "\t", float_format = '%.3f', index = True)
        print("writing to csv with index...")
        df.to_csv(filename, sep = "\t", float_format = '%.3f', index = True)
        #feather.write_feather(pd.DataFrame(mat, columns = names + ["GC_ratio", "Masked_ratio"] + [i+"_pattern" for i in kmers()], index=index), f"{filename}.feather")
        print("writing to hdf5 with index...")
        df.to_hdf(filename+'.h5', key='df', mode='w', format='fixed', complib='zlib', complevel=9)

class mode_type(str, Enum):
    max = "max"
    average = "average"

class kernel_type(str, Enum):
    TFFM = "TFFM"
    PWM = "PWM"
    
class norm_type(str, Enum):
    iid = "iid"
    motif_based = "motif_based"
    mixture = "mixture"
    from_mono = "from_mono"


class score_type(str, Enum):
    FABIAN = "FABIAN"
    probNorm = "probNorm"
    NONE = "NONE"
    
class nucleotide_type(str, Enum):
    mono = "mono"
    di = "di"
    
@app.command()
def variantdiff(genome: str = typer.Option(None, help="fasta file for the genome"),
                motif_file: str = typer.Option(..., "--motif", help="meme file for the motifs"),
                seqs: str = typer.Option(..., help="path to sequences (one folder per gene containing the orthologous sequences in fasta format), or a bed file of coordinates to extract the sequence from the given genome."), 
                diff_score: score_type = typer.Option(score_type.NONE, "--method", help="how to calculate the diff score (FABIAN/probNorm/NONE)"),
                max_scale: bool = typer.Option(False, "--MaxScale", help="Apply max transformation or not"),
                nucleotide: nucleotide_type = typer.Option(nucleotide_type.mono, "--nuc", help="length of the nucleotides in the motifs (mono/di)"),
                normalization_file: str = typer.Option(None, "--norm", help="file including normalization params. should be consistent with the transform option"),
                up: int = typer.Option(0, help="add upstream"),
                mode: mode_type = typer.Option(mode_type.max, help="Operation mode for the pooling layer (max/average)"),                 
                batch: int = typer.Option(128, help="batch size"),
                out_file: str = typer.Option(..., "--out", help="output directory"),
                window:int = typer.Option(500, "--window", help="window size"), # change this in a way that if it gets a value use that value and if not the default will be kernel size (instead of setting 0 as the default, make it none or not receiving an input as the default)
                kernel: kernel_type = typer.Option(kernel_type.PWM, help="Choose between PWM (4 dimensional) or TFFM (16 dimensional) (default = PWM)."),
                bin: int = typer.Option(1, help="number of bins")
                ):
    
    s = time.time()

    kernel = kernel.value
    if kernel == "PWM":
        if diff_score == "FABIAN":
            motif = MEME_FABIAN()
            print(f"Reading the motifs from {motif_file}")
            kernels, kernel_mask, kernel_norms = motif.parse(motif_file, nuc=nucleotide)
        else:
            if normalization_file is None:
                motif = MEME_probNorm()
                kernels, kernel_mask = motif.parse(motif_file, nuc=nucleotide, transform=max_scale)
                if diff_score=="probNorm":
                    print("normalization params not given. normalization process running...")
                    spline_list = MCspline_fitting(kernels, nuc=nucleotide)                    
            else:
                if "max_scaled" in normalization_file:
                    max_scale=True
                else:
                    max_scale=False
                motif = MEME_probNorm()
                kernels, kernel_mask = motif.parse(motif_file, nuc=nucleotide, transform=max_scale)
                with open(normalization_file, "rb") as fp:
                    spline_list = pickle.load(fp)

    

    if seqs.endswith(".bed"):
        print("positions given.")
        if genome==None:
            raise ValueError("genome is required to extract the sequence. use --genome argument.")
        else:
            segments = SegmentDataBed(seqs, batch, genome, int((window+up)/2), up, dinucleotide=(nucleotide=="di"))
    if os.path.isdir(seqs):
        print("sequences given.")
        segments = SegmentDataSeq(seqs, batch, int((window+up)/2), up, dinucleotide=(nucleotide=="di"))
    out = np.empty((segments.n, bin*motif.nmotifs+segments.additional)) # segments.additional is for the stats columns
    
    print(f"Batch size: {batch}")
    print("Calculating convolutions")
    for i in range(len(segments)):
        print(f"Batch {i+1}:")
        i1, i2 = i*batch, (i+1)*batch
        if i2 >= segments.n: i2 = segments.n
        mat, out[i1:i2, bin*motif.nmotifs:] = segments[i]

        tmp = F.conv1d(mat, kernels)

        if diff_score == "NONE":
            tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
            tmp = F.max_pool1d(tmp, tmp.shape[2])
            tmp = np.squeeze(tmp).numpy()

        if diff_score == "probNorm":
            if mode == "average":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
                tmp = mc_spline(tmp, spline_list)
                tmp = np.nan_to_num(tmp, nan=0)           
                tmp = F.avg_pool1d(torch.tensor(tmp), tmp.shape[2])
            if mode == "max":
                tmp = tmp.view(tmp.shape[0], tmp.shape[1]//2, tmp.shape[2]*2)
                tmp = np.nan_to_num(tmp, nan=0)
                tmp = F.max_pool1d(torch.tensor(tmp), tmp.shape[2])
                tmp = mc_spline(tmp, spline_list)
            #tmp = np.squeeze(tmp).numpy()

        if diff_score == "FABIAN":
            tmp = F.max_pool1d(tmp, tmp.shape[2]).numpy()
            tmp = np.max(tmp.reshape(tmp.shape[0],-1,2), axis=2) # separates the convolutions from the original kernel and the reverse complement kernel into two differetn columns AND then keeps the maximum between those two

        out[i1:i2, :bin*motif.nmotifs] = tmp.reshape(tmp.shape[0], tmp.shape[1]*tmp.shape[2])

    
    names = motif.names
    new_names =[]
    for i in range(bin*len(names)):
        if i%bin>int(bin/2): new_names.append(f'{names[int(i/bin)]} - right - {i%bin}'); #print(f'new_names{i} = names {int(i/args.bin)} which is "{names[int(i/args.bin)]}" - right')
        if i%bin==int(bin/2): new_names.append(f'{names[int(i/bin)]} - middle'); #print(f'new_names{i} = names {int(i/args.bin)} which is "{names[int(i/args.bin)]}" - middle')
        if i%bin<int(bin/2): new_names.append(f'{names[int(i/bin)]} - left - {i%bin}'); #print(f'new_names{i} = names {int(i/args.bin)} which is "{names[int(i/args.bin)]}" - left')

    write_output_motif_features(out_file+f"_{nucleotide}_{diff_score}_{mode}_{window}", out, new_names, segments.names())
    

    
    e = time.time()
    print(f"real runtime for {out_file}_{nucleotide}_{diff_score}_{mode}_{window} = ", e-s)


if __name__ == "__main__":
    app()
