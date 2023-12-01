'''
┌─┐  ┌─┐  ┌┬┐  ┬  ┌┐┌  ┬
│ ┬  ├┤   │││  │  │││  │
└─┘  └─┘  ┴ ┴  ┴  ┘└┘  ┴ annot
---------------------------------------------------
-*- coding: utf-8 -*-                              |
title            : geminiannot.py                  |
description      : gemini annotation functions     |
author           : dooguypapua                     |
lastmodification : 20210713                        |
version          : 0.1                             |
python_version   : 3.8.5                           |
---------------------------------------------------
'''

import os
import sys
import shutil
import geminiset
import copy
import time
import pandas
import pickle
import numpy as np
import torch
import warnings
import glob
from typing import Tuple
from pathlib import Path
from yaspin import yaspin
from yaspin.spinners import Spinners
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from scipy.special import expit, logit
from geminini import printcolor, title, read_file, dump_json, exit_gemini, launch_threads, path_converter
from geminini import fct_checker, get_input_files, get_gemini_path, get_sys_info, cat_lstfiles, reverse_complement
from geminini import get_ORF_info, tokenize_aa_seq, kmerize, predict, predict_tis, analyze_overlap, tensor_to_seq
from geminiparse import make_fasta_dict, make_gbk_from_fasta
from geminiexternal import get_best_solution, export_defense_finder_genes, export_defense_finder_systems
warnings.filterwarnings('ignore')


'''
-------------------------------------------------------------------------------------------------------
                                     SYNTAXIC & TRANSLATION FUNCTIONS
-------------------------------------------------------------------------------------------------------
'''


@fct_checker
def phanotate(pathIN: str, pathOUT: str, minLen: int = 0, fromPhageDb: bool = False, ext: str = ".fna") -> Tuple[str, str, int, bool, str]:
    '''
     -----------------------------------------------------------
    |             PHANOTATE PHAGE SYNTAXIC ANNOTATION           |
     -----------------------------------------------------------
    |    Phanotate syntaxic annotation from phage genome files  |
     -----------------------------------------------------------
    | Avoid to double genome to consider circular last ORF      |
    | because it generate false start.                          |
     -----------------------------------------------------------
    |PARAMETERS                                                 |
    |    pathIN     : path of input files or folder (required)  |
    |    pathOUT    : path of output files (required)           |
    |    minLen     : minimum ORF length (default=0)            |
    |    fromPhageDb: bool if call from phageBb (default=False) |
    |    ext        : extension of input files (default=.fna)   |
     -----------------------------------------------------------
    | TOOLS: phanotate                                          |
     -----------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "phanotate", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: phanotate]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'phanotate' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['phanotate'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    pathTMP = geminiset.pathTMP
    if not os.path.isdir(pathTMP):
        pathTMP = "/tmp"
    spinner = yaspin(Spinners.aesthetic, text="♊ Phanotate", side="right")
    spinner.start()
    title("Phanotate", None)
    dicoThread = {}
    lstNewFile = []
    for pathFNA in lstFiles:
        file = os.path.basename(pathFNA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pathFFN = pathOUT+"/"+orgName+".ffn"
        if not os.path.isfile(pathFFN) or os.path.getsize(pathFFN) == 0:
            lstNewFile.append(pathFNA)
            # Construct phanotate threads
            cmdPhanotate = dicoGeminiPath['TOOLS']['phanotate']+" --outfmt fasta -o "+pathFFN+" "+pathFNA
            dicoThread[orgName] = {"cmd": cmdPhanotate, "returnstatut": None, "returnlines": []}
    # Launch threads
    if len(dicoThread) > 0:
        launch_threads(dicoThread, "phanotate", cpu, pathTMP, spinner)
    spinner.stop()
    printcolor("♊ Phanotate"+"\n")
    # Reformat phanotate FFN
    printcolor("♊ Reformat"+"\n")
    if fromPhageDb is True:
        pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=75, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    else:
        pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFNA in lstNewFile:
        file = os.path.basename(pathFNA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        if "GCA" in file:
            ltPrefix = "GCA"+file.replace(ext, "").split("_")[-1].upper()+"_p"
        else:
            ltPrefix = orgName.replace("Vibrio_phage_", "VP").replace("-", "_").upper()+"_p"
        pathFFN = pathOUT+"/"+orgName+".ffn"
        if fromPhageDb is True:
            if "GCA" in file:
                pbar.set_description_str("GCA_"+file.replace(ext, "").split("_")[-1])
            else:
                pbar.set_description_str(file.replace(ext, "")[-15:])
        else:
            pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        # Retrieve genome sequence
        dicoFNA = make_fasta_dict(pathFNA)
        genomeSeq = list(dicoFNA.values())[0]
        # Reformat phanotate FFN
        dicoFFN = make_fasta_dict(pathFFN)
        FFN = open(pathFFN, 'w')
        cpt = 1
        for key in dicoFFN:
            if len(dicoFFN[key])/3 >= minLen:
                # Determine the frame
                if dicoFFN[key] in genomeSeq:
                    frame = "1"
                if reverse_complement(dicoFFN[key]) in genomeSeq:
                    frame = "-1"
                # Write gene sequence
                newHeader = ltPrefix+str(cpt).zfill(4)+"|"+frame
                FFN.write(">"+newHeader+" ["+orgName+"]\n"+dicoFFN[key]+"\n")
                cpt += 1
        FFN.close()
        pbar.update(1)
        title("Reformat", pbar)
    pbar.close()


@fct_checker
def balrog(pathIN: str, pathOUT: str, topology: str, division: str, taxID: int = 0, minLen: int = 50, boolMmseqs: bool = True, ext: str = ".fna") -> Tuple[str, str, str, str, int, int, bool, str]:
    '''
     -----------------------------------------------------------
    |                 BALROG SYNTAXIC ANNOTATION                |
     -----------------------------------------------------------
    |                 Balrog syntaxic annotation                |
    |  Copyright (c) 2020 Markus J. Sommer & Steven L. Salzberg |
     -----------------------------------------------------------
    |PARAMETERS                                                 |
    |    pathIN     : path of input files or folder (required)  |
    |    pathOUT    : path of output files (required)           |
    |    topology   : topology [linear or circular] (required)  |
    |    division   : division [BCT or PHG] (required)          |
    |    taxID      : genbank taxonomy ID (default=0)           |
    |    minLen     : minimum ORF length (default=50)           |
    |    boolMmseqs : filter gene with mmseqs2 (default=True)   |
    |    ext        : extension of input files (default=.fna)   |
     -----------------------------------------------------------
    | TOOLS: mmseqs                                             |
     -----------------------------------------------------------
    |    mmseqs index must be done with system mmseqs version   |
    |taxonomyID: Caudo=28883, Myo=10662, Podo=10744, Sipho=10699|
    |            V.chagasii=170679, V.crassostreae=246167       |
     -----------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "balrog", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: balrog]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    if topology not in ["linear", "circular"]:
        printcolor("[ERROR: make_gbk_from_fasta]\tTopology must be 'linear' or 'circular'\n", 1, "212;64;89", "None", True)
        exit_gemini()
    if division not in ["BCT", "PHG"]:
        printcolor("[ERROR: make_gbk_from_fasta]\nDivision must be 'BCT' for bacteria or 'PHG' for phage\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'mmseqs' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['mmseqs'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    pathTMP = geminiset.pathTMP
    if not os.path.isdir(pathTMP):
        pathTMP = "/tmp"
    pathLOG = pathOUT+"/balrog.log"
    if os.path.isfile(pathLOG):
        os.remove(pathLOG)
    Path(pathLOG).touch()
    # ***** Balrog required variables ***** #
    max_gene_overlap = 60  # Maximum ORF (open reading frame) overlap length in nucleotides.
    protein_kmer_filter = True  # Use kmer prefilter to increase gene sensitivity. May not play nice with very high GC genomes.
    translation_table = 11
    # Maximum number of forward connections in the directed acyclic graph used to find a set of coherent genes in each genome.
    # Higher values will slow execution time and increase memory usage, but may slightly increase performance (Recommended range ~30-50)
    max_forward_connections = 50
    # Batch size for the temporal convolutional network used to score genes.
    gene_batch_size = 200
    TIS_batch_size = 1000
    # All following are internal parameters. Change at your own risk.
    weight_gene_prob = 0.9746869839852076
    weight_TIS_prob = 0.25380288790532707
    score_threshold = 0.47256101519707244
    weight_ATG = 0.84249804151264
    weight_GTG = 0.7083689705744909
    weight_TTG = 0.7512400826652517
    unidirectional_penalty_per_base = 3.895921717182765  # 3' 5' overlap
    convergent_penalty_per_base = 4.603432608883688  # 3' 3' overlap
    divergent_penalty_per_base = 3.3830814940689975  # 5' 5' overlap
    k_seengene = 10
    multimer_threshold = 2
    nuc_encode = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 0, "M": 0, "R": 0, "Y": 0, "W": 0, "K": 0}
    start_enc = {"ATG": 0, "GTG": 1, "TTG": 2}
    # Required paths
    genexa_kmer_path = dicoGeminiPath['DATABASES']['balrog_data']+"/kmer_filter/10mer_thresh2_minusARF_all.pkl"
    # genexa_fasta_path = dicoGeminiPath['balrog_data']+"/protein_filter/genexa_genes.fasta"
    genexa_DB_path = dicoGeminiPath['DATABASES']['balrog_data']+"/protein_filter/genexaDB"
    swissprot_DB_path = dicoGeminiPath['DATABASES']['balrog_data']+"/protein_filter/swissprotDB"
    phanns_DB_path = dicoGeminiPath['DATABASES']['balrog_data']+"/protein_filter/phannsDB"
    pvogs_DB_path = dicoGeminiPath['DATABASES']['balrog_data']+"/protein_filter/pvogsDB"
    # ***** LOAD MODEL ***** #
    spinner = yaspin(Spinners.aesthetic, text="♊ Load models", side="right")
    spinner.start()
    title("Loading models", None)
    model = torch.hub.load(dicoGeminiPath['DATABASES']['balrog_data'], "geneTCN", source='local', force_reload=True)
    model_tis = torch.hub.load(dicoGeminiPath['DATABASES']['balrog_data'], "tisTCN", source='local', force_reload=True)
    time.sleep(0.5)
    spinner.stop()
    printcolor("♊ Load models"+"\n")
    # ***** LOAD KMER FILTER ***** #
    spinner = yaspin(Spinners.aesthetic, text="♊ Load kmer filters", side="right")
    spinner.start()
    title("Loading Kmers", None)
    with open(genexa_kmer_path, "rb") as f:
        aa_kmer_set = pickle.load(f)
    spinner.stop()
    printcolor("♊ Load kmer filters"+"\n")
    # ***** FIND GENES ***** #
    dicoContigToOrg = {}
    contig_name_list = []
    contig_length_list = []
    contig_seq_list = []
    contig_gene_coord_list = []
    contig_gene_strand_list = []
    printcolor("♊ Find genes"+"\n")
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=75+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFile in lstFiles:
        orgName = os.path.basename(pathFile).replace(ext, "").replace("."+ext, "")
        pathFFN = pathOUT+"/"+orgName+".ffn"
        if os.path.isfile(pathFFN):
            os.remove(pathFFN)
        pathGFF = pathOUT+"/"+orgName+".gff"
        if os.path.isfile(pathGFF):
            os.remove(pathGFF)
        pbar.set_description_str(orgName.rjust(maxpathSize-len(orgName)))
        # read genome sequence
        seq_list = []
        contig_name_sublist = []
        contig_length_sublist = []
        with open(pathFile, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_list.append(record.seq)
                contig_name_sublist.append(record.id)
                dicoContigToOrg[record.id] = orgName
                contig_length_sublist.append(len(record.seq))
        contig_name_list.append(contig_name_sublist)
        contig_length_list.append(contig_length_sublist)
        contig_seq_list.append(seq_list)
        # get sequences and coordinates of ORFs
        pbar.set_description_str(orgName+" > Predict ORFs".rjust(maxpathSize-len(orgName)-len(" > Predict ORFs")))
        ORF_seq_list, ORF_nucseq_list, ORF_coord_list = get_ORF_info(seq_list, minLen, translation_table)
        # combine ORFs to submit to GPU in batches
        ORF_seq_combine = []
        for i, contig in enumerate(ORF_seq_list):
            for j, frame in enumerate(contig):
                for k, coord in enumerate(frame):
                    ORF_seq_combine.append(coord)
        # encode amino acids as integers
        pbar.set_description_str(orgName+" > Encode ORFs".rjust(maxpathSize-len(orgName)-len(" > Encode ORFs")))
        ORF_seq_enc = [tokenize_aa_seq(x) for x in ORF_seq_combine]
        # seengene check
        if protein_kmer_filter:
            pbar.set_description_str(orgName+" > Filter kmers".rjust(maxpathSize-len(orgName)-len(" > Filter kmers")))
            seengene = []
            for s in ORF_seq_enc:
                kmerset = kmerize(s, k_seengene)
                s = [x in aa_kmer_set for x in kmerset]
                seen = np.sum(s) >= multimer_threshold
                seengene.append(seen)
        # score
        pbar.set_description_str(orgName+" > Score ORFs".rjust(maxpathSize-len(orgName)-len(" > Score ORFs")))
        # sort by length to minimize impact of batch padding
        ORF_lengths = np.asarray([len(x) for x in ORF_seq_enc])
        length_idx = np.argsort(ORF_lengths)
        ORF_seq_sorted = [ORF_seq_enc[i] for i in length_idx]
        # pad to allow creation of batch matrix
        prob_list = []
        for i in range(0, len(ORF_seq_sorted), gene_batch_size):
            batch = ORF_seq_sorted[i: i+gene_batch_size]
            seq_lengths = torch.LongTensor(list(map(len, batch)))
            seq_tensor = torch.zeros((len(batch), seq_lengths.max())).long()
            for idx, (seq, seqlen) in enumerate(zip(batch, seq_lengths)):
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
            pred_all = predict(seq_tensor, model)
            pred = []
            for j, length in enumerate(seq_lengths):
                subseq = pred_all[j, 0, 0: int(length)]
                predprob = float(expit(torch.mean(logit(subseq))))
                pred.append(predprob)
            prob_list.extend(pred)
        prob_arr = np.asarray(prob_list, dtype=float)
        # unsort
        unsort_idx = np.argsort(length_idx)
        ORF_prob = prob_arr[unsort_idx]
        # recombine ORFs and fill coord matrix with scores
        idx = 0
        ORF_gene_score = copy.deepcopy(ORF_coord_list)
        for i, contig in enumerate(ORF_gene_score):
            for j, frame in enumerate(contig):
                for k, coord in enumerate(frame):
                    ORF_gene_score[i][j][k] = float(ORF_prob[idx])
                    idx += 1
        # create strand information
        ORF_strand_flat = []
        for i, seq in enumerate(ORF_seq_list):
            if not any(seq):
                ORF_strand_flat.append([])
                continue
            n_forward = len(seq[0]) + len(seq[1]) + len(seq[2])
            n_reverse = len(seq[3]) + len(seq[4]) + len(seq[5])
            ORF_allframe_strand = [1]*n_forward + [-1]*n_reverse
            ORF_strand_flat.append(ORF_allframe_strand)
        # flatten coords within contigs
        ORF_coord_flat = [[item for sublist in x for item in sublist] for x in ORF_coord_list]
        # get ORF lengths
        # ORF_length_flat = [[coords[1]-coords[0] for coords in x] for x in ORF_coord_flat]
        # extract nucleotide sequence surrounding potential start codons
        pbar.set_description_str(orgName+" > Score TISs".rjust(maxpathSize-len(orgName)-len(" > Score TISs")))
        ORF_TIS_seq = copy.deepcopy(ORF_coord_list)
        ORF_start_codon = copy.deepcopy(ORF_coord_list)
        for i, contig in enumerate(ORF_TIS_seq):
            n = 0  # count to index into flat structure # TODO make sure this works as expected
            nucseq = ORF_nucseq_list[i][0]  # easier to use coords relative to single nucseq
            nucseq_c = ORF_nucseq_list[i][3][::-1]
            contig_nuclength = len(nucseq)
            for j, frame in enumerate(contig):
                for k, temp in enumerate(frame):
                    if any(temp):
                        coords = ORF_coord_list[i][j][k]
                        strand = ORF_strand_flat[i][n]
                        n += 1
                        if strand == 1:
                            fiveprime = coords[0]
                            if fiveprime >= 16 + 3:  # NOTE 16 HARD CODED HERE
                                downstream = nucseq[fiveprime: fiveprime + 16]
                                upstream = nucseq[fiveprime - 16 - 3: fiveprime - 3]
                                start_codon = start_enc[nucseq[fiveprime-3: fiveprime]]
                                TIS_seq = torch.tensor([nuc_encode[c] for c in (upstream + downstream)[::-1]], dtype=int)  # model scores 3' to 5' direction
                            else:
                                TIS_seq = -1  # deal with gene fragments later
                                start_codon = 2
                            ORF_TIS_seq[i][j][k] = TIS_seq
                            ORF_start_codon[i][j][k] = start_codon
                        else:
                            fiveprime = coords[1]
                            if contig_nuclength - fiveprime + 3 >= 16 + 3:  # NOTE 16 HARD CODED HERE
                                downstream = nucseq_c[fiveprime - 16: fiveprime][::-1]
                                upstream = nucseq_c[fiveprime + 3: fiveprime + 3 + 16][::-1]
                                start_codon = start_enc[nucseq_c[fiveprime: fiveprime + 3][::-1]]
                                TIS_seq = torch.tensor([nuc_encode[c] for c in (upstream + downstream)[::-1]], dtype=int)  # model scores 3' to 5' direction
                            else:
                                TIS_seq = -1  # deal with gene fragments later
                                start_codon = 2
                            ORF_TIS_seq[i][j][k] = TIS_seq
                            ORF_start_codon[i][j][k] = start_codon
        # flatten TIS for batching
        ORF_TIS_prob = copy.deepcopy(ORF_TIS_seq)
        ORF_TIS_seq_flat = []
        ORF_TIS_seq_idx = []
        for i, contig in enumerate(ORF_TIS_seq):
            for j, frame in enumerate(contig):
                for k, seq in enumerate(frame):
                    if type(seq) == int:  # fragment
                        ORF_TIS_prob[i][j][k] = 0.5  # HOW BEST TO DEAL WITH FRAGMENT TIS?
                    elif len(seq) != 32:
                        ORF_TIS_prob[i][j][k] = 0.5
                    else:
                        ORF_TIS_seq_flat.append(seq)
                        ORF_TIS_seq_idx.append((i, j, k))
        # batch score TIS
        TIS_prob_list = []
        for i in range(0, len(ORF_TIS_seq_flat), TIS_batch_size):
            batch = ORF_TIS_seq_flat[i: i+TIS_batch_size]
            TIS_stacked = torch.stack(batch)
            pred = predict_tis(TIS_stacked, model_tis)
            TIS_prob_list.extend(pred)
        y_pred_TIS = np.asarray(TIS_prob_list, dtype=float)
        # reindex batched scores
        for i, prob in enumerate(y_pred_TIS):
            idx = ORF_TIS_seq_idx[i]
            ORF_TIS_prob[idx[0]][idx[1]][idx[2]] = float(prob)
        # combine all info into single score for each ORF
        if protein_kmer_filter:
            ORF_score_flat = []
            for i, contig in enumerate(ORF_gene_score):
                if not any(contig):
                    ORF_score_flat.append([])
                    continue
                temp = []
                seengene_idx = 0
                for j, frame in enumerate(contig):
                    for k, geneprob in enumerate(frame):
                        length = ORF_coord_list[i][j][k][1] - ORF_coord_list[i][j][k][0] + 1
                        TIS_prob = ORF_TIS_prob[i][j][k]
                        start_codon = ORF_start_codon[i][j][k]
                        ATG = start_codon == 0
                        GTG = start_codon == 1
                        TTG = start_codon == 2
                        combprob = geneprob * weight_gene_prob + TIS_prob * weight_TIS_prob + ATG * weight_ATG + GTG * weight_TTG + TTG * weight_GTG
                        maxprob = weight_gene_prob + weight_TIS_prob + max(weight_ATG, weight_TTG, weight_GTG)
                        probthresh = score_threshold * maxprob
                        score = (combprob - probthresh) * length + 1e6*seengene[seengene_idx]
                        seengene_idx += 1
                        temp.append(score)
                ORF_score_flat.append(temp)
        else:
            ORF_score_flat = []
            for i, contig in enumerate(ORF_gene_score):
                if not any(contig):
                    ORF_score_flat.append([])
                    continue
                temp = []
                for j, frame in enumerate(contig):
                    for k, geneprob in enumerate(frame):
                        length = ORF_coord_list[i][j][k][1] - ORF_coord_list[i][j][k][0] + 1
                        TIS_prob = ORF_TIS_prob[i][j][k]
                        start_codon = ORF_start_codon[i][j][k]
                        ATG = start_codon == 0
                        GTG = start_codon == 1
                        TTG = start_codon == 2
                        combprob = geneprob * weight_gene_prob + TIS_prob * weight_TIS_prob + ATG * weight_ATG + GTG * weight_TTG + TTG * weight_GTG
                        maxprob = weight_gene_prob + weight_TIS_prob + max(weight_ATG, weight_TTG, weight_GTG)
                        probthresh = score_threshold * maxprob
                        score = (combprob - probthresh) * length
                        temp.append(score)
                ORF_score_flat.append(temp)
        # DAGs to maximize geneiness on each contig
        contig_gene_coord = []
        contig_gene_strand = []
        for i, coords in enumerate(ORF_coord_flat):
            contigNum = i+1
            pbar.set_description_str(orgName+" > Create graph ctg"+str(contigNum).rjust(maxpathSize-len(orgName)-len(" > Create graph ctg"+str(contigNum))))
            # sort coords, lengths, strands, and scores
            startpos = np.array([x[0] for x in coords])
            sortidx = list(np.argsort(startpos))
            coords_sorted = [coords[j] for j in sortidx]
            # lengths = ORF_length_flat[i]
            # lengths_sorted = [lengths[j] for j in sortidx]
            scores = ORF_score_flat[i]
            scores_sorted = [scores[j] for j in sortidx]
            strands = ORF_strand_flat[i]
            strands_sorted = [strands[j] for j in sortidx]
            # create DAG
            # keep track of graph path and score
            predecessor = np.zeros(len(scores_sorted), dtype=np.int64)
            max_path_score = np.zeros(len(scores_sorted), dtype=np.int64)
            # add null starting node
            n_connections = 0
            idx_offset = 1
            while n_connections < max_forward_connections:
                k = idx_offset
                idx_offset += 1
                # dont try to add edge past last ORF
                if k > len(scores_sorted)-1:
                    n_connections += 1
                    continue
                edge_weight = scores_sorted[k-1]
                # initial scores from null node
                max_path_score[k] = edge_weight
                predecessor[k] = 0
                n_connections += 1
            # add edges between compatible ORFs
            for j in range(1, len(scores_sorted)-1):
                n_connections = 0
                idx_offset = 1
                while n_connections < max_forward_connections:
                    k = j + idx_offset
                    idx_offset += 1
                    # dont try to add edge past end of contigs
                    if k > len(scores_sorted)-1:
                        n_connections += 1
                        continue
                    coords0 = coords_sorted[j-1]
                    coords1 = coords_sorted[k-1]
                    strand0 = strands_sorted[j-1]
                    strand1 = strands_sorted[k-1]
                    compat, penalty = analyze_overlap(coords0, coords1, strand0, strand1, unidirectional_penalty_per_base, convergent_penalty_per_base, divergent_penalty_per_base, max_gene_overlap)
                    if compat:
                        score = scores_sorted[k-1] - penalty
                        path_score = max_path_score[j] + score
                        if path_score > max_path_score[k]:
                            max_path_score[k] = path_score
                            predecessor[k] = j
                        n_connections += 1
            # solve for geneiest path through contig
            pbar.set_description_str(orgName+" > Max geneiness ctg"+str(contigNum).rjust(maxpathSize-len(orgName)-len(" > Max geneiness ctg"+str(contigNum))))
            pred_idx = np.argmax(max_path_score)
            pred_path = []
            while pred_idx > 0:
                pred_path.append(pred_idx)
                pred_idx = predecessor[pred_idx]
            # 0 isnt added
            max_ORF_PATH = [x-1 for x in pred_path[:]]
            gene_predict_coords = [coords_sorted[j] for j in max_ORF_PATH]
            gene_predict_strand = [strands_sorted[j] for j in max_ORF_PATH]
            # mmseqs filter
            if boolMmseqs:
                cutoff = 200
                # get amino acid sequence from coherent ORFs
                # 3' TO 5'
                aa_sorted = [ORF_seq_enc[j] for j in sortidx]
                aa_tensor = [aa_sorted[j] for j in max_ORF_PATH]
                aa_seq = [tensor_to_seq([int(y) for y in x]) for x in aa_tensor]
                # make temp dir to store mmseqs stuff
                finding_empty_dir = True
                dir_idx = 0
                while finding_empty_dir:
                    dirpath = os.path.join(pathTMP, str(dir_idx))
                    if os.path.isdir(dirpath):
                        dir_idx += 1
                        continue
                    else:
                        os.makedirs(dirpath)
                        finding_empty_dir = False
                # make mmseqs 3'5' and 5'3' DB
                query_fasta_path_35 = os.path.join(dirpath, "candidate_genes_35.fasta")
                with open(query_fasta_path_35, "w") as f:
                    for i, s in enumerate(aa_seq):
                        f.writelines(">" + str(i) + "\n"+str(s) + "\n")
                query_fasta_path_53 = os.path.join(dirpath, "candidate_genes_53.fasta")
                with open(query_fasta_path_53, "w") as f:
                    for i, s in enumerate(aa_seq):
                        f.writelines(">" + str(i) + "\n"+str(s)[::-1] + "\n")
                pbar.set_description_str(orgName+" > mmseqs createdb ctg"+str(contigNum).rjust(maxpathSize-len(orgName)-len(" > mmseqs createdb ctg"+str(contigNum))))
                query_DB_path_35 = os.path.join(dirpath, "candidateDB_35")
                os.system(dicoGeminiPath['TOOLS']['mmseqs']+" createdb "+query_fasta_path_35+" "+query_DB_path_35+" >> "+pathLOG+" 2>&1")
                query_DB_path_53 = os.path.join(dirpath, "candidateDB_53")
                os.system(dicoGeminiPath['TOOLS']['mmseqs']+" createdb "+query_fasta_path_53+" "+query_DB_path_53+" >> "+pathLOG+" 2>&1")
                # ***** GENEXA 3'>5' ***** #
                pbar.set_description_str(orgName+" > mmseqs genexa ctg"+str(contigNum).rjust(maxpathSize-len(orgName)-len(" > mmseqs genexa ctg"+str(contigNum))))
                results_DB_path_35 = os.path.join(dirpath, "resultsDB_35")
                os.system(dicoGeminiPath['TOOLS']['mmseqs']+" search -s 7.0 "+query_DB_path_35+" "+genexa_DB_path+" "+results_DB_path_35+" "+dirpath+" --threads "+str(cpu)+" >> "+pathLOG+" 2>&1")
                m8_path_genexa = os.path.join(dirpath, "resultDB_genexa.m8")
                os.system(dicoGeminiPath['TOOLS']['mmseqs']+" convertalis "+query_DB_path_35+" "+genexa_DB_path+" "+results_DB_path_35+" "+m8_path_genexa+" --format-output \"query,target,evalue,raw\" --threads "+str(cpu)+" > "+pathLOG+" 2>&1")
                mmseqs_results_genexa = pandas.read_table(m8_path_genexa, header=None, names=["query", "target", "evalue", "raw"]).to_numpy()
                hit_idx_genexa = np.unique(mmseqs_results_genexa[:, 0]).astype(int)
                # ***** SWISSPROT 5'>3' ***** #
                pbar.set_description_str(orgName+" > mmseqs swissprot ctg"+str(contigNum).rjust(maxpathSize-len(orgName)-len(" > mmseqs swissprot ctg"+str(contigNum))))
                results_DB_path_53 = os.path.join(dirpath, "resultsDB_53")
                os.system(dicoGeminiPath['TOOLS']['mmseqs']+" search -s 7.0 "+query_DB_path_53+" "+swissprot_DB_path+" "+results_DB_path_53+" "+dirpath+" --threads "+str(cpu)+" >> "+pathLOG+" 2>&1")
                m8_path_secondary = os.path.join(dirpath, "resultDB_secondary.m8")
                os.system(dicoGeminiPath['TOOLS']['mmseqs']+" convertalis "+query_DB_path_53+" "+swissprot_DB_path+" "+results_DB_path_53+" "+m8_path_secondary+" --format-output \"query,target,evalue,raw\" --threads "+str(cpu)+" > "+pathLOG+" 2>&1")
                mmseqs_results_secondary = pandas.read_table(m8_path_secondary, header=None, names=["query", "target", "evalue", "raw"]).to_numpy()
                hit_idx_secondary = np.unique(mmseqs_results_secondary[:, 0]).astype(int)
                if division == "PHG":
                    # ***** phannsDB 5'>3' ***** #
                    pbar.set_description_str(orgName+" > mmseqs phanns ctg"+str(contigNum).rjust(maxpathSize-len(orgName)-len(" > mmseqs phanns ctg"+str(contigNum))))
                    results_DB_path_53_phanns = os.path.join(dirpath, "resultsDB_53_phanns")
                    os.system(dicoGeminiPath['TOOLS']['mmseqs']+" search -s 7.0 "+query_DB_path_53+" "+phanns_DB_path+" "+results_DB_path_53_phanns+" "+dirpath+" --threads "+str(cpu)+" >> "+pathLOG+" 2>&1")
                    m8_path_secondary_phanns = os.path.join(dirpath, "resultDB_secondary_phanns.m8")
                    os.system(dicoGeminiPath['TOOLS']['mmseqs']+" convertalis "+query_DB_path_53+" "+phanns_DB_path+" "+results_DB_path_53_phanns+" "+m8_path_secondary_phanns+" --format-output \"query,target,evalue,raw\" --threads "+str(cpu)+" > "+pathLOG+" 2>&1")
                    mmseqs_results_secondary_phanns = pandas.read_table(m8_path_secondary_phanns, header=None, names=["query", "target", "evalue", "raw"]).to_numpy()
                    hit_idx_secondary_phanns = np.unique(mmseqs_results_secondary_phanns[:, 0]).astype(int)
                    # ***** pVOGS 5'>3' ***** #
                    pbar.set_description_str(orgName+" > mmseqs pvogs ctg"+str(contigNum).rjust(maxpathSize-len(orgName)-len(" > mmseqs pvogs ctg"+str(contigNum))))
                    results_DB_path_53_pvogs = os.path.join(dirpath, "resultsDB_53_pvogs")
                    os.system(dicoGeminiPath['TOOLS']['mmseqs']+" search -s 7.0 "+query_DB_path_53+" "+pvogs_DB_path+" "+results_DB_path_53_pvogs+" "+dirpath+" --threads "+str(cpu)+" >> "+pathLOG+" 2>&1")
                    m8_path_secondary_pvogs = os.path.join(dirpath, "resultDB_secondary_pvogs.m8")
                    os.system(dicoGeminiPath['TOOLS']['mmseqs']+" convertalis "+query_DB_path_53+" "+pvogs_DB_path+" "+results_DB_path_53_pvogs+" "+m8_path_secondary_pvogs+" --format-output \"query,target,evalue,raw\" --threads "+str(cpu)+" > "+pathLOG+" 2>&1")
                    mmseqs_results_secondary_pvogs = pandas.read_table(m8_path_secondary_pvogs, header=None, names=["query", "target", "evalue", "raw"]).to_numpy()
                    hit_idx_secondary_pvogs = np.unique(mmseqs_results_secondary_pvogs[:, 0]).astype(int)
                    cutoffpath = [x for i, x in enumerate(max_ORF_PATH) if (scores_sorted[x] > cutoff or (i in hit_idx_genexa or i in hit_idx_secondary or i in hit_idx_secondary_phanns or i in hit_idx_secondary_pvogs))]
                else:
                    cutoffpath = [x for i, x in enumerate(max_ORF_PATH) if (scores_sorted[x] > cutoff or (i in hit_idx_genexa or i in hit_idx_secondary))]
                # filter gene predictions, keep if mmseqs hit or high enough gene score
                gene_predict_coords = [coords_sorted[j] for j in cutoffpath]
                gene_predict_strand = [strands_sorted[j] for j in cutoffpath]
                # graph_score_cutoff = [scores_sorted[j] for j in cutoffpath]
                contig_gene_coord.append(gene_predict_coords)
                contig_gene_strand.append(gene_predict_strand)
                # n_genes = len(gene_predict_coords)
            else:
                cutoff = 100
                cutoffpath = [x for x in max_ORF_PATH if scores_sorted[x] > cutoff]
                gene_predict_coords = [coords_sorted[j] for j in cutoffpath]
                gene_predict_strand = [strands_sorted[j] for j in cutoffpath]
                # graph_score_cutoff = [scores_sorted[j] for j in cutoffpath]
                contig_gene_coord.append(gene_predict_coords)
                contig_gene_strand.append(gene_predict_strand)
                # n_genes = len(gene_predict_coords)
        contig_gene_coord_list.append(contig_gene_coord)
        contig_gene_strand_list.append(contig_gene_strand)
        pbar.update(1)
        title("Find genes", pbar)
    pbar.close()
    # ***** Write output GFF & FFN & FAA ***** #
    printcolor("♊ Output"+"\n")
    pbar = tqdm(total=len(contig_name_list), dynamic_ncols=True, ncols=75, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt}")
    for i in range(len(contig_name_list)):
        for j in range(len(contig_name_list[i])):
            contigName = contig_name_list[i][j]
            orgName = dicoContigToOrg[contigName]
            contigSeq = contig_seq_list[i][j]
            contigGeneCoord = contig_gene_coord_list[i][j][::-1]
            contigGeneStrand = contig_gene_strand_list[i][j][::-1]
            pathFFN = pathOUT+"/"+orgName+".ffn"
            pathGFF = pathOUT+"/"+orgName+".gff"
            # Add header to GFF if first opening
            if not os.path.isfile(pathGFF):
                GFF = open(pathGFF, 'w')
                GFF.write("##gff-version 3\n##sequence-region "+orgName+" 1 "+str(len(contigSeq))+"\n")
            else:
                GFF = open(pathGFF, 'a')
            FFN = open(pathFFN, 'a')
            for k in range(len(contigGeneCoord)):
                geneStart, geneEnd = contigGeneCoord[k]
                geneStrand = contigGeneStrand[k]
                geneName = contigName+"_gene"+str(k+1)+"|"+str(geneStrand)
                if geneStrand == 1:
                    geneSeq = contigSeq[geneStart-3: geneEnd+3]
                    gffStrand = "+"
                else:
                    geneSeq = contigSeq[geneStart-3: geneEnd+3].reverse_complement()
                    gffStrand = "-"
                FFN.write(">"+geneName+"\n"+str(geneSeq)+"\n")
                GFF.write(contigName+"\tBalrog\tCDS\t"+str(geneStart-3+1)+"\t"+str(geneEnd+3)+"\t.\t"+gffStrand+"\t0\tinference = ab initio prediction: Balrog;product = hypothetical protein;locus_tag = "+geneName+"\n")
            FFN.close()
            GFF.close()
        pbar.update(1)
        title("Output", pbar)
    pbar.close()
    # ***** Translate to FAA ***** #
    transeq(pathIN=pathOUT, pathOUT=pathOUT, fromPhageDb=False, ext=".ffn")
    # ***** Make GBK files ***** #
    for pathFile in lstFiles:
        orgName = os.path.basename(pathFile).replace(ext, "").replace("."+ext, "")
        pathFFN = pathOUT+"/"+orgName+".ffn"
        pathFAA = pathOUT+"/"+orgName+".faa"
        pathGBK = pathOUT+"/"+orgName+".gbk"
        make_gbk_from_fasta(pathIN1=pathFile, pathIN2=pathFFN, pathIN3=pathFAA, pathOUT=pathGBK,
                            identifier=orgName, topology=topology, division=division, taxID=taxID)


@fct_checker
def transeq(pathIN: str, pathOUT: str, boolOrgName: bool = True, fromPhageDb: bool = False, ext: str = ".ffn") -> Tuple[str, str, bool, bool, str]:
    '''
     -----------------------------------------------------------
    |                  TRANSLATE GENE TO PROTEIN                |
     -----------------------------------------------------------
    |          Translate gene CDS to protein (transeq)          |
     -----------------------------------------------------------
    |PARAMETERS                                                 |
    |    pathIN     : path of input files or folder (required)  |
    |    pathOUT    : path of output files (required)           |
    |    boolOrgName: add "[orgName]" to header (default=True)  |
    |    fromPhageDb: function call from phageDB (default=False)|
    |    ext        : extension of input files (default=.ffn)   |
     -----------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "transeq", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: transeq]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    printcolor("♊ Transeq"+"\n")
    if fromPhageDb is True:
        pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=75, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    else:
        pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFFN in lstFiles:
        file = os.path.basename(pathFFN)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pathFFN = pathIN+"/"+orgName+".ffn"
        pathFAA = pathOUT+"/"+orgName+".faa"
        if fromPhageDb is True:
            if "GCA" in file:
                pbar.set_description_str("GCA_"+file.replace(ext, "").split("_")[-1])
            else:
                pbar.set_description_str(file.replace(ext, "")[-15:])
        else:
            pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        if not os.path.isfile(pathFAA) or os.path.getsize(pathFAA) == 0:
            dicoFFN = make_fasta_dict(pathFFN)
            FAA = open(pathFAA, 'w')
            cpt = 1
            for key in dicoFFN:
                cdna = Seq(dicoFFN[key])
                seqProt = str(cdna.translate(table=11, cds=False, to_stop=True))
                if fromPhageDb is True:
                    FAA.write(">"+key.split(" [START")[0]+"\n"+seqProt+"\n")
                elif boolOrgName is True:
                    FAA.write(">"+key+" ["+orgName+"]\n"+seqProt+"\n")
                else:
                    FAA.write(">"+key+"\n"+seqProt+"\n")
                cpt += 1
            FAA.close()
        pbar.update(1)
        title("Transeq", pbar)
    pbar.close()


'''
-------------------------------------------------------------------------------------------------------
                                     NON-CODING ANNOTATION FUNCTIONS
-------------------------------------------------------------------------------------------------------
'''


@fct_checker
def trnascan_se(pathIN: str, pathOUT: str, model: str = "-B", ext: str = ".fna") -> Tuple[str, str, str, str]:
    '''
     ------------------------------------------------------------
    |              tRNA GENE ANNOTATION (tRNAscan-SE)            |
     ------------------------------------------------------------
    |          Launch tRNAscan-SE to predict tRNA genes          |
     ------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN : path of input files or folder (required)       |
    |    pathOUT: path of output files (required)                |
    |    model  : model -E,-B,-A,-M <model>,-O,-G (default=-B)   |
    |    ext    : extension of input files (default=.fna)        |
     ------------------------------------------------------------
    | TOOLS: trnascanse                                          |
     -----------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "trnascan_se", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: trnascan_se]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'trnascanse' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['trnascanse'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    printcolor("♊ tRNAscan-SE"+"\n")
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFNA in lstFiles:
        file = os.path.basename(pathFNA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        pathTRNASCANSE = pathOUT+"/"+orgName+".trnascanse"
        # Launch tRNAscan-SE (empty if any prediction)
        if not os.path.isfile(pathTRNASCANSE):
            cmdTRNAscanse = dicoGeminiPath['TOOLS']['trnascanse']+" "+model+" --forceow --quiet --thread "+str(cpu)+" -o "+pathTRNASCANSE+" "+pathFNA
            os.system(cmdTRNAscanse)
        pbar.update(1)
        title("tRNAscan-SE", pbar)
    pbar.close()


'''
-------------------------------------------------------------------------------------------------------
                                       CODING ANNOTATION FUNCTIONS
-------------------------------------------------------------------------------------------------------
'''


@fct_checker
def pvogs(pathIN: str, pathOUT: str, ext: str = ".faa") -> Tuple[str, str, str]:
    '''
     ------------------------------------------------------------
    |               PHAGE pVOGs HMMSCAN ANNOTATION               |
     ------------------------------------------------------------
    |           hmmscan against pVOGS profiles database          |
     ------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN : path of input files or folder (required)       |
    |    pathOUT: path of pVOGs hmmscan tblout file (required)   |
    |    ext    : extension of input files (default=.faa)        |
     ------------------------------------------------------------
    | TOOLS: hmmsearch                                           |
     -----------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "pvogs", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: pvogs]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'hmmsearch' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['hmmsearch'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    printcolor("♊ pVOGs"+"\n")
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFAA in lstFiles:
        file = os.path.basename(pathFAA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        pathPVOGS = pathOUT+"/"+orgName+".pvogs"
        # Launch hmmscan on pVOGs database
        if not os.path.isfile(pathPVOGS) or os.path.getsize(pathPVOGS) == 0:
            cmdHmmsearch = dicoGeminiPath['TOOLS']['hmmsearch']+" --cpu "+str(cpu)+" -o /dev/null --noali --tblout "+pathPVOGS+" "+dicoGeminiPath['DATABASES']['pvogs_hmm']+" "+pathFAA
            os.system(cmdHmmsearch)
        pbar.update(1)
        title("pVOGs", pbar)
    pbar.close()


@fct_checker
def diamond_p(pathIN: str, pathDB: str, pathOUT: str, boolSeq: bool = False, ext: str = ".faa") -> Tuple[str, str, str, bool, str]:
    '''
     ------------------------------------------------------------
    |                       DIAMOND BLASTP                       |
    |------------------------------------------------------------
    |           diamond blastP against input database            |
    |------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN : path of input files or folder (required)       |
    |    pathDB : path of diamond dmnd file (required)           |
    |    pathOUT: path of blast results (required)               |
    |    boolSeq: add sequences to output (default=false)        |
    |    ext    : extension of input files (default=.faa)        |
     ------------------------------------------------------------
    | TOOLS: diamond                                             |
     -----------------------------------------------------------
    '''
    pathDB = path_converter(pathDB)
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "diamond_blastp", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: diamond_blastp]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    if os.path.isdir(pathIN):
        os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'diamond' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['diamond'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    pathTMP = geminiset.pathTMP
    # Make db if not done
    if os.path.basename(pathDB)[-5:] != ".dmnd":
        if not os.path.isdir(pathTMP):
            pathTMP = "/tmp"
        pathTMPDB = pathTMP+"/"+os.path.basename(pathDB)
        cmdDiamondDB = dicoGeminiPath['TOOLS']['diamond']+" makedb --in "+pathDB+" -d "+pathTMPDB+" --threads "+str(cpu)+" --quiet"
        os.system(cmdDiamondDB)
        pathDB = pathTMPDB
    dbTitle = os.path.basename(pathDB).replace(".dmnd", "")
    if boolSeq is True:
        outfmt = "6 qseqid stitle pident length qlen qstart qend slen sstart send evalue full_sseq"
    else:
        outfmt = "6 qseqid stitle pident length qlen qstart qend slen sstart send evalue"
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFAA in lstFiles:
        file = os.path.basename(pathFAA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        if os.path.isdir(pathIN):
            pathRES = pathOUT+"/"+orgName+"_"+dbTitle+".tsv"
        else:
            pathRES = pathOUT
        # Launch diamond blastp on collection database
        if not os.path.isfile(pathRES) or os.path.getsize(pathRES) == 0:
            cmdDiamond = dicoGeminiPath['TOOLS']['diamond']+" blastp -d "+pathDB+" -q "+pathFAA+" -o "+pathRES+" --threads "+str(cpu)+" --outfmt "+outfmt+" --max-target-seqs 1000000000 --header --quiet"
            os.system(cmdDiamond)
        pbar.update(1)
        title("diamond_p "+dbTitle, pbar)
    pbar.close()


@fct_checker
def interproscan(pathIN: str, pathOUT: str, ext: str = ".faa") -> Tuple[str, str, str]:
    '''
     ------------------------------------------------------------
    |                  INTERPROSCAN ANNOTATION                   |
     ------------------------------------------------------------
    |                  InterProScan annotation                   |
     ------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN : path of input files or folder (required)       |
    |    pathOUT: path of output JSON file (required)            |
    |    ext    : extension of input files (default=.faa)        |
     ------------------------------------------------------------
    | TOOLS: interproscan                                        |
     ------------------------------------------------------------
    | App: CDD-3.18, Coils-2.2.1, Gene3D-4.3.0, Hamap-2020_05    |
    |      MobiDBLite-2.0, PANTHER-15.0, Pfam-33.1               |
    |      Phobius-1.01, PIRSF-3.10, PIRSR-2021_02, PRINTS-42.0  |
    |      ProSitePatterns-2021_01, ProSiteProfiles-2021_01      |
    |      SFLD-4, SignalP_GRAM_NEGATIVE-4.1, SMART-7.1          |
    |      SUPERFAMILY-1.75, TIGRFAM-15.0, TMHMM-2.0c            |
     ------------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "interproscan", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: interproscan]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'interproscan' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['interproscan'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    strLstApp = "CDD, Hamap, PANTHER, Pfam, PIRSF, PIRSR, PRINTS, ProSiteProfiles, SFLD, SMART, SUPERFAMILY, TIGRFAM"
    spinner = yaspin(Spinners.aesthetic, text="♊ InterProScan", side="right")
    spinner.start()
    title("IPRSCAN", None)
    pathINTERPRO = pathOUT+"/interproscan.tsv"
    if not os.path.isfile(pathINTERPRO) or os.path.getsize(pathINTERPRO) == 0:
        # CAT input FASTA
        cat_lstfiles(lstFiles, geminiset.pathTMP+"/concat.faa")
        # LAUNCH interproscan
        if ".jar" in dicoGeminiPath['TOOLS']['interproscan']:
            os.chdir(os.path.dirname(dicoGeminiPath['TOOLS']['interproscan']))
            cmdIprscan = "java -Xms"+str(memMin)+"g -Xmx"+str(memMax)+"g -jar "+dicoGeminiPath['TOOLS']['interproscan'] + \
                         " --input "+geminiset.pathTMP+"/concat.faa --outfile "+pathINTERPRO+" --formats TSV --tempdir "+geminiset.pathTMP + \
                         " --seqtype p --applications "+strLstApp+" --disable-precalc --disable-residue-annot" + \
                         " --cpu "+str(cpu)+" > "+geminiset.pathTMP+"/iprscan.log 2>&1"
        else:
            cmdIprscan = dicoGeminiPath['TOOLS']['interproscan'] + \
                         " --input "+geminiset.pathTMP+"/concat.faa --outfile "+pathINTERPRO+" --formats TSV --tempdir "+geminiset.pathTMP + \
                         " --seqtype p --applications "+strLstApp+" --disable-precalc --disable-residue-annot" + \
                         " --cpu "+str(cpu)+" > "+geminiset.pathTMP+"/iprscan.log 2>&1"
        os.system(cmdIprscan)
        # CHECK errors
        if not os.path.isfile(pathINTERPRO) or os.path.getsize(pathINTERPRO) == 0:
            spinner.stop()
            printcolor("♊ InterProScan"+"\n")
            printcolor("[ERROR: interproscan]\nCheck log file \""+geminiset.pathTMP+"/iprscan.log\"\n", 1, "212;64;89", "None", True)
            exit_gemini()
    spinner.stop()
    printcolor("♊ InterProScan"+"\n")


@fct_checker
def eggnog(pathIN: str, pathOUT: str, idThr: int = 20, covThr: int = 50, ext: str = ".faa") -> Tuple[str, str, int, int, str]:
    '''
     ------------------------------------------------------------
    |                     EGGNOG ANNOTATION                      |
     ------------------------------------------------------------
    |             Annotate protein FASTA with EggNOG             |
     ------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN : path of input files or folder (required)       |
    |    pathOUT: path of output file (required)                 |
    |    idThr  : %identity threshold (default=20)               |
    |    covThr : %cover threshold (default=50)                  |
    |    ext    : extension of input files (default=.faa)        |
     ------------------------------------------------------------
    | TOOLS: eggnog-mapper, eggnog-db                            |
     ------------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "eggnog", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: eggnog]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'eggnog-mapper' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['eggnog-mapper'])
    if 'eggnog-db' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['eggnog-db'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    spinner = yaspin(Spinners.aesthetic, text="♊ EggNOG", side="right")
    spinner.start()
    title("EGGNOG", None)
    pathEGGNOG = pathOUT+"/eggnog.annotations"
    if not os.path.isfile(pathEGGNOG) or os.path.getsize(pathEGGNOG) == 0:
        # CAT input FASTA
        cat_lstfiles(lstFiles, geminiset.pathTMP+"/concat.faa")
        # LAUNCH EggNOG
        cmdEmapper = dicoGeminiPath['TOOLS']['eggnog-mapper']+" -i "+geminiset.pathTMP+"/concat.faa --output all_proteins --output_dir "+geminiset.pathTMP+" --temp_dir "+geminiset.pathTMP + \
            " -m diamond --itype proteins --pident "+str(idThr)+" --query_cover "+str(covThr)+" --subject_cover "+str(covThr) + \
            " --data_dir "+dicoGeminiPath['TOOLS']['eggnog-db']+" --cpu "+str(cpu)+" --hmm_maxhits 0 > "+geminiset.pathTMP+"/eggnog.log 2>&1"
        os.system(cmdEmapper)
        # CHECK errors
        if not os.path.isfile(geminiset.pathTMP+"/all_proteins.emapper.annotations") or os.path.getsize(geminiset.pathTMP+"/all_proteins.emapper.annotations") == 0:
            spinner.stop()
            printcolor("♊ EggNOG"+"\n")
            printcolor("[ERROR: eggnog]\nCheck log file \""+geminiset.pathTMP+"/eggnog.log\"\n", 1, "212;64;89", "None", True)
            exit_gemini()
        # COPY result, dont use shutil move to avoid "OSError: [Errno 18] Invalid cross-device link"
        shutil.copy(geminiset.pathTMP+"/all_proteins.emapper.annotations", pathEGGNOG)
    spinner.stop()
    printcolor("♊ EggNOG"+"\n")


@fct_checker
def recombinase(pathIN: str, pathOUT: str, ext: str = ".faa") -> Tuple[str, str, str]:
    '''
     ------------------------------------------------------------
    |               RECOMBINASE HMMSCAN ANNOTATION               |
     ------------------------------------------------------------
    |  hmmscan against VirFam/PFAM recombinase profiles database |
     ------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN : path of input files or folder (required)       |
    |    pathOUT: path of hmmscan tblout file (required)         |
    |    ext    : extension of input files (default=.faa)        |
     ------------------------------------------------------------
    | TOOLS: hmmsearch                                           |
     ------------------------------------------------------------
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "recombinase", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: recombinase]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'hmmsearch' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['hmmsearch'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    printcolor("♊ Recombinase"+"\n")
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFAA in lstFiles:
        file = os.path.basename(pathFAA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        pathREC = pathOUT+"/"+orgName+".recomb"
        # Launch hmmscan on VirFam/PFAM recombinase database
        if not os.path.isfile(pathREC) or os.path.getsize(pathREC) == 0:
            cmdHmmsearch = dicoGeminiPath['TOOLS']['hmmsearch']+" --cpu "+str(cpu)+" -o /dev/null --noali --tblout "+pathREC+" "+dicoGeminiPath['DATABASES']['recombinase_hmm']+" "+pathFAA
            os.system(cmdHmmsearch)
        pbar.update(1)
        title("Recombinase", pbar)
    pbar.close()


@fct_checker
def defense_system(pathIN: str, pathOUT: str, dfmodelV: str = "1.1.0", plmodelV: str = "1.4.0", ext: str = ".faa") -> Tuple[str, str, str, str, str]:
    '''
     ------------------------------------------------------------
    |              SEARCH BACTERIAL DEFENSE SYSTEMS              |
     ------------------------------------------------------------
    |    Search defense systems using DefenseFinder & PADLOC     |
     ------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN  : path of input files or folder (required)      |
    |    pathOUT : path of output folder (required)              |
    |    dfmodelV: defenseFinder model version (default=1.1.0)   |
    |    plmodelV: PADLOC model version (default=1.4.0)          |
    |    ext     : extension of input files (default=.faa)       |
     ------------------------------------------------------------
    | TOOLS: macsyfinder, padloc                                 |
     ------------------------------------------------------------
    dicoDF / dicoPL
        "perOrg" : {"org1": {"syst1": [[lt1],[lt2],[lt3]], "syst2": [[lt4],[lt5]]}}
        "perLT"  : {"org1": {"lt1": {"syst1": "geneName1"}, "lt1": {"syst1": "geneName1"}}}
        "perSyst": {"syst1": {"org1": [lt1,lt2,lt3], "org2": [lt7,lt8,lt9]}}
    ERROR in macsyfinder, replace with following
        def _parse_exchangeable(self, node, gene_ref, curr_model):
            name = node.get("name")
            model_name = split_def_name(curr_model.fqn)[0]
            key = (model_name, name)
            c_gene = self.gene_bank[key]
            ex = Exchangeable(c_gene, gene_ref)
            return ex
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "defense_system", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: defense_system]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'macsyfinder' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['macsyfinder'])
    if 'padloc' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['padloc'])
    slurmBool, cpu, memMax, memMin = get_sys_info()
    pathLOG = pathOUT+"/gemini.log"
    # ***** DefenseFinder ***** #
    printcolor("♊ DefenseFinder"+"\n")
    lstModels = os.listdir(dicoGeminiPath['DATABASES']['defense-finder-models']+"/"+dfmodelV+"/definitions")
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFAA in lstFiles:
        file = os.path.basename(pathFAA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        pathDFgenesOUT = pathOUT+"/"+orgName+"_DefenseFinder_"+dfmodelV+"_genes.tsv"
        pathDFsystOUT = pathOUT+"/"+orgName+"_DefenseFinder_"+dfmodelV+"_systems.tsv"
        # Launch macsyfinder for each model
        if not os.path.isfile(pathDFgenesOUT) and not os.path.isfile(pathDFsystOUT):
            cpt = 1
            for model in lstModels:
                pbar.set_description_str(orgName+" ("+str(cpt)+"/"+str(len(lstModels))+")"+" ".rjust(maxpathSize-len(orgName)))
                if model == "Cas":
                    covProfile = 0.4
                    accWeight = " --accessory-weight 1"
                else:
                    covProfile = 0.1
                    accWeight = ""
                cmdMacsyfinder = dicoGeminiPath['TOOLS']['macsyfinder']+" --db-type ordered_replicon --sequence-db "+pathFAA + \
                    " --models-dir "+dicoGeminiPath['DATABASES']['defense-finder-models']+" --models "+dfmodelV+"/"+model+" all" + \
                    " --out-dir "+geminiset.pathTMP+"/"+model+" --coverage-profile "+str(covProfile) + \
                    " --w "+str(cpu)+" --exchangeable-weight 1"+accWeight+" >> "+pathLOG+" 2>&1"
                os.system(cmdMacsyfinder)
                cpt += 1
            # DefenseFinder post treatment scripts
            bs = get_best_solution(geminiset.pathTMP)
            export_defense_finder_genes(bs, pathOUT, orgName, dfmodelV)
            export_defense_finder_systems(bs, pathOUT, orgName, dfmodelV)
            # Remove files
            for model in lstModels:
                shutil.rmtree(geminiset.pathTMP+"/"+model)
            if os.path.isfile(pathFAA+".idx"):
                os.remove(pathFAA+".idx")
        title("DefenseFinder", pbar)
        pbar.update(1)
    pbar.close()
    # ***** PADLOC ***** #
    printcolor("♊ PADLOC"+"\n")
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFAA in lstFiles:
        file = os.path.basename(pathFAA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        pathPADLOCOUT = pathOUT+"/"+orgName+"_PADLOC_"+plmodelV+".tsv"
        if not os.path.isfile(pathPADLOCOUT):
            # Create artficial required gff
            dicoFAA = make_fasta_dict(pathFAA)
            GFF = open(geminiset.pathTMP+"/padloc.gff", 'w')
            FAA = open(geminiset.pathTMP+"/padloc.faa", 'w')
            cpt = 1
            for key in dicoFAA:
                lt = key.split(" ")[0].split("|")[0]
                line = orgName+"\tPADLOC GFF\tCDS\t"+str(cpt)+"\t"+str(cpt+99)+"\t.\t+\t0\tID=cds-"+lt+";Parent=gene-"+lt+";Name="+lt+";gbkey=CDS;locus_tag="+lt+";protein_id="+lt+";transl_table=11"
                FAA.write(">"+lt+"\n"+dicoFAA[key]+"\n")
                GFF.write(line+"\n")
                cpt += 100
            GFF.close()
            FAA.close()
            # Launch PADLOC
            cmdPADLOC = dicoGeminiPath['TOOLS']['padloc']+" --cpu "+str(cpu)+" --data "+dicoGeminiPath['DATABASES']['padloc-db']+"/"+plmodelV + \
                " --faa "+geminiset.pathTMP+"/padloc.faa"+" --gff "+geminiset.pathTMP+"/padloc.gff --outdir "+geminiset.pathTMP+" >> "+pathLOG+" 2>&1"
            os.system(cmdPADLOC)
            pathTMPCSV = geminiset.pathTMP+"/padloc_padloc.csv"
            if not os.path.isfile(pathTMPCSV) or os.path.getsize(pathTMPCSV) == 0:
                OUT = open(pathTMPCSV, 'w')
                OUT.write("system.number\tseqid\tsystem\ttarget.name\thmm.accession\thmm.name\tprotein.name\tfull.seq.E.value\tdomain.iE.value\ttarget.coverage\thmm.coverage\tstart\tend\tstrand\ttarget.description\trelative.position\tcontig.end\tall.domains\tbest.hits\n")
                OUT.close()
            cmdSeq = "cat "+geminiset.pathTMP+"/padloc_padloc.csv | sed s/\",\"/\"\\t\"/g > "+pathPADLOCOUT
            os.system(cmdSeq)
            os.remove(geminiset.pathTMP+"/padloc.gff")
            os.remove(geminiset.pathTMP+"/padloc.faa")
            os.remove(geminiset.pathTMP+"/padloc.domtblout")
            os.remove(geminiset.pathTMP+"/padloc_padloc.csv")
        title("PADLOC", pbar)
        pbar.update(1)
    pbar.close()
    # ***** PARSER ***** #
    printcolor("♊ PARSE"+"\n")
    dicoDF = {'perOrg': {}, 'perLT': {}, 'perSyst': {}}
    dicoPL = {'perOrg': {}, 'perLT': {}, 'perSyst': {}}
    for pathFAA in lstFiles:
        file = os.path.basename(pathFAA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        dicoDF['perOrg'][orgName] = {}
        dicoDF['perLT'][orgName] = {}
        dicoPL['perOrg'][orgName] = {}
        dicoPL['perLT'][orgName] = {}
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        # DefenseFinder
        pathDFsystOUT = pathOUT+"/"+orgName+"_DefenseFinder_"+dfmodelV+"_systems.tsv"
        for dfLine in read_file(pathDFsystOUT)[1:]:
            splitLine = dfLine.split("\t")
            subtype = splitLine[2]
            lstGenes = splitLine[5].split(",")
            lstGeneNames = splitLine[7].split(",")
            # Add to perORg entry
            try:
                dicoDF['perOrg'][orgName][subtype].append(lstGenes)
            except KeyError:
                dicoDF['perOrg'][orgName][subtype] = [lstGenes]
            # Add to perLT entry
            for i in range(len(lstGenes)):
                try:
                    dicoDF['perLT'][orgName][lstGenes[i]][subtype] = lstGeneNames[i]
                except KeyError:
                    dicoDF['perLT'][orgName][lstGenes[i]] = {subtype:  lstGeneNames[i]}
            # Add to perSyst entry
            try:
                dicoDF['perSyst'][subtype][orgName].extend(lstGenes)
            except KeyError:
                try:
                    dicoDF['perSyst'][subtype][orgName] = lstGenes
                except KeyError:
                    dicoDF['perSyst'][subtype] = {orgName: lstGenes}
        # PADLOC
        pathPADLOCOUT = pathOUT+"/"+orgName+"_PADLOC_"+plmodelV+".tsv"
        for pdLine in read_file(pathPADLOCOUT)[1:]:
            splitLine = pdLine.split("\t")
            systNum = int(splitLine[0])
            syst = splitLine[2]
            gene = splitLine[3]
            geneName = splitLine[6]
            # Add to perORg entry
            try:
                dicoPL['perOrg'][orgName][syst][systNum].append(gene)
            except KeyError:
                dicoPL['perOrg'][orgName][syst] = {systNum: [gene]}
            # Add to perLT entry
            try:
                dicoPL['perLT'][orgName][gene][syst] = geneName
            except KeyError:
                dicoPL['perLT'][orgName][gene] = {syst:  geneName}
            # Add to perSyst entry
            try:
                dicoPL['perSyst'][syst][orgName][systNum].append(gene)
            except KeyError:
                try:
                    dicoPL['perSyst'][syst][orgName] = {systNum: [gene]}
                except KeyError:
                    dicoPL['perSyst'][syst] = {orgName: {systNum: [gene]}}
        # Merge perOrg systems
        for syst in dicoPL['perOrg'][orgName]:
            lstSyst = []
            for systNum in dicoPL['perOrg'][orgName][syst]:
                lstSyst.append(dicoPL['perOrg'][orgName][syst][systNum])
            dicoPL['perOrg'][orgName][syst] = lstSyst
    # Merge perSyst genes
    for syst in dicoPL['perSyst']:
        for orgName in dicoPL['perSyst'][syst]:
            lstGene = []
            for systNum in dicoPL['perSyst'][syst][orgName]:
                lstGene.append(dicoPL['perSyst'][syst][orgName][systNum])
            dicoPL['perSyst'][syst][orgName] = lstGene
    # Dump JSONs
    printcolor("♊ DUMP JSONS"+"\n")
    dump_json(dicoDF, pathOUT+"/DefenseFinder_"+dfmodelV+".json")
    dump_json(dicoPL, pathOUT+"/PADLOC_"+plmodelV+".json")


@fct_checker
def satellite_finder(pathIN: str, pathOUT: str, modelSel: str = "ALL", ext: str = ".faa") -> Tuple[str, str, str, str]:
    '''
     ------------------------------------------------------------
    |                  SEARCH PHAGES SATELLITES                  |
     ------------------------------------------------------------
    |       Search phage satellites using SatelliteFinder        |
     ------------------------------------------------------------
    |PARAMETERS                                                  |
    |    pathIN  : path of input files or folder (required)      |
    |    pathOUT : path of output folder (required)              |
    |    model   : SatelliteFinder model version (default=ALL)   |
    |    ext     : extension of input files (default=.faa)       |
     ------------------------------------------------------------
    | TOOLS: satellite_finder                                    |
     ------------------------------------------------------------
    model = ['ALL', 'cfPICI', 'PICI', 'P4', 'PLE']
    No "#" in fasta headers and required Macsyfinder v2.0
    '''
    pathOUT = path_converter(pathOUT)
    lstFiles, maxpathSize = get_input_files(pathIN, "satellite_finder", [ext])
    if len(lstFiles) == 0:
        printcolor("[ERROR: satellite_finder]\nAny input files found, check extension\n", 1, "212;64;89", "None", True)
        exit_gemini()
    # Look for available models
    dicoGeminiPath, dicoGeminiModule = get_gemini_path()
    if 'satellite_finder' in dicoGeminiModule:
        os.system("module load "+dicoGeminiModule['satellite_finder'])
    lstModels = [sub.replace(".xml", "") for sub in os.listdir(dicoGeminiPath['TOOLS']['satellite_finder']+"/models/SatelliteFinder/definitions/")]
    if modelSel not in lstModels+["ALL"]:
        printcolor("[ERROR: satellite_finder]\nInvalid model \""+modelSel+"\". Availables = "+str(lstModels+["ALL"])+"\n", 1, "212;64;89", "None", True)
        exit_gemini()
    os.makedirs(pathOUT, exist_ok=True)
    os.makedirs(pathOUT+"/raw", exist_ok=True)
    slurmBool, cpu, memMax, memMin = get_sys_info()
    pathLOG = pathOUT+"/gemini.log"
    # ***** SatelliteFinder ***** #
    dicoResults = {}
    printcolor("♊ SatelliteFinder"+"\n")
    if modelSel == "ALL":
        lstSelectedModels = lstModels
    else:
        lstSelectedModels = [modelSel]
    for model in lstSelectedModels:
        dicoResults[model] = {'header': "", 'lines': ""}
    pbar = tqdm(total=len(lstFiles), dynamic_ncols=True, ncols=50+maxpathSize, leave=False, desc="", file=sys.stdout, bar_format="  {percentage: 3.0f}%|{bar}| {n_fmt}/{total_fmt} [{desc}]")
    for pathFAA in lstFiles:
        file = os.path.basename(pathFAA)
        orgName = file.replace(ext, "").replace("."+ext, "")
        pbar.set_description_str(orgName+" ".rjust(maxpathSize-len(orgName)))
        pathTMPFAA = geminiset.pathTMP+"/"+orgName+".faa"
        os.system("sed s/\"#\"/\" \"/g "+pathFAA+" > "+pathTMPFAA)
        # Launch macsyfinder for each model
        cpt = 1
        for model in lstSelectedModels:
            pbar.set_description_str(orgName+" ("+str(cpt)+"/"+str(len(lstSelectedModels))+")"+" ".rjust(maxpathSize-len(orgName)))
            pathModelOUT = pathOUT+"/raw/"+orgName+"____"+model+".tsv"
            if not os.path.isfile(pathModelOUT):
                cmdSatelliteFinder = dicoGeminiPath['TOOLS']["python"]+" "+dicoGeminiPath['TOOLS']['satellite_finder'] + "/bin/satellite_finder.py" + \
                    " --models " + model + " --sequence-db " + pathTMPFAA + " --db-type ordered_replicon" + \
                    " --out-dir " + geminiset.pathTMP + "/" + model + " --worker " + str(cpu) + " --idx" + " >> " + pathLOG + " 2>&1"
                os.system(cmdSatelliteFinder)
                pathCSV = glob.glob(geminiset.pathTMP+"/"+model+"/*.csv")[0]
                df = pandas.read_csv(pathCSV, escapechar='\n')
                # Drop first column
                df.drop(columns=df.columns[0], axis=1, inplace=True)
                df.to_csv(pathModelOUT, sep='\t', encoding='utf-8', index=False)
                shutil.rmtree(geminiset.pathTMP+"/"+model)
            # Parse
            lstLines = read_file(pathModelOUT)
            dicoResults[model]['header'] = lstLines[0]
            for line in lstLines[1:]:
                dicoResults[model]['lines'] += line.replace("UserReplicon.", "").replace("UserReplicon_", "").replace("UserReplicon", orgName)+"\n"
            cpt += 1
        title("SatelliteFinder", pbar)
        pbar.update(1)
    pbar.close()
    # ***** Write final tsv ***** #
    for model in lstSelectedModels:
        if dicoResults[model]['lines'] != "":
            pathTSVOUT = pathOUT+"/"+model+".tsv"
            TSV = open(pathTSVOUT, 'w')
            TSV.write(dicoResults[model]['header']+"\n")
            TSV.write(dicoResults[model]['lines'])
            TSV.close()
