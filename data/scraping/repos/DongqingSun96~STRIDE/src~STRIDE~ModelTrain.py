# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-07 22:11:05
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-09-01 13:55:22


import os
import scipy
import gensim
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp_sparse

from sklearn.preprocessing import StandardScaler
from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary

from STRIDE.utility.IO import read_10X_h5, read_count, write_10X_h5
from STRIDE.ModelEvaluate import *


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def scProcess(sc_count_file, sc_anno_file, out_dir, out_prefix, sc_scale_factor = None):
    # read scRNA-seq data
    if sc_count_file.endswith(".h5"):    
        sc_count = read_10X_h5(sc_count_file)
        sc_count_mat = sc_count.matrix
        sc_count_genes = sc_count.names.tolist()
        sc_count_cells = sc_count.barcodes.tolist()
        if type(sc_count_genes[0]) == bytes:
            sc_count_genes = [i.decode() for i in sc_count_genes]
        if type(sc_count_cells[0]) == bytes:
            sc_count_cells = [i.decode() for i in sc_count_cells]
        h5_filename = sc_count_file
    else:
        sc_count = read_count(sc_count_file)
        sc_count_mat = sc_count["matrix"]
        sc_count_mat = sp_sparse.csc_matrix(sc_count_mat, dtype=np.float32)
        sc_count_genes = sc_count["features"]
        sc_count_cells = sc_count["barcodes"]
        h5_filename = os.path.join(out_dir, "%s_scRNA_count.h5" %(out_prefix))
        write_10X_h5(filename = h5_filename, matrix = sc_count_mat,
                     features = sc_count_genes, barcodes = sc_count_cells)

    # scale the count matrix
    count_per_cell = np.asarray(sc_count_mat.sum(axis=0))
    count_per_cell = np.array(count_per_cell.tolist()[0])
    if not sc_scale_factor:
        sc_scale_factor = np.round(np.quantile(count_per_cell, 0.75)/1000, 0)*1000
    r,c = sc_count_mat.nonzero()
    count_per_cell_sp = sp_sparse.csr_matrix(((1.0/count_per_cell)[c], (r,c)), shape=(sc_count_mat.shape))
    sc_count_scale_mat = sc_count_mat.multiply(count_per_cell_sp)*sc_scale_factor
    sc_count_scale_mat = sp_sparse.csc_matrix(sc_count_scale_mat)
    # read cell-type meta file
    cell_celltype_dict = {}
    for line in open(sc_anno_file, "r"):
        items = line.strip().split("\t")
        cell_celltype_dict[items[0]] = items[1]

    return({'scale_matrix': sc_count_scale_mat, "raw_matrix": sc_count_mat,
        "genes": sc_count_genes, "cells": sc_count_cells, "cell_celltype": cell_celltype_dict})


def stProcess(st_count_file, st_scale_factor = None):
    # read spatial count file
    if st_count_file.endswith(".h5"):    
        st_count = read_10X_h5(st_count_file)
        st_count_mat = st_count.matrix
        st_count_genes = st_count.names.tolist()
        st_count_spots = st_count.barcodes.tolist()
        if type(st_count_genes[0]) == bytes:
            st_count_genes = [i.decode() for i in st_count_genes]
        if type(st_count_spots[0]) == bytes:
            st_count_spots = [i.decode() for i in st_count_spots]
    else:
        st_count = read_count(st_count_file)
        st_count_mat = st_count["matrix"]
        st_count_mat = sp_sparse.csc_matrix(st_count_mat, dtype=np.float32)
        st_count_genes = st_count["features"]
        st_count_spots = st_count["barcodes"]
    # scale the count matrix
    count_per_spot = np.asarray(st_count_mat.sum(axis=0))
    count_per_spot = np.array(count_per_spot.tolist()[0])
    if not st_scale_factor:
        st_scale_factor = np.round(np.quantile(count_per_spot, 0.75)/1000, 0)*1000
    r,c = st_count_mat.nonzero()
    count_per_spot_sp = sp_sparse.csr_matrix(((1.0/count_per_spot)[c], (r,c)), shape=(st_count_mat.shape))
    st_count_scale_mat = st_count_mat.multiply(count_per_spot_sp)*st_scale_factor
    st_count_scale_mat = sp_sparse.csc_matrix(st_count_scale_mat)

    return({'scale_matrix': st_count_scale_mat, "raw_matrix": st_count_mat,"genes": st_count_genes, "spots": st_count_spots})


def LDA(sc_corpus, ntopics, genes_dict, genes_shared, cell_gene_list, sc_count_cells, cell_celltype_list, model_dir):
    lda = LdaModel(corpus = sc_corpus, num_topics = ntopics, id2word = genes_dict)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_file = os.path.join(model_dir, "lda_model_%s" %(ntopics))
    lda.save(model_file)
    # compute the coherence
    cm = CoherenceModel(model = lda, corpus = sc_corpus, coherence='u_mass')
    umass_coherence = cm.get_coherence()
    cm = CoherenceModel(model = lda, corpus = sc_corpus, texts = cell_gene_list, coherence='c_v')
    cv_coherence = cm.get_coherence()
    # save the topic-cell matrix
    topic_cell = lda.get_document_topics(sc_corpus)
    topic_cell_mat = gensim.matutils.corpus2csc(topic_cell)
    topic_cell_file = os.path.join(model_dir, "topic_cell_mat_%s.npz" %(ntopics))
    topic_cell_df_file = os.path.join(model_dir, "topic_cell_mat_%s.txt" %(ntopics))
    scipy.sparse.save_npz(topic_cell_file, topic_cell_mat)
    topic_cell_df = pd.DataFrame(topic_cell_mat.todense(), 
        index = ["Topic %s" %i for i in range(1, 1 + topic_cell_mat.shape[0])], 
        columns = sc_count_cells)
    topic_cell_df.to_csv(topic_cell_df_file, sep = "\t", index = True, header = True)
    # save the gene-topic matrix
    topic_gene_mat_list = lda.get_topics()
    topic_gene_mat = np.array(topic_gene_mat_list)
    gene_topic_mat = topic_gene_mat.transpose()
    gene_topic_mat_list = gene_topic_mat.tolist()
    gene_topic_file = os.path.join(model_dir, "gene_topic_mat_%s.txt" %(ntopics))
    gene_topic_out = open(gene_topic_file, "w")
    gene_topic_out.write("\t".join(["Topic%s" %i for i in range(1, ntopics + 1)]) + "\n")
    for i in range(len(gene_topic_mat_list)):
        gene_topic_out.write(genes_shared[i] + "\t" + "\t".join([str(j) for j in gene_topic_mat_list[i]]) + "\n")
    gene_topic_out.close()
    # convert topic_cell_mat to topic_celltype_mat
    celltype_topic_dict = {}
    celltype_num_dict = {}
    celltypes = sorted(list(set(cell_celltype_list)))
    for celltype in celltypes:
        celltype_topic_dict[celltype] = [0]*ntopics
        celltype_num_dict[celltype] = 0
    for i in range(topic_cell_mat.shape[1]):
        cell_celltype = cell_celltype_list[i]
        celltype_topic_dict[cell_celltype] = [celltype_topic_dict[cell_celltype][j] + topic_cell_mat[j,i] for j in range(topic_cell_mat.shape[0])]
        celltype_num_dict[cell_celltype] = celltype_num_dict[cell_celltype] + 1
    celltype_topic_mean_dict = {}
    for celltype in celltypes:
        celltype_topic_mean_dict[celltype] = [i/celltype_num_dict[celltype] for i in celltype_topic_dict[celltype]]
    topic_celltype_df = pd.DataFrame(data = celltype_topic_mean_dict)
    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat_%s.txt" %(ntopics))
    topic_celltype_df.to_csv(topic_celltype_file, sep="\t")
    # return results
    res_dict = {"coherence": [umass_coherence, cv_coherence], 
    "topic_cell_mat": topic_cell_mat, 
    "topic_celltype_df": topic_celltype_df,
    "celltype_num_dict": celltype_num_dict}

    return(res_dict)


def ModelEvaluate(sc_corpus, genes_dict, genes_shared, ntopics_list, cell_gene_list, sc_count_cells, cell_celltype_list, model_dir):
    umass_coherence_values = []
    cv_coherence_values = []
    accuracy_raw_list = []
    nmi_list = []
    accuracy_norm_list = []
    accuracy_normbysd_list = []
    accuracy_bayes_list = []
    accuracy_bayesnorm_list = []
    celltype_prediction_dict = defaultdict(dict)
    for ntopics in ntopics_list:
        print("Number of topics: %s" %(ntopics))
        res_dict = LDA(sc_corpus, ntopics, genes_dict, genes_shared, cell_gene_list, sc_count_cells, cell_celltype_list, model_dir)
        umass_coherence_values.append(res_dict["coherence"][0])
        cv_coherence_values.append(res_dict["coherence"][1])
        topic_cell_mat = res_dict["topic_cell_mat"]
        topic_celltype_df = res_dict["topic_celltype_df"]
        celltype_num_dict = res_dict["celltype_num_dict"]
        raw_res = ModelEvaluateRaw(topic_cell_mat, topic_celltype_df, cell_celltype_list)
        nmi_list.append(raw_res["nmi"])
        accuracy_raw_list.append(raw_res["accuracy"])
        celltype_prediction_dict["Raw"][ntopics] = raw_res["celltype_prediction"]
        norm_res = ModelEvaluateNorm(topic_cell_mat, topic_celltype_df, cell_celltype_list)
        accuracy_norm_list.append(norm_res["accuracy"])
        celltype_prediction_dict["Norm"][ntopics] = norm_res["celltype_prediction"]
        normbysd_res = ModelEvaluateNormBySD(topic_cell_mat, topic_celltype_df, cell_celltype_list)
        accuracy_normbysd_list.append(normbysd_res["accuracy"])
        celltype_prediction_dict["NormBySD"][ntopics] = normbysd_res["celltype_prediction"]
        bayes_res = ModelEvaluateBayes(topic_cell_mat, topic_celltype_df, cell_celltype_list, celltype_num_dict, model_dir)
        accuracy_bayes_list.append(bayes_res["accuracy"])
        celltype_prediction_dict["Bayes"][ntopics] = bayes_res["celltype_prediction"]
        celltype_topic_bayes_df = bayes_res["celltype_topic_bayes_df"]
        bayesnorm_res = ModelEvaluateBayesNorm(topic_cell_mat, celltype_topic_bayes_df, cell_celltype_list, celltype_num_dict)
        accuracy_bayesnorm_list.append(bayesnorm_res["accuracy"])
        celltype_prediction_dict["BayesNorm"][ntopics] = bayesnorm_res["celltype_prediction"]
    metrics_df = pd.DataFrame({"Topic": ntopics_list, 
        "umass_coherence": umass_coherence_values, 
        "cv_coherence": cv_coherence_values, 
        "nmi": nmi_list,
        "Raw_accuracy": accuracy_raw_list,
        "Norm_accuracy": accuracy_norm_list,
        "NormBySD_accuracy": accuracy_normbysd_list,
        "Bayes_accuracy": accuracy_bayes_list,
        "BayesNorm_accuracy": accuracy_bayesnorm_list})

    return({"metrics_df": metrics_df, "celltype_prediction_dict": celltype_prediction_dict})


def ModelSelect(sc_corpus, genes_dict, genes_shared, ntopics_list, cell_gene_list, sc_count_cells, cell_celltype_list, out_dir):
    model_dir = os.path.join(out_dir, "model")
    evaluate_res = ModelEvaluate(sc_corpus, genes_dict, genes_shared, ntopics_list, cell_gene_list, sc_count_cells, cell_celltype_list, model_dir)
    celltype_prediction_dict = evaluate_res["celltype_prediction_dict"]
    metrics_df = evaluate_res["metrics_df"]
    metrics_df.to_csv(os.path.join(out_dir, "Model_selection.txt"), sep="\t", index = False)
    # select the model and the optimal topic number
    # model_selected = metrics_df.iloc[:, range(4,9)].max().idxmax()
    model_selected = metrics_df.iloc[:, range(4,9)].mean(0).idxmax()
    ntopic_selected = metrics_df.iloc[metrics_df.loc[:, model_selected].idxmax(), 0]
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(6, 4))
    ax = sns.barplot(x="Topic", y=model_selected, data=metrics_df, ci = None, color = "#3C5488FF")
    ax.set(xlabel='Number of topics', ylabel='Accuracy')
    ax.grid(linewidth = 0.7, color = "#DBDBDB")
    plt.savefig(os.path.join(out_dir,"Model_accuracy_plot.pdf"), dpi = 300, bbox_inches = "tight")
    plt.close(f)
    plt.clf()
    # compare between the prediction and truth
    celltype_prediction = celltype_prediction_dict[model_selected.split("_")[0]][ntopic_selected]
    celltype_pair_df = pd.DataFrame({"Prediction":celltype_prediction, "Truth":cell_celltype_list})
    celltype_pair_df = celltype_pair_df.groupby(["Prediction", "Truth"]).size().reset_index(name='counts')
    celltype_pair_table = pd.pivot_table(celltype_pair_df, values='counts', index=['Prediction'],
                        columns=['Truth'])
    celltype_pair_table = celltype_pair_table.fillna(0)
    celltype_pair_table = celltype_pair_table.astype('int64')
    celltype_pair_table_sum_by_col = np.array(np.sum(celltype_pair_table, axis = 0))
    celltype_pair_table_norm = np.divide(celltype_pair_table,celltype_pair_table_sum_by_col)
    f, ax = plt.subplots(figsize=(8, 6.6))
    ax = sns.heatmap(celltype_pair_table_norm, cmap="YlGnBu", annot = celltype_pair_table, fmt = "d",linewidths=.5)
    plt.savefig(os.path.join(out_dir,"Prediction_truth_heatmap_%s_%s.pdf" %(model_selected.split("_")[0], ntopic_selected)), dpi = 300, bbox_inches = "tight")
    plt.close(f)
    plt.clf()

    return({"model":model_selected.split("_")[0], "ntopics":ntopic_selected})


def ModelRetrieve(model_dir, st_count_genes):
    metrics_df = pd.read_csv(os.path.join(model_dir, "..", "Model_selection.txt"), sep="\t", index_col=False, header=0)
    # select the model and the optimal topic number
    # model_selected = metrics_df.iloc[:, range(4,9)].max().idxmax()
    model_selected = metrics_df.iloc[:, range(4,9)].mean(0).idxmax()
    ntopic_selected = metrics_df.iloc[metrics_df.loc[:, model_selected].idxmax(), 0]
    genes_dict_file = os.path.join(model_dir, "..", "Gene_dict.txt")
    genes_dict = Dictionary.load_from_text(genes_dict_file)

    return({"model":model_selected.split("_")[0], "ntopics":ntopic_selected, "genes_dict":genes_dict})


def scLDA(sc_count_mat, sc_count_genes, sc_count_cells, cell_celltype_dict,
          st_count_mat, st_count_genes, st_count_spots,
          normalize, gene_use, ntopics_list, out_dir):
    sc_count_genes_array = np.array(sc_count_genes)
    sc_count_genes_sorter = np.argsort(sc_count_genes_array)
    if normalize:
        sc_count_mat = StandardScaler(with_mean=False).fit_transform(sc_count_mat.transpose()).transpose()
    if gene_use == "All":
        genes_shared = list(set(st_count_genes) & set(sc_count_genes))
    else:
        genes_shared = list(set(st_count_genes) & set(sc_count_genes) & set(gene_use))
    genes_shared = sorted(genes_shared)
    genes_shared_array = np.array(genes_shared)
    genes_shared_index = sc_count_genes_sorter[np.searchsorted(sc_count_genes_array, genes_shared_array, sorter = sc_count_genes_sorter)]
    sc_count_mat_use = sc_count_mat[genes_shared_index,:]
    cell_gene_list = []
    sc_count_mat_use_nonzero = sc_count_mat_use.nonzero()
    for i in range(sc_count_mat_use.shape[1]):
        gene_ind = sc_count_mat_use_nonzero[0][sc_count_mat_use_nonzero[1] == i]
        genes = genes_shared_array[gene_ind].tolist()
        cell_gene_list.append(genes)
        # evaluate the model
    # construct single-cell gene corpus
    sc_corpus = gensim.matutils.Sparse2Corpus(sc_count_mat_use)
    genes_dict = Dictionary([genes_shared])
    genes_dict_file = os.path.join(out_dir, "Gene_dict.txt")
    genes_dict.save_as_text(genes_dict_file)
    cell_celltype_list = []
    for i in range(len(sc_count_cells)):
        cell_celltype = cell_celltype_dict[sc_count_cells[i]]
        cell_celltype_list.append(cell_celltype)
    print("Selecting the optimal model.")
    model_selection_res = ModelSelect(sc_corpus = sc_corpus, genes_dict = genes_dict, genes_shared = genes_shared,
        ntopics_list = ntopics_list, cell_gene_list = cell_gene_list, sc_count_cells = sc_count_cells, 
        cell_celltype_list = cell_celltype_list, out_dir = out_dir)
    model_selected = model_selection_res["model"]
    ntopics_selected = model_selection_res["ntopics"]

    return({"genes_dict": genes_dict, "model_selected": model_selected, "ntopics_selected": ntopics_selected})

