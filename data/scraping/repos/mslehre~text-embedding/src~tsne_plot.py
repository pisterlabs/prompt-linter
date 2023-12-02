#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn  as sns
from adjustText import adjust_text
from openai.embeddings_utils import cosine_similarity


def try_to_read_file(file_path: str) -> str:
    # a quick function for argparse to check if given arg is a readable file
    if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
        raise argparse.ArgumentTypeError("The file " + file_path + "does not "
                                         + "exist or is not readable!")
    return file_path

def check_thinning_args(thinning_arg: str,
                        dir_path: str) -> tuple[int, str]:
    # function to check arguments for thinning data

    # check if thinning_arg is an int
    try:
        thinning_arg = int(thinning_arg)
    except ValueError as ve:
        raise argparse.ArgumentTypeError("Argument " + thinning_arg + " is not"
                                         + " an integer!") 

    # check if dir_path is a directory
    if not os.path.isdir(dir_path) or not os.access(dir_path, os.R_OK):
        raise argparse.ArgumentTypeError("The directory " + dir_path + " does "
                                         + "not exist or is not readable!")
    dir_path = os.path.join(dir_path, '')  # append '/' 

    return thinning_arg, dir_path


def count_pubs(author_ids: pd.DataFrame,
               dir_path: str) -> list[int]:
    """
    Count number of publications for each provided author ID in the specified 
    directory.

    Args:
        author_ids (pandas.DataFrame): Author IDs to count the number of 
            publications for.
        dir_path (str): Directory containing the publication lists, the file 
            names need to be <author_id>.txt.

    Returns:
        num_pubs (list): List of number of publications for each author ID. 
            A -1 is appended if no publication list was found for the author.
    """

    num_pubs = []
    for id in author_ids[0]:  
        file_path = dir_path + str(id) + ".txt"
        n = -1  # num pubs in file, -1 because the first line is not a pub
        # check if file exists and is readable, if yes read
        if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip(): n += 1  # if line is not empty, count + 1
            num_pubs.append(n)  # add count for author id
        else:
            print("WARNING: Can't find or read file ", file_path, 
                  " for author ID ", id, "!")
            num_pubs.append(n)  # add -1 pubs for authors with no file
    
    return num_pubs


def thin_out_data(author_ids: pd.DataFrame,
                  embeddings: pd.DataFrame,
                  dir_path: str,
                  min_pubs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 
    Remove data of authors with less than min_pubs publications from author_ids
    and embeddings.

    Args:
        author_ids (pandas.DataFrame): Author IDs of authors to thin out.
        embeddings (pandas.DataFrame): Embeddings corresponding to the author
            IDs to thin out.
        dir_path (str): Path to directory containing the publication lists.
        min_pubs (int): Minimum number of publications. Every author with less 
            than min_pubs publications is being pruned from author_ids and 
            embeddings.

    Returns:
        author_ids (pandas.DataFrame): Pruned author IDs.
        embeddings (pandas.DataFrame): Pruned embeddings.
    """
    num_pubs = count_pubs(author_ids, dir_path)

    # get indices of authors with less than min_pubs pubs
    drop_indices =[i for i, num in enumerate(num_pubs) if num < min_pubs]
    # drop those authors
    embeddings = embeddings.drop(drop_indices)
    author_ids = author_ids.drop(drop_indices)

    return author_ids, embeddings


def get_author_info_and_palette(authors: pd.DataFrame,
                                author_ids: np.ndarray,
                                affiliation_map: pd.DataFrame, 
                                affiliation: str) -> tuple[list[str], dict]:
    """
    Get affiliations of autors by ID and generate color palette for the
    affiliations

    Args:
        authors (pandas.DataFrame): Table containing at least author ids and 
            their faculties and institutes.
        author_ids (numpay.ndarray): List of author ids for which to get the 
            affiliations.
        affiliation_map (pandas.DataFrame): Table with all possible 
            faculties or institutes.
        affiliation (str): Which affiliation to use - "faculty" or "institute".

    Returns:
        lnames (list[str]): List of authors last names.
        affil (list[str]): List of affiliations for author IDs.
        pal (dict): Color palette for the affiliations.
    """

    # get last names of authors
    lnames = authors.loc[authors['id'].isin(author_ids), 'lastname'].to_list()

    # get affiliations of authors by ID
    affil = authors.loc[authors['id'].isin(author_ids), affiliation].to_list()
    
    # switch long names of institutions to short names
    if affiliation == 'institute':
        mapping = dict(zip(affiliation_map['institute_long'], 
                           affiliation_map['institute_short']))
        affil_ = [mapping[item] for item in affil]
        affil = affil_
        affil_uniq = affiliation_map['institute_short'].to_list()
    else:
        affil_uniq = affiliation_map['faculty'].to_list()
    
    # generate color palette
    num_col = len(affil_uniq)  # number of colors
    colors = sns.color_palette("hls", num_col).as_hex()  # get colors
    pal = dict(zip(affil_uniq, colors))  # color palette for plot

    return lnames, affil, pal


def compute_tsne(X: np.ndarray, 
                 pca_reduction: bool = False, 
                 pca_components: int = 50,
                 tsne_perplexity: float = 30.0) -> np.ndarray:
    """ 
    Compute t-SNE for embeddings 
    
    Args:
        X (numpy.ndarray): Embeddings to transform
        pca_reduction (bool): If True, a PCA is performed before the t-SNE to 
            reduce computation resources. Default: True.
        pca_components (int): Number of components to keep, when PCA is 
            performed, i.e. dimension of the result. Default: 50.
        tsne_perplexity (float): Number of nearest neighbors that is used in 
            other manifold learning algorithms, must be less than the number of
            samples. Default: 30.0.

    Returns:
        tsne_result (numpy.ndarray): t-SNE transformed data.
    """
    # ada dimensions: 1536
    if pca_reduction:
        # build PCA, reduce to dim size pca_components
        pca = PCA(n_components = pca_components)
        pca_result = pca.fit_transform(X)  # fit model and apply dim reduction
        X = pca_result
    tsne = TSNE(perplexity = tsne_perplexity)  # perplexity = knn
    tsne_result = tsne.fit_transform(X)  # fit model and apply dim reduction
    return tsne_result

def compute_cosinesim(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for embeddings

    Args:
        embeddings (numpy.ndarray): A 2-dim array with embeddings

    Returns:
        cosine_sims (numpy.ndarray): A symmetric matrix containing pairwise
            cosine similarities of the embeddings and Nan on the diagonal.
    """
    n = embeddings.shape[0]  # number of embeddings
    # n x n matrix for pairwise distances
    cosine_sims = np.zeros((n,n), dtype = np.float64)
    # indices of upper triangle matrix with offset 1
    iupper = np.stack(np.triu_indices(n, 1), axis = -1)
    # compute cosine similarity
    for i,j in iupper:
        cosine_sims[i,j] = cosine_similarity(embeddings[i], embeddings[j])

    cosine_sims += cosine_sims.T  # make matrix symmetric
    np.fill_diagonal(cosine_sims, np.nan)  # fill diagonal with NaN

    return cosine_sims

def get_edges(similarities: np.ndarray,
              k: int = None,
              threshold: float = None) -> set[tuple[int,int]]:
    """
    Given a matrix containing pairwise similarity scores, for each element get 
    the indices for the k highest scores and/or the indices for the elements
    with a score of >= threshold.

    Args:
        similarities (np.ndarray): A symmetric matrix containing pairwise
            similarities and Nan on the diagonal.   
        k (int): For each element, get the index pairs for the k elements with
            the highest similarity score.
        threshold (float): Get index pairs with a similarity score of >=
            threshold.

    Returns:
        edges (set[tuple[int,int]]): A set of 0-based index tuples computed 
            according to k/threshold.
    """
    if not k and not threshold:
        print("ERROR: You need to specify at least one criterium for selecting"
              + " the edges!")
        exit(1)

    edges = []

    # get edges to the k nearest neightbours for each node
    if k:
        indices = np.argsort(similarities)
        # get knn -> adjacency lists
        # argsort sorts ascending, need the last elements
        knn = indices[:, -(k+1):-1]  # nans get sorted last, exclude them
        for i,k in zip(range(len(knn)), knn):
            for j in k:
                # sort indices so duplicates can be identified by set()
                ind = sorted([i,j])
                edges.append(tuple(ind))

    # get edges for nodes with a similarity of >= threshold
    if threshold:
        indices = np.stack(np.where(similarities >= threshold), axis = -1)
        for i in indices:
            ind = sorted(i)
            edges.append(tuple(ind))

    edges = set(edges)   # duplicate edges are removed

    return edges

def tsne_plot(X: np.ndarray,
              lnames: list[str],
              affiliation: list[str],
              legend_title: str,
              palette: dict,
              edges = None):
    """
    Plot t-SNE 

    Args:
        X (numpy.ndarray): Result of the t-SNE transformation.
        affiliation (list[str]): Containing the institutes or 
            faculties corresponding to the data points. This decides how to 
            color the points in the plot. 
        legend_title (str): Title for the plot legend
        palette (dict): Color palette for the plot. Specifies which color to 
            use for which institute or faculty.

    Returns:
        matplotlib.figure.Figure: Figure of the plot.
    """

    plt.figure(figsize=(15,15))
    ax = sns.scatterplot(
        x = X[:, 0], y = X[:, 1],
        hue = affiliation,
        hue_order = list(palette.keys()),
        palette = palette,
        style = affiliation,
        style_order = list(palette.keys()),
        legend = "full",
        alpha = 1,
        s=200,
        zorder = 5
    )
    
    plt.title("t-SNE Plot of Publication Lists", fontsize = 20)  # plot title
    # adjust legend position and style
    ax.legend(fancybox=True, ncol = 2, fontsize = 14, title = legend_title)
    sns.move_legend(ax, "upper right", bbox_to_anchor=(-0.05, 1))
    
    # add edges to plot
    if edges:
        for i, j in edges:
            # add line from i to j 
            x1 = X[i,0]
            y1 = X[i,1]
            x2 = X[j,0] - x1
            y2 = X[j,1] - y1
            plt.arrow(x1, y1, x2, y2, 
                      color='gray', linewidth=1, length_includes_head=True, 
                      head_width=0, alpha = 0.2, zorder = 0)
            # for  some reason plt.plot doesnt work well with scatter plots
            #plt.plot(X[i, :], X[j, :], marker = None, linewidth = 2, 
            #        color = "gray", alpha = 0.1)  
    
    # annotate dots with last names
    text = []
    # box style for labels
    bbox_props = dict(boxstyle="round", fc="white", alpha=0.3) 
    #boxstyle="round, pad=0.15"
    for i, label in enumerate(lnames):        
        text += [ax.text(X[i, 0], X[i, 1], label, bbox=bbox_props, 
                 zorder = 10)]  # fontsize = 'x-small'
    adjust_text(text)  # prevent labels from overlapping

    return ax.get_figure()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize the embeddings of publications with a t-SNE '
                    + 'plot.')
    parser.add_argument('embed_file', type = try_to_read_file,
                        help = 'hdf5 file with embeddings.')
    parser.add_argument('author_file', type = try_to_read_file,
                        help = 'File with table containing information about '
                        + 'the authors, like ID, faculty, and institute.')
    parser.add_argument('affiliation_map', type = try_to_read_file,
                        help = 'File with table containig all possible '
                        + 'faculties/institutes.')
    parser.add_argument('-o', '--outfile', default = 'tsne_plot',
                        help = 'Stem for output file to save plot. Default is '
                        + '\"tsne_plot\".')
    parser.add_argument('--format', default = 'pdf', 
                        choices = ['png', 'pdf', 'svg'],
                        help = 'Format for plot. Default is png.')
    parser.add_argument('--tsne_perplexity', type = float, default = 30.0,
                        help = 'Perplexity for the t-SNE algorithm. Default is'
                        + ' 30.0.')                    
    parser.add_argument('--pca', action = 'store_true',
                        help = 'Perform a PCA before the t-SNE.')
    parser.add_argument('--pca_components', type = int, default = 50,
                        help = 'Number of components to keep after performing'
                        + ' the PCA. Default is 50.')
    parser.add_argument('--affiliation', default = 'institute', 
                        choices = ['institute', 'faculty'],
                        help = 'Decides after which fashion to color the ' 
                        + 'plot. Default is \"institute\".')
    parser.add_argument('-k', '--k_edges', type = int,
                        help = 'For each author, plot edges to the authors '
                        + 'with the k highest cosine similarities.')
    parser.add_argument('-t', '--threshold_edges', type = float,
                        help='For each author, plot edges to the authors '
                        + 'with a cosine similarity of >= threshold.')
    parser.add_argument('--thinning', nargs = 2, metavar = ('INT', 'DIR'),
                        help = 'Prune author data by number of publications. '
                        + 'Specify a minimum number of publications and the '
                        + 'directory containing the publication lists. Authors'
                        + ' with less than the required number of publications'
                        + ' will not be plotted.') 
    args = parser.parse_args()
    
    # check thinning args
    if args.thinning:
        min_pubs, pub_dir = check_thinning_args(args.thinning[0], 
                                                args.thinning[1])

    outfile = args.outfile + '.' + args.format

    # read data
    hdf = pd.HDFStore(args.embed_file, mode='r')
    embeddings = pd.read_hdf(hdf, "embeddings") 
    author_ids = pd.read_hdf(hdf, "ids")
    hdf.close()
    authors = pd.read_table(args.author_file, delimiter = '\t', dtype = str)
    # encoding='latin1'
    affiliation_map = pd.read_table(args.affiliation_map, delimiter = '\t')

    # thinning data 
    if args.thinning:
        author_ids, embeddings = thin_out_data(author_ids, embeddings, 
                                               pub_dir, min_pubs) 

    # convert data to numpy arrays for further steps
    author_ids = author_ids.to_numpy().flatten()
    embeddings = embeddings.to_numpy(dtype = np.float64)

    # perplexity for tsne needs to be smaller than the number of samples
    k = args.tsne_perplexity \
        if len(author_ids) > args.tsne_perplexity \
        else float(len(author_ids) - 1)
    # pca components needs to be <= min(n_samples, n_features)
    pca_components = args.pca_components \
        if min(embeddings.shape) >= args.pca_components \
        else min(embeddings.shape)
    # transform embeddings
    tsne_result = compute_tsne(embeddings, pca_reduction = args.pca, 
                               pca_components = pca_components, 
                               tsne_perplexity = k)
    # get last names, affiliations, and color palette
    lnames, affiliation, palette = get_author_info_and_palette(
        authors, 
        author_ids, 
        affiliation_map, 
        args.affiliation)
    # get edges
    edges = None
    if args.k_edges or args.threshold_edges:
        similarities = compute_cosinesim(embeddings)
        edges = get_edges(similarities, args.k_edges, args.threshold_edges)
    # plot
    fig = tsne_plot(tsne_result, lnames, affiliation, args.affiliation, 
                    palette, edges)  
    fig.savefig(outfile, format = args.format, bbox_inches='tight')
    plt.show()
       
    exit(0)


if __name__ == "__main__":
    main()
