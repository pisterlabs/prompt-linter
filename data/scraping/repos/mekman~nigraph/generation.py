#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Commonly used utility functions."""

# mainly backports from future numpy here

from __future__ import absolute_import, division, print_function
from copy import deepcopy
import numpy as np
import networkx as nx
from scipy.stats import spearmanr

from .utilities import tril_indices, triu_indices, convert_to_graph, \
    fill_diagonal


def adj_static(ts, measure='corr', pval=False, TR=2, fq_l=None, fq_u=None,
               order=1, scale=2, w=None, idp=None, excl_zero_cov=False):
    """returns a *static* graph representation (adjacency matrix)

    Parameters
    ----------
    ts : ndarray, shape(n_rois, n_tps)
        Pre-processed timeseries information
    measure : string
        Similarity measure for the adjacency matrix calculation.

        * ``corr``: Pearson product momement correlation coefficient [-1,1].
        * ``cov``: Covariance.
        * ``coh``: Coherence [0,1]
        * ``delay``: Coherence phase delay estimates. The metric was used in
          [3]_ and [4]_
        * ``granger``: Granger-causality. As suggested by [3]_ and [4]_ this
          returns the difference `F(x-->y)` and `F(y-->x)`.
        * ``wtc``: Correlation of the wavelet coefficients [-1,1]. The metric
          was used in [1]_
        * ``pcorr``: Partial correlation. Calculated using the inverse
          covariance matrix (precision matrix) [0,1]
        * ``pcoh``: Partial coherence in the range [0,1]. The metric was used
          in [2]_ #not impl yet
        * ``spcoh`` : Semi-partial coherence in the range [0,1]. The metric was
          used in [7]_. The correlation between two time-series is conditioned
          on a third time-series given by ``idp``.
        * ``ktau``: Kendall's tau, a correlation measure for ordinal data.
        * ``rho``: Spearman rank-order correlation coefficient rho [-1,1]. This
          is a nonparametric measure of the linear relationship between two
          datasets. Unlike e.g. the Pearson correlation, the Spearman
          correlation does not assume that both datasets are normally
          distributed.
        * ``mic``: Maximal information criterion [0,1]
        * ``non_linearity``: Non-linearity of the relationship [0,1]
        * ``mi``: Mutual information. The metric was used in [5]_
        * ``nmi``: Normalized mutual information [0,1]
        * ``ami``: Adjusted mutual information [0,1]
        * ``cmi`` Conditional mutual information. The metric was used in [6]_
          # not impl yet
        * ``graph_lasso``: Sparse inverse covariance matrix estimation with l1
          penalization using the GraphLasso. The connection of two nodes is
          estimated by conditioning on all other nodes [-1,1].
        * ``ledoit_wolf``: Sparse inverse covariance matrix estimation with l2
          shrinkage using Ledoit-Wolf [-1,1].
        * ``dcorr``: Distance correlation [0,1]. This metric can capture
          non-linear relationships.
        * ``dcov``: Distance covariance.
        * ``eu``: Euclidean distance.

    pval : boolean, optional
        return p-values, only available for ``corr``, ``wtc``, ``ktau``,``rho``
        and ``mic`` so far (default=False).
    TR : float
        Time to repeat: the sampling interval (only for ``coh``, ``pcoh``,
        ``delay`` and ``granger``).
    fq_l : float
        lower frequency bound (only for ``coh``, ``pcoh``, ``delay`` and
        ``granger``).
    fq_u : float
        upper frequency bound (only for ``coh``, ``pcoh``, ``delay`` and
        ``granger``).
    order : integer
        time-lag (only for ``measure='granger'``)
    scale : integer [1,2]
        Wavelet scale (only for ``measure='wtc'``)
    w : pywt.wavelet object
        default is pywt.Wavelet('db4')
    idp : integer
        Index of the timeseries to condition the semi-partial coherence on
        (only if ``measure='spcoh'``)
    excl_zero_cov : boolean (default: False)
        Automatically exclude node timeseries with zero covariance. Values in
        the adjacency matrix are set to zero.

    Returns
    -------
    A : ndarray, shape(n_rois, n_rois)
        Adjacency matrix of the graph.

    P : ndarray, shape(n_rois, n_rois)
        Statistical p-values (2-tailed) for the similarity measure. Only if
        ``pval=True``

    Notes
    -----
    The calculation runs faster if ``pval=False`` (default). The diagonal is
    always zero.

    See Also
    --------
    adj_dynamic: for a dynamic graph representation/adjacency matrix
    nt.timeseries.utils.cross_correlation_matrix: cross-correlation matrix

    References
    ----------
    .. [1] Bassett, D. S., Wymbs, N. F., Porter, M. A., Mucha, P. J., Carlson,
           J. M., & Grafton, S. T. (2011). Dynamic reconfiguration of human
           brain networks during learning. Proceedings of the National Academy
           of Sciences, 108(18), 7641–7646. doi:10.1073/pnas.1018985108
    .. [2] Salvador, R., Suckling, J., Schwarzbauer, C., & Bullmore, E. (2005).
           Undirected graphs of frequency-dependent functional connectivity in
           whole brain networks. Philosophical transactions of the Royal
           Society of London Series B, Biological sciences, 360(1457), 937–946.
           doi:10.1098/rstb.2005.1645
    .. [3] Kayser, A. S., Sun, F. T., & D'Esposito, M. (2009). A comparison of
           Granger causality and coherency in fMRI-based analysis of the motor
           system. Human Brain Mapping,30(11), 3475–3494. doi:10.1002/hbm.20771
    .. [4] Roebroeck, A., Formisano, E., & Goebel, R. (2005). Mapping directed
           influence over the brain using Granger causality and fMRI.
           NeuroImage, 25(1), 230–242. doi:10.1016/j.neuroimage.2004.11.017
    .. [5] Zamora-López, G., Zhou, C., & Kurths, J. (2010). Cortical hubs form
           a module for multisensory integration on top of the hierarchy of
           cortical networks. Frontiers in neuroinformatics, 4.
    .. [6] Salvador, R., Anguera, M., Gomar, J. J., Bullmore, E. T., &
           Pomarol-Clotet, E. (2010). Conditional Mutual Information Maps as
           Descriptors of Net Connectivity Levels in the Brain. Frontiers in
           neuroinformatics, 4. doi:10.3389/fninf.2010.00115
    .. [7] Sun, F. T., Miller, L. M., & D'Esposito, M. (2004). Measuring
           interregional functional connectivity using coherence and partial
           coherence analyses of fMRI data. NeuroImage, 21(2), 647–658.
           doi:10.1016/j.neuroimage.2003.09.056

    Examples
    --------
    >>> data = get_fmri_data()
    >>> d = percent_signal_change(data) # normalize data
    >>> print data.shape
    (31, 250) # 31 nodes and 250 time-points
    >>> # adjacency matrix based on correlation metric
    >>> A = adj_static(data, measure='corr')
    >>> print A.shape
    (31, 31) # node-wise connectivity matrix

    >>> # get adjacency matrix and p-values
    >>> A, P = adj_static(data, measure='corr', pval=True)
    >>> print P.shape
    (31, 31) # p-value for every edge in the adjacency matrix
    """

    data = deepcopy(ts)
    n_channel = data.shape[0]
    # n_tps = data.shape[1]

    # TODO think about option to automatically exclude zero covaraince nodes
    # especially important for granger
    n_nodes = data.shape[0]
    if excl_zero_cov:
        # test for zero covariance to exclude
        std = np.std(data, axis=1)
        idx = np.where(std != 0.0)[0]
        data = data[idx, :]

    # this performs just the wavelet transformation, the correlation part
    # is identical to measure='corr'
    # if measure == 'wtc':
    #     data = wavelet_transform(data, w=w, scale=scale)
    #     measure = 'corr' # perform correlation of wavelet transformed ts

    if measure == 'corr':

        # correlation = np.dot(x,y)/(np.dot(x,x) * np.dot(y,y))
        ADJ = np.corrcoef(data)

        if pval:
            # ADJ = np.zeros((data.shape[0], data.shape[0]))
            # P = np.zeros((data.shape[0], data.shape[0]))
            #
            # idx = tril_indices(data.shape[0], -1)
            # ADJ[idx] = -99 # save some running time by calculating only
            # for i in range(data.shape[0]): # the lower traingle
            #     for j in range(data.shape[0]):
            #         if ADJ[i,j] == -99:
            #             ADJ[i,j], P[i,j] = pearsonr(data[i,:], data[j,:])
            #
            # ADJ = ADJ + ADJ.T
            # P = P + P.T
            # P = pearsonr_2pval(ADJ, n=n_tps)
            # fill_diagonal(P,1)
            P = np.ones((n_channel, n_channel))

    elif measure == 'cov':
        # d = data.copy()
        # mean = d.mean(axis=1)
        # std = d.std(axis=1)
        # d -= mean.reshape(mean.shape[0], 1)
        # d /= std.reshape(mean.shape[0], 1)
        ADJ = np.cov(data)

    elif measure == 'pcorr':
        # data needs to be normalized?!
        # inv vs. pinv vs pinv2:
        # http://blog.vene.ro/2012/08/18/inverses-pseudoinverses-numerical
        # issues-speed-symmetry/
        ADJ = np.linalg.inv(np.cov(data))  # or pinv?
        d = 1 / np.sqrt(np.diag(ADJ))
        ADJ *= d
        ADJ *= d[:, np.newaxis]

        # TODO: this might be much faster
        # from scipy.linalg.lapack import get_lapack_funcs
        # getri, getrf = get_lapack_funcs(('getri', 'getrf'),
        #                                 (np.empty((), dtype=np.float64),
        #                                  np.empty((), dtype=np.float64)))
        #
        # covariance = np.cov(data)
        # lu, piv, _ = getrf(np.dot(covariance.T, covariance), True)
        # precision, _ = getri(lu, piv, overwrite_lu=True)
        # precision = np.dot(covariance, precision)

    # elif measure == 'ktau':
    #
    #     ADJ = np.zeros((data.shape[0], data.shape[0]))
    #     P = np.zeros((data.shape[0], data.shape[0]))
    #
    #     idx = tril_indices(data.shape[0], -1)
    #     ADJ[idx] = -99
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[0]):
    #             if ADJ[i, j] == -99:
    #                 ADJ[i, j], P[i, j] = kendalltau(data[i,:], data[j,:])

        ADJ = ADJ + ADJ.T
        # P = P + P.T
        # fill_diagonal(P, 1)

    elif measure == 'rho':

        ADJ = np.zeros((data.shape[0], data.shape[0]))
        P = np.zeros((data.shape[0], data.shape[0]))

        idx = tril_indices(data.shape[0], -1)
        # save some running time by calculating only the lower traingle
        ADJ[idx] = -99
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                if ADJ[i, j] == -99:
                    ADJ[i, j], P[i, j] = spearmanr(data[i, :], data[j, :])

        ADJ = ADJ + ADJ.T
        P = P + P.T
        fill_diagonal(P, 1)

    elif measure == 'coh':
        from nitime import TimeSeries
        from nitime.analysis.coherence import CoherenceAnalyzer
        T = TimeSeries(data, sampling_interval=TR)
        # Initialize the coherence analyzer
        C = CoherenceAnalyzer(T)
        COH = C.coherence
        # remove Nan's
        COH[np.isnan(COH)] = 0.

        freq_idx = np.where((C.frequencies > fq_l) * (C.frequencies < fq_u))[0]
        # averaging over the last dimension (=frequency)
        ADJ = np.mean(COH[:, :, freq_idx], -1)

    if excl_zero_cov:
        ADJ = np.zeros((n_nodes, n_nodes))
        ADJ[idx] = ADJ

    fill_diagonal(ADJ, 0)
    ADJ[np.isnan(ADJ)] = 0.  # might occur if zero cov

    if pval:
        return ADJ, P
    else:
        return ADJ


def get_random_graph(n=30, weighted=False, directed=False, fmt='np'):
    """returns a random graph for testing purpose

    Parameters
    ----------
    n : integer
        Number of nodes.
    weighted : boolean
        The adjacency matrix is weighted.
    directed : boolean
        The adjacency matrix is directed.
    fmt : string ['nx'|'ig'|'gt'|'bct'|'mat'|'snp']
        Output format

        * ``nx``: networkx.Graph
        * ``ig``: igraph.Graph
        * ``gt``: graph_tool.Graph
        * ``bct``: Brain connectivity toolbox (BCT) graph (C++ version)
        * ``mat``: matlab file
        * ``snp``: sparse numpy graph (scipy lil_matrix)

    Returns
    -------
    graphs : graph object

    Examples
    --------
    >>> A = get_random_graph(30, directed=True)
    >>> is_directed(A)
    True
    """

    G = nx.random_graphs.watts_strogatz_graph(n, np.int(n * 3 / float(n)), 0.2)
    ADJ = np.asarray(nx.to_numpy_matrix(G))

    if directed:
        idx = tril_indices(n, -1)
        edge_info_permutation = np.random.permutation(ADJ[idx])
        ADJ[idx] = edge_info_permutation

    if weighted:
        # idx = tril_indices(n, -1)
        idy = triu_indices(n, 0)
        id = np.where(ADJ == 0)
        weights = np.random.rand(n, n)
        weights[idy] = 0.  # zero upper triangle
        weights[id] = 0.
        ADJ = weights + weights.T

    if fmt is not 'np':
        ADJ = convert_to_graph(ADJ, weighted=weighted, directed=directed,
                               fmt=fmt)
    return ADJ


def karate_club(weighted=False):
    """Zachary's Karate club graph (n_nodes: 34).

    The graph is undirected.

    Returns
    -------
    A : ndarray, shape(n, n)
        Adjacency matrix of the graph.
    weighted : boolean
        The adjacency matrix is weighted.

    References
    ----------
    .. [1] Zachary W.
       An information flow model for conflict and fission in small groups.
       Journal of Anthropological Research, 33, 452-473, (1977).
    .. [2] Data file from:
       http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm

    Examples
    --------
    >>> A = karate_club()
    >>> is_weighted(A)
    False

    >>> A = karate_club(weighted=True)
    >>> is_weighted(A)
    True
    """

    G = nx.Graph()
    G.add_nodes_from(range(34))

    zacharydat_w = """\
0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0
1 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0
1 1 0 1 0 0 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0
1 1 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1
0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 1
0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 1 1 0 0 0 0 0 1 1 1 0 1
0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 0
0 4 5 3 3 3 3 2 2 0 2 3 1 3 0 0 0 2 0 2 0 2 0 0 0 0 0 0 0 0 0 2 0 0
4 0 6 3 0 0 0 4 0 0 0 0 0 5 0 0 0 1 0 2 0 2 0 0 0 0 0 0 0 0 2 0 0 0
5 6 0 3 0 0 0 4 5 1 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 2 0
3 3 3 0 0 0 0 3 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 2 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 5 0 0 0 3 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 2 5 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 4 4 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 3 4
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
2 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
3 5 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 4
0 0 0 0 0 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2
2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 1
2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0 4 0 3 0 0 5 4
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 3 0 0 0 2 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 2 0 0 0 0 0 0 7 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 2
0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 3 0 0 0 0 0 0 0 0 4
0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 4 0 0 0 0 0 4 2
0 2 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 3 3
2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 7 0 0 2 0 0 0 4 4
0 0 2 0 0 0 0 0 3 0 0 0 0 0 3 3 0 0 1 0 3 0 2 5 0 0 0 0 0 4 3 4 0 5
0 0 0 0 0 0 0 0 4 2 0 0 0 3 2 4 0 0 2 1 1 0 3 4 0 0 2 4 2 2 3 4 5 0"""

    zacharydat = """\
0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0
1 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0
1 1 0 1 0 0 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0
1 1 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1
0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 1
0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 1 1 0 0 0 0 0 1 1 1 0 1
0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 0"""
    row = 0

    if weighted:
        for line in zacharydat_w.split('\n'):
            thisrow = list(map(int, line.split(' ')))
            for col in range(0, len(thisrow)):
                if thisrow[col] > 0:
                    G.add_edge(row, col, weight=thisrow[col])
            row += 1
    else:
        for line in zacharydat.split('\n'):
            thisrow = list(map(int, line.split(' ')))
            for col in range(0, len(thisrow)):
                if thisrow[col] == 1:
                    G.add_edge(row, col)
            row += 1
    return np.asarray(nx.to_numpy_matrix(G))
