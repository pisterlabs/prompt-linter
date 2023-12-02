# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This tools auto eliminate aliens from CDI experiment data. It is configuration driven.

"""

import numpy as np
import os
import tifffile as tif
import cohere_core.utilities.utils as ut

__author__ = "Kenly Pelzer, Ross Harder"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_asymmetry',
           'analyze_clusters',
           'crop_center',
           'save_arr',
           'save_arrays',
           'auto_alien1',
           'filter_aliens',
           'remove_blocks',
           'remove_aliens']


def get_asymmetry(arr):
    """
    Returns asymmetry of an array.

    Parameters
    ----------
    arr : ndarray
        an array to find asymmetry

    Returns
    -------
    ndarray
        an array capturing asymmetry of original array
    """

    arr_rev = arr[::-1, ::-1, ::-1]
    denom = (arr + arr_rev) / 2.0
    denom_nz = np.where(denom == 0, 1.0, denom)
    asym = np.where(denom > 0.0, abs(arr - arr_rev) / denom_nz, 0.0)
    # asym only assigned to non-zero intensity points in the passed array
    return np.where(arr > 0, asym, 0)


# add output of absolute cluster size.
def analyze_clusters(arr, labels, nz):
    """
    Analyzes clusters and returns characteristics in arrays.

    Parameters
    ----------
    arr : ndarray
        the analyzed array
    labels: arr
        cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
    nz : tuple
        tuple of arrays, each array containing indices of elements in arr that are non-zero along one axis.

    Returns
    -------
    tuple
        tuple containing the following arrays:
        nlabels              # number of labels, i.e clusters
        labels_arr           # array with label for each non zero point
        rel_cluster_size     # array with cluster size divided by max cluster size for each
                             # non zero point
        cluster_avg          # array with cluster average for each non zero point
        noise_arr            # array with points that are non zero but not in cluster
        no_noise             # array with noise poits set to 0
        label_counts         # tuple of two arrays: First is label number, second is number of
                             # occurances of that label (size of cluster)
        cluster_avg_asym     # array with average asymmetry of a points in cluster
        asymmetry            # array of asymmetry with regard to entire array
        cluster_size         # array with cluster size for each non zero point
    """

    labels_arr = np.zeros_like(arr)
    noise_arr = np.zeros_like(arr)
    cluster_size = np.zeros_like(arr)
    cluster_avg = np.zeros_like(arr).astype(np.float32)
    cluster_avg_asym = np.zeros_like(arr).astype(np.float32)
    asymmetry = get_asymmetry(arr)

    # label_counts is tuple of two arrays.  First is label number, second is number of occurances of that label (size of cluster).
    label_counts = np.unique(labels, return_counts=True)

    # nz and labels are the same length. so the indicies given by nz will be set
    # to their corresponding cluster number (includes noise pts).
    labels_arr[nz] = labels

    # this selects the nz indicies where labels=-1 (noise)
    noise_pts = tuple([nz[n][labels == -1] for n in range(3)])
    no_noise = arr

    # move the points labeled noise into their own array
    # remove the noise out of arr (no_noise is copy of arr)
    noise_arr[noise_pts] = arr[noise_pts]
    no_noise[noise_pts] = 0
    nlabels = len(label_counts[0])
    # print("processing labels")
    # loop over the labels (clusters).  label_counts[0] is the unique labels
    for n in range(1, nlabels):
        #    print("   %i %i      "%(label_counts[0][n],label_counts[1][n]), end='\r')
        # the nth label from the first array of the label_counts tuple
        n_lab = label_counts[0][n]
        # the indicies of the points belonging to label n
        cluspts = tuple([nz[d][labels == n_lab] for d in range(3)])
        # the second array of the label_counts tuple is the number of points
        # with that label.  So put those into an array.
        cluster_size[cluspts] = label_counts[1][n]
        # compute the average intensity of each cluster and write into an array.
        cluster_avg[cluspts] = np.sum(arr[cluspts]) / cluspts[0].size
        # compute average asym of each cluster and store in array.
        cluster_avg_asym[cluspts] = np.sum(asymmetry[cluspts]) / cluspts[0].size
        # print("   %i %i %f %f     "%(label_counts[0][n],label_counts[1][n],np.sum(asymmetry[cluspts]),cluspts[0].size), end='\n')
    # print("largest clus size", cluster_size.max())
    # compute relative cluster sizes to largest (main) cluster.
    rel_cluster_size = cluster_size / cluster_size.max()

    # return all of these arrays
    return (
        nlabels, labels_arr, rel_cluster_size, cluster_avg, noise_arr, no_noise, label_counts, cluster_avg_asym,
        asymmetry,
        cluster_size)


def crop_center(arr):
    """
    Finds max element in array and crops the array to be symetrical with regard to this point in each direction.

    Parameters
    ----------
    arr : ndarray
        an array

    Returns
    -------
    centered : ndarray
        an array symetrical in all dimensions around the max element of input array
    """

    shape = arr.shape
    # This tells us the point of highest intensity, which we will use as the center for inversion operations
    center = np.unravel_index(np.argmax(arr, axis=None), shape)

    # clip the largest possible cuboid putting the point of highest intensity at the center
    principium = []
    finis = []
    for i in range(len(shape)):
        half_shape = min(center[i], shape[i] - center[i] - 1)
        principium.append(center[i] - half_shape)
        finis.append(center[i] + half_shape + 1)
    centered = arr[principium[0]:finis[0], principium[1]:finis[1], principium[2]:finis[2]]

    return centered


def save_arr(arr, dir, fname):
    """
    Saves an array in 'tif' format file.

    Parameters
    ----------
    arr : ndarray
        an array to save
    dir : str
        directory to save the file to
    fname : str
        file name

    Returns
    -------
    nothing
    """

    if dir is not None:
        full_name = dir + '/' + fname
    else:
        full_name = fname  # save in the current dir
    tif.imsave(full_name, arr.transpose().astype(np.float32))


def save_arrays(arrs, iter, thresh, eps, dir):
    """
    Saves multiple arrays in 'tif' format files. Determines file name from given parameters: iteration, threshold, and eps.

    Parameters
    ----------
    arr : tuple
        a tuple of arrays to save
    iter, thresh, eps : str, str, str
        parameters: iteration, threshold, and eps, to deliver file name from
    dir : str
        directory to save the file to

    Returns
    -------
    nothing
    """

    save_arr(arrs[1], dir, "db%d_%3.2f_labels_arr%3.2f.tif" % (iter, thresh, eps))
    save_arr(arrs[2], dir, "db%d_%3.2f_rel_clustersizes%3.2f.tif" % (iter, thresh, eps))
    save_arr(arrs[3], dir, "db%d_%3.2f_clusteravg%3.2f.tif" % (iter, thresh, eps))
    save_arr(arrs[4], dir, "db%d_%3.2f_noise%3.2f.tif" % (iter, thresh, eps))
    save_arr(arrs[5], dir, "db%d_%3.2f_no_noise%3.2f.tif" % (iter, thresh, eps))
    save_arr(arrs[7], dir, "db%d_%3.2f_clusteravgasym%3.2f.tif" % (iter, thresh, eps))
    save_arr(arrs[8], dir, "db%d_%3.2f_asym%3.2f.tif" % (iter, thresh, eps))
    save_arr(arrs[9], dir, "db%d_%3.2f_abs_clustersizes%3.2f.tif" % (iter, thresh, eps))


def auto_alien1(data, config, data_dir=None):
    from sklearn.cluster import DBSCAN
    """
    Removes aliens from experimental CDI data using iterative algorithm and returns the result.

    The algorithm follows the steps:
    1. Initialization:
    - initialize variables with the configuration parameters
    - crop the data array around maximum element to it's biggest size
    - sets points below threshold value to 0
    - finds non-zero elements of the data array and keeps them as tuples of indices
    2. Iteration loop, runs until number of clasters remains unchanged
    - runs DBSCAN algorithm on the non-zero and returns clasters labels
    - analyzes the results to find relative clusters sizes, and clusters average asymmetry, and other characteristics
    - removes alien clusters, i.e. the ones with relative cluster size below configured size threshold and with average asymmetry over configured asymmetry threshold
    - go back to the loop using the non-zero elements of alien removed array to the DBSCAN
    3. If configured, add final step to apply gaussian convolusion to the result and use it as a filter with configured sigma as threshold

    Parameters
    ----------
    data : ndarray
        an array with experiment data
    config : Object
        configuration object providing access to configuration parameters
    data_dir : str
        a directory where 'alien_analysis' subdirectory will be created to save results of analysis if configured

    Returns
    -------
    cuboid : ndarray
        data array with removed aliens
    """
    data_dir = data_dir.replace(os.sep, '/')
    if 'AA1_size_threshold' in config:
        size_threshold = config['AA1_size_threshold']
    else:
        size_threshold = 0.01
    if 'AA1_asym_threshold' in config:
        asym_threshold = config['AA1_asym_threshold']
    else:
        asym_threshold = 1.75
    if 'AA1_min_pts' in config:
        min_pts = config['AA1_min_pts']
    else:
        min_pts = 5
    if 'AA1_eps' in config:
        eps = config['AA1_eps']
    else:
        eps = 1.1
    if 'AA1_amp_threshold' in config:
        threshold = config['AA1_amp_threshold']
    else:
        threshold = 6
    if 'AA1_save_arrs' in config:
        save_arrs = config['AA1_save_arrs']
        if save_arrs:
            save_dir = data_dir + '/alien_analysis'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    else:
        save_arrs = False

    if 'AA1_expandcleanedsigma' in config:
        expandcleanedsig = config['AA1_expandcleanedsigma']
    else:
        expandcleanedsig = 0.0

    cuboid = crop_center(data)
    cuboid = np.where(cuboid >= threshold, cuboid, 0)
    if (save_arrs):
        save_arr(cuboid, save_dir, "db%3.2f_cuboid%3.2f.tif" % (threshold, eps))
        save_arr(cuboid[::-1, ::-1, ::-1], save_dir, "db%3.2f_cuboidrev%3.2f.tif" % (threshold, eps))

    # the non_zero is a tuple of arrays, each array containing indices of elements that are non-zero along one axis.
    non_zero = cuboid.nonzero()

    # https://scikit-learn.org/stable/modules/clustering.html#dbscan
    # labels is same size as input arr with a cluster label per point
    iter = 0
    nclusters = 0
    finished = False
    while (not finished):
        non_zero = cuboid.nonzero()
        # print("running db", iter)
        labels = DBSCAN(eps=eps, metric='euclidean', min_samples=min_pts, n_jobs=-1).fit_predict(
            np.array(non_zero).transpose().astype(np.float32))
        # print("running analyze_clusters", iter)
        arrs = analyze_clusters(cuboid, labels, non_zero)
        if (save_arrs):
            save_arrays(arrs, iter, threshold, eps, save_dir)
        if nclusters == arrs[0]:
            finished = True
        nclusters = arrs[0]
        if iter == 0:  # keep values for all iterations
            rel_cluster_size = arrs[2]
            cluster_avg_asym = arrs[7]
        # print("cleaning cuboid", iter)
        cuboid = np.where(np.logical_and(rel_cluster_size < size_threshold, cluster_avg_asym > asym_threshold), 0.0,
                          cuboid)
        # print("iter", iter, nclusters)
        iter += 1

    if (expandcleanedsig > 0):
        cuboid = np.where(cuboid > 0, 1.0, 0.0)
        sig = [expandcleanedsig, expandcleanedsig, 1.0]
        cuboid = ut.gauss_conv_fft(cuboid, sig)
        no_thresh_cuboid = crop_center(data)
        cuboid = np.where(cuboid > 0.1, no_thresh_cuboid, 0.0)
    return cuboid


def remove_blocks(data, config_map):
    """
    Sets to zero given alien blocks in the data array.

    Parameters
    ----------
    data : ndarray
        an array with experiment data
    config : Object
        configuration object providing access to configuration parameters

    Returns
    -------
    data : ndarray
        data array with zeroed out aliens
    """
    import ast

    if 'aliens' in config_map:
        aliens = ast.literal_eval(config_map['aliens'])
        for alien in aliens:
            # The ImageJ swaps the x and y axis, so the aliens coordinates needs to be swapped, since ImageJ is used
            # to find aliens
            data[alien[0]:alien[3], alien[1]:alien[4], alien[2]:alien[5]] = 0
    return data


def filter_aliens(data, config_map):
    """
    Sets to zero points in the data array defined by a file.

    Parameters
    ----------
    data : ndarray
        an array with experiment data
    config : Object
        configuration object providing access to configuration parameters

    Returns
    -------
    data : ndarray
        data array with zeroed out aliens
    """
    if 'alien_file' in config_map:
        alien_file = config_map['alien_file']
        if os.path.isfile(alien_file):
            mask = np.load(alien_file)
            for i in range(len(mask.shape)):
                if mask.shape[i] != data.shape[i]:
                    print('exiting, mask must be of the same shape as data:', data.shape)
                    return
            data = np.where((mask == 1), data, 0.0)
        else:
            print('alien file does not exist ', alien_file)
    else:
        print('alien_file parameter not configured')
    return data


def remove_aliens(data, config_map, data_dir=None):
    """
    Finds which algorithm is cofigured to remove the aliens and applies it to clean the data.

    Parameters
    ----------
    data : ndarray
        an array with experiment data
    config : Object
        configuration object providing access to configuration parameters
    data_dir : str
        a directory where 'alien_analysis' subdirectory will be created to save results of analysis if configured
    Returns
    -------
    data : ndarray
        data array without aliens
    """

    if 'alien_alg' in config_map:
        algorithm = config_map['alien_alg']
        if algorithm == 'block_aliens':
            data = remove_blocks(data, config_map)
        elif algorithm == 'alien_file':
            data = filter_aliens(data, config_map)
        elif algorithm == 'AutoAlien1':
            data = auto_alien1(data, config_map, data_dir)
        elif algorithm != 'none':
            print('not supported alien removal algorithm', algorithm)
    else:
        print('alien_alg not configured')

    return data

## https://stackoverflow.com/questions/51503672/decorator-for-timeit-timeit-method/51503837#51503837
# from functools import wraps
# from time import time
#
# def measure(func):
#    @wraps(func)
#    def _time_it(*args, **kwargs):
#        start = int(round(time() * 1000))
#        try:
#            return func(*args, **kwargs)
#        finally:
#            end_ = int(round(time() * 1000)) - start
#            print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
#

