#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:47:10 2021
@author: Ajit Johnson Nirma
Latent Dirichlet Allocation (LDA) for identifying spatial motifs
EAch spatial motif is a mixture of over a set of cell-type proximity organisation probabilities.
"""


from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import re

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Function
def spatial_lda (adata, x_coordinate='X_centroid',y_coordinate='Y_centroid',
                 phenotype='phenotype', method='radius', radius=30, knn=10,
                 imageid='imageid',num_motifs=10, random_state=0, subset=None,
                 label='spatial_lda',**kwargs):
    """
    

    Parameters
    ----------
    adata : AnnData object

    x_coordinate : float, required
        Column name containing the x-coordinates values. The default is 'X_centroid'.
    y_coordinate : float, required
        Column name containing the y-coordinates values. The default is 'Y_centroid'.
    phenotype : string, required
        Column name of the column containing the phenotype information. 
        It could also be any categorical assignment given to single cells. The default is 'phenotype'.
    method : string, optional
        Two options are available: a) 'radius', b) 'knn'.
        a) radius - Identifies the neighbours within a given radius for every cell.
        b) knn - Identifies the K nearest neigbours for every cell.
        The default is 'radius'.
    radius : int, optional
        The radius used to define a local neighbhourhood. The default is 30.
    knn : int, optional
        Number of cells considered for defining the local neighbhourhood. The default is 10.
    imageid : string, optional
        Column name of the column containing the image id. The default is 'imageid'.
    subset : string, optional
        imageid of a single image to be subsetted for analyis. The default is None.
    num_motifs : int, optional
        The number of requested latent motifs to be extracted from the training corpus. The default is 10.
    random_state : int, optional
        Either a randomState object or a seed to generate one. Useful for reproducibility. The default is 0.
    label : string, optional
        Key for the returned data, stored in `adata.obs`. The default is 'spatial_lda'.

    Returns
    -------
    adata : AnnData object
        Updated AnnData object with the results stored in `adata.obs['spatial_lda']`.
    
    Example
    -------
    # Running the radius method
    adata = spatial_lda (adata, num_motifs=10, radius=100)

    """

    
    def spatial_lda_internal (adata_subset, x_coordinate,y_coordinate,phenotype, 
                              method, radius, knn, imageid):
        
        # Print which image is being processed
        print('Processing: ' + str(np.unique(adata_subset.obs[imageid])))
        
        # Create a DataFrame with the necessary inforamtion
        data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'phenotype': adata_subset.obs[phenotype]})
        
        # Identify neighbourhoods based on the method used
        # a) KNN method
        if method == 'knn':
            print("Identifying the " + str(knn) + " nearest neighbours for every cell")
            tree = BallTree(data[['x','y']], leaf_size= 2)
            ind = tree.query(data[['x','y']], k=knn, return_distance= False)
        
        # b) Local radius method
        if method == 'radius':
            print("Identifying neighbours within " + str(radius) + " pixels of every cell")
            kdt = BallTree(data[['x','y']], leaf_size= 2) 
            ind = kdt.query_radius(data[['x','y']], r=radius, return_distance=False)
            
        # Map phenotype
        phenomap = dict(zip(list(range(len(ind))), data['phenotype'])) # Used for mapping
        for i in range(len(ind)):
            ind[i] = [phenomap[letter] for letter in ind[i]]
            
        # return
        return ind
    
    # Subset a particular image if needed
    if subset is not None:
        adata_list = [adata[adata.obs[imageid] == subset]]
    else:
        adata_list = [adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()]
        
    # Apply function to all images
    # Create lamda function 
    r_spatial_lda_internal = lambda x: spatial_lda_internal(adata_subset=x,
                                                            x_coordinate=x_coordinate,
                                                            y_coordinate=y_coordinate,
                                                            phenotype=phenotype, 
                                                            method=method, 
                                                            radius=radius, 
                                                            knn=knn, 
                                                            imageid=imageid) 
    all_data = list(map(r_spatial_lda_internal, adata_list)) # Apply function 
    
    # combine all the data into one
    texts = np.concatenate( all_data, axis=0 ).tolist()
    
    # LDA pre-processing
    print ('Pre-Processing Spatial LDA')
    # Create Dictionary
    id2word = corpora.Dictionary(texts)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Build LDA model
    print ('Training Spatial LDA')
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=num_motifs, 
                                               random_state=random_state,**kwargs)
    
    # Compute Coherence Score
    print ('Calculating the Coherence Score')
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # isolate the latent features
    print ('Gathering the latent weights')
    topic_weights = []
    for row_list in lda_model[corpus]:
        tmp = np.zeros(num_motifs)
        for i, w in row_list:
            tmp[i] = w
        topic_weights.append(tmp)
    # conver to dataframe
    arr = pd.DataFrame(topic_weights, index=adata.obs.index).fillna(0)
    arr = arr.add_prefix('Motif_')
    
    # isolate the weights of phenotypes
    pattern = "(\d\.\d+).\"(.*?)\""
    cell_weight = pd.DataFrame(index=np.unique(adata.obs[phenotype]))
    for i in range(0, len(lda_model.print_topics())):
        level1 = lda_model.print_topics()[i][1]
        tmp = pd.DataFrame(re.findall(pattern, level1))
        tmp.index = tmp[1]
        tmp = tmp.drop(columns=1)
        tmp.columns = ['Motif_'+ str(i)]
        cell_weight = cell_weight.merge(tmp, how='outer', left_index=True, right_index=True)
    # fill zeros
    cell_weight = cell_weight.fillna(0).astype(float)
    
    # save the results in anndata object
    adata.uns[label] = arr # save the weight for each cell
    adata.uns[str(label)+'_probability'] = cell_weight # weights of each cell type
    adata.uns[str(label)+'_model'] = lda_model
    
    # return
    return adata
