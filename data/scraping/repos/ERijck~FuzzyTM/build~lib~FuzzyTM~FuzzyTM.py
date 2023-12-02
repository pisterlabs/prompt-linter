# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 13:49:08 2021

@author: Emil Rijcken
"""
import numpy as np
import pandas as pd
import math
from collections import Counter
from sparsesvd import sparsesvd
from scipy.sparse import dok_matrix
from pyfume import Clustering
import warnings
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

class FuzzyTM():
    def __init__(self, input_file, algorithm = 'flsa-w', num_topics = 15, num_words = 20, word_weighting = 'normal', cluster_method = 'fcm', svd_factors = 2):
        self.input_file = input_file #List in which each element is a list of tokens
        self._check_data_format_input_data(self.input_file)
        self.algorithm = algorithm
        self.num_topics = num_topics
        self.num_words = num_words
        self.word_weighting = word_weighting
        self.cluster_method = cluster_method
        self.svd_factors = svd_factors        
        
        if not isinstance(self.num_topics, int) or self.num_topics < 1:
            raise ValueError("Please use a positive int for num_topics")
        if not isinstance(self.num_words, int) or self.num_words <1 :
            raise ValueError("Please use a positive int for num_words")
        if self.algorithm in ["flsa", "flsa-w"] and self.word_weighting not in ["entropy","idf","normal","probidf"]:
            raise ValueError("Invalid word weighting method. Please choose between: 'entropy','idf','normal' and'probidf'")
        if self.cluster_method not in ["fcm", "fst-pso","gk"]:
            raise ValueError("Invalid 'cluster_method. Please choose between 'fcm', 'fst-pso' or 'gk'")
        if not isinstance(self.svd_factors, int) and self.svd_factors >0:
            raise ValueError("Please use a positive int for svd_factors")
             
        self.vocabulary, self.vocabulary_size = self.create_vocabulary(self.input_file) #List with all unique words and the length of the list
        self.word_to_index, self.index_to_word = self.get_index_dicts(self.vocabulary) #dictionaries that map words to an index and vice versa
        self.sum_words = self.get_sum_words(self.input_file) #dictionary with keys: words, values: counts
    
    
    def _check_data_format_input_data(self, input_data):
        '''
        Check whether the input data has the right format. 
        Correct format: list of list of str (tokens)
        
        The function  raises errors if the format is incorrect. However, it does not return anything.
        '''
        if not isinstance(input_data, list):
            raise TypeError("Input file is not a list")        
        for i, doc in enumerate(input_data):
            if not isinstance(doc,list):
                raise TypeError("The element of input_file at index " + str(i) + " is not a list")
            if not len(doc) > 0:
                raise ValueError("The input_file has an empty list at index " + str(i) + " and should contain at least one str value")
            for j, word in enumerate(doc):
                if not isinstance(word, str):
                    raise TypeError(f"Word {j} of document {i} is not a str")
        return
    
    def get_vocabulary(self):
        '''
        Returns a set of all the words in the corpus 
        
        Example:
     
        After initializing an instance of the FuzzyTM models as 'model'
         
        corpus= [['this','is','the','first','file'],
             ['and','this','is','second','file']]
    
        model.get_vocabulary()
    
        >>> {'this','is','the','first','file','and','second'}    
        '''
        return self.vocabulary
    
    
    def get_vocabulary_size(self):
        '''
        Returns the number of words in the vocabulary
        
        Example:
            
        After initializing an instance of the FuzzyTM models as 'model'
         
        corpus= [['this','is','the','first','file'],
             ['and','this','is','second','file']]
    
        model.get_vocabulary_size()
    
        >>> 7
        '''
        return self.vocabulary_size
    
    def get_word_to_index(self):
        '''
        Returns a dictionary with:
            keys: vocabulary words (str)
            values: unique number associated to a word (int)
        '''
        return self.word_to_index
   
    def get_index_to_word(self):
        '''
        Returns a dictionary with:
            keys: unique number associated to a word (int)
            values: vocabulary words (str) 
        ''' 
        return self.index_to_word
    
    def get_input_file(self):
        '''
        returns the input file 'input_file'
        '''
        return self.input_file
    
    def create_vocabulary(self, input_file):
        '''
        Create the vocabulary from input data
        Parameter:
             - input-file: list of lists of strings (tokens)
        
        Returns:
            vocabulary: set with all the vocabulary words   
        '''
        vocabulary = set([el for lis in input_file for el in lis])
        return vocabulary, len(vocabulary)
        
    def get_index_dicts(self, vocabulary):
        '''
        Create the dictionaries with mappings between words and indices
        
        Parameter:
            vocabulary: all the words in the corpus (set)
            
        Returns:
            word_to_index: dictionary
                keys: vocabulary words (str)
                values: unique number associated to a word (int)
            
            index_to_word: dictionary
                keys: unique number associated to a word (int)
                values: vocabulary words (str) 
        '''
        if not isinstance(vocabulary, set):
            raise ValueError("Please use a 'set' type for 'vocabulary'.")
        word_to_index = dict()
        index_to_word = dict()
        for i, word in enumerate(vocabulary):
            word_to_index[word] = i
            index_to_word[i] = word
        return word_to_index, index_to_word
            
    def get_sum_words (self, input_file):
        '''
        Creates a Counter object that provides the count of each word in the corpus (input_file).
        
        Parameter:
            input-file: list of lists of strings (tokens)
            
        Returns:
            Counter object:
                keys: vocabulary words (str)
                values: the count of word in the corpus (int)
        '''
        sum_words = Counter()
        for document_index, document in enumerate(input_file):
            sum_words.update(Counter(document))
        return sum_words
    
    def get_sparse_local_term_weights(self, input_file, vocabulary_size, word_to_index):
        '''
        Creates a sparse matrix showing the frequency of each words in documents. 
        (See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html)
    
        Axes:
            rows: documents (size: number of documents in corpus)
            columns: words (size: vocabulary length)
            
        Parameters:
             - input-file: list of lists of strings (tokens)
             - vocabulary_size: int (positive)
             - word_to_index: dictionary
                keys: vocabulary words (str)
                values: unique number associated to a word (int)
        '''
        sparse_local_term_weights = dok_matrix((len(input_file), vocabulary_size), dtype=np.float32) 
        for document_index, document in enumerate(input_file):         
            document_counter = Counter(document)
            for word in document_counter.keys():
                sparse_local_term_weights[document_index, word_to_index[word]] = document_counter[word]
        return sparse_local_term_weights   
    
    def get_sparse_global_term_weights(self, 
                                       input_file, 
                                       word_weighting, 
                                       vocabulary_size=None, 
                                       sparse_local_term_weights=None, 
                                       index_to_word=None, 
                                       word_to_index=None, 
                                       sum_words=None):        
        '''
        Apply a word_weighting method on the sparse_local_term_weights to create sparse_global_term_weights
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)
        
        Parameters:
             - input-file: list of lists of strings (tokens)
             - word_weighting: string that indicated the method used for word_weighting. Choose from:
                  - entropy
                  - normal
                  - idf
                  - probidf
             - vocabulary_size: int (positive)
             - sparse_local_term_weights: a sparse matrix showing the frequency of each words in documents. (dok matrix)
             - index_to_word: dictionary
                     keys: unique number associated to a word (int)
                     values: vocabulary words (str) 
             - word_to_index: dictionary
                    keys: vocabulary words (str)
                    values: unique number associated to a word (int)
             - sum_words: Counter object:
                     keys: vocabulary words (str)
                     values: the count of word in the corpus (int)
        '''
        num_documents = len(input_file)
        
        if word_weighting in ['entropy','normal']:
            if sparse_local_term_weights == None:
                raise ValueError("Please feed the algorithm 'sparse_local_term_weights'")
                
        if word_weighting in ['entropy']:
            if index_to_word == None:
                raise ValueError("Please feed the algorithm 'index_to_word'")
            if sum_words == None:
                raise ValueError("Please feed the algorithm 'sum_words'")
        
        if word_weighting in ['entropy', 'idf', 'probidf']:
            if vocabulary_size == None:
                raise ValueError("Please feed the algorithm 'vocabulary_size'")
                
        if word_weighting in ['idf', 'probidf']:
            if word_to_index == None:
                raise ValueError("Please feed the algorithm 'word_to_index'")
        
        if word_weighting ==  'entropy':
            global_term_weights = self._calculate_entropy(num_documents, vocabulary_size, sparse_local_term_weights, index_to_word, sum_words)
        elif word_weighting == 'idf':
            global_term_weights = self._calculate_idf(num_documents, vocabulary_size, input_file, word_to_index)
        elif word_weighting == 'normal':
            global_term_weights = self._calculate_normal(sparse_local_term_weights)
        elif word_weighting == 'probidf':
            global_term_weights = self._calculate_probidf(num_documents, vocabulary_size, input_file, word_to_index)
        else:
            raise ValueError('Invalid word weighting method')
        return sparse_local_term_weights.multiply(global_term_weights).tocsc() #Multiply local weighting with global weighting 
       
    def _calculate_entropy(self, 
                           num_documents, 
                           vocabulary_size, 
                           sparse_local_term_weights, 
                           index_to_word, 
                           sum_words):
        '''
        Use the entropy word weightingg method.
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)
        
        Parameters:
             - num_documents: int (positive)
             - vocabulary_size: int (positive)
             - sparse_local_term_weights:a sparse matrix showing the frequency of each words in documents. (dok matrix)
             - index_to_word: dictionary
                     keys: unique number associated to a word (int)
                     values: vocabulary words (str) 
             - sum_words: Counter object:
                     keys: vocabulary words (str)
                     values: the count of word in the corpus (int)
        
        Returns:
            np.array
        ''' 
        p_log_p_ij = self._create_p_log_p_ij(num_documents, vocabulary_size, sparse_local_term_weights, index_to_word, sum_words)
        summed_p_log_p = p_log_p_ij.sum(0).tolist()[0]
        return np.array([1+ summed_p_log_p_i /np.log2(num_documents) for summed_p_log_p_i in summed_p_log_p])
    
    def _calculate_idf(self, num_documents, vocabulary_size, input_file, word_to_index):
        '''
        Use the idf word weightingg method.
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)
        
        Parameters:
             - num_documents: int (positive)
             - vocabulary_size: int (positive)
             - input-file: list of lists of strings (tokens)
             - word_to_index: dictionary
                    keys: vocabulary words (str)
                    values: unique number associated to a word (int)
        Returns:
            np.array
        '''    
        binary_sparse_dtm = self._create_sparse_binary_dtm(num_documents, vocabulary_size, input_file, word_to_index)
        summed_words = binary_sparse_dtm.sum(0).tolist()[0]
        return np.array([np.log2(num_documents/word_count) for word_count in summed_words])   
    
    def _calculate_normal(self, sparse_local_term_weights):
        '''
        Use the normal word weightingg method.
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)
        
        Parameters:
             - sparse_local_term_weights:a sparse matrix showing the frequency of each words in documents. (dok matrix)
    
        Returns:
            np.array    
        '''    
        squared_dtm = sparse_local_term_weights.multiply(sparse_local_term_weights)
        summed_words = squared_dtm.sum(0).tolist()[0]
        return np.array([1/(math.sqrt(word_count)) for word_count in summed_words])
        
    def _calculate_probidf (self, num_documents, vocabulary_size, input_file, word_to_index):
        '''
        Use the probidf word weightingg method.
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)
        
        Parameters:
             - num_documents: int (positive)
             - vocabulary_size: int (positive)
             - input-file: list of lists of strings (tokens)
             - word_to_index: dictionary
                    keys: vocabulary words (str)
                    values: unique number associated to a word (int)
        Returns:
            np.array    
        '''   
        binary_sparse_dtm = self._create_sparse_binary_dtm(num_documents, vocabulary_size, input_file, word_to_index)
        summed_binary_words_list = binary_sparse_dtm.sum(0).tolist()[0]
        return np.array([np.log2((num_documents - binary_word_count)/binary_word_count) for binary_word_count in summed_binary_words_list])
    
    def _create_p_log_p_ij (self, num_documents, 
                            vocabulary_size, 
                            sparse_local_term_weights, 
                            index_to_word, 
                            sum_words):
        '''
        Create probability of word i in document j, multiplied by its base-2 logarithm. Format: sparse matrix - used for entropy
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)
        
        Parameters:
             - num_documents: int (positive)
             - vocabulary_size: int (positive)
             - sparse_local_term_weights:a sparse matrix showing the frequency of each words in documents. (dok matrix)
             - index_to_word: dictionary
                     keys: unique number associated to a word (int)
                     values: vocabulary words (str) 
            - sum_words: Counter object:
                     keys: vocabulary words (str)
                     values: the count of word in the corpus (int)
        
        Returns:
             - dok-matrix    
        '''
        p_log_p_ij = dok_matrix((num_documents, vocabulary_size), dtype=np.float32)
        for j in range(num_documents):
            row_counts = sparse_local_term_weights.getrow(j).toarray()[0]
            word_index = row_counts.nonzero()[0]
            non_zero_row_counts = row_counts[row_counts != 0]
            for i, count in enumerate(non_zero_row_counts):
                word = index_to_word[word_index[i]]
                prob_ij = count/sum_words[word]
                p_log_p_ij[j,word_index[i]] = prob_ij * np.log2(prob_ij)
        return p_log_p_ij
    
    def _create_sparse_binary_dtm(self, num_documents, vocabulary_size, input_file, word_to_index):
        '''
        Create a binary sparse dtm. Format: sparse matrix - used for idf and probidf
        (See: https://link.springer.com/article/10.1007/s40815-017-0327-9)
        
        Parameters:
            - num_documents: int (positive)
            - vocabulary_size: int (positive)
            - input-file: list of lists of strings (tokens)
            - word_to_index: dictionary
                    keys: vocabulary words (str)
                    values: unique number associated to a word (int)
        
        Returns:
             - dok-matrix
        '''
        binary_sparse_dtm = dok_matrix((num_documents, vocabulary_size), dtype=np.float32)
        for doc_index, document in enumerate(input_file):
            binary_document_counter = dict.fromkeys(document, 1)
            for word in set(document):
                binary_sparse_dtm[doc_index, word_to_index[word]] = binary_document_counter[word]    
        return binary_sparse_dtm
    
    def get_projected_data(self, algorithm, sparse_weighted_matrix, svd_factors):
        '''
        Perform singular decomposition for dimensionality reduction. 
        (See: https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm)
        
        For SVD on sparse data, the sparsesvd package is used (https://pypi.org/project/sparsesvd/)
        
        Depending on the algorithm, a different matrix from SVD is returned.
        
        Parameters:
             - algorithm: str ('flsa' or 'flsa-w')
             - sparse_weighted_matrix: the ouput from self.get_sparse_global_term_weights
             - svd_factors: int (positive): number of singular values to include
             
        Returns:
            - Numpy array (2d) - shape depends on algorithm
        '''
        ut, _, v = sparsesvd(sparse_weighted_matrix, svd_factors)
        if algorithm == 'flsa':
            return ut.T
        elif algorithm == 'flsa-w':
            return v.T
        else: 
            raise ValueError('Invalid algorithm selected. Only "flsa" ans "flsa-w" are currently supported.')
   
    def get_partition_matrix(self, data, number_of_clusters, method = 'fcm'):
        '''
        Perform clustering on the projected data. 
        For clustering the pyFUME package is used (https://pyfume.readthedocs.io/en/latest/Clustering.html)
        (https://ieeexplore.ieee.org/abstract/document/9177565?casa_token=PuwV0p6mhRUAAAAA:hOMWmjx-mDpG5BsuVfbyUFhZ5yZDtWwowmRgZFIhhHWq308z5x9lCCaESuiYhTyhQGIx9vV7aQ)
        
        Parameters:
             - data: Numpy array - the output from self.get_projected_data()
             - number of clusters: int (positive)
             - method: str - cluster method, choose from: 'fcm', 'gk', 'fst-pso'
            
        Returns:
            - Numpy array (shape: vocabulary size x number of topics)
        '''
        clusterer = Clustering.Clusterer(nr_clus = number_of_clusters, data= data)
        _, partition_matrix, _ = clusterer.cluster(method= method)
        return partition_matrix
    
    def get_prob_document_j(self, sparse_matrix):
        '''
        Get the probability of document j - used for calculations later on. 
        
        Parameters:
            - sparse_matrix: a sparse matrix (dok in this case)
            
        Returns:
            Numpy array (shape: number of documents x 1)
        '''
        document_sum = np.array([doc[0] for doc in sparse_matrix.sum(1).tolist()]) #vector with the length of num_document, each cell represents the sum of all weights of a document
        total_sum_d = sum(sparse_matrix.sum(0).tolist()[0]) #sum of all the elements in the weighted matrix
        return document_sum/total_sum_d #normalized probability
    
    def get_prob_word_i(self, sparse_matrix):
        '''
        Get the probability of word i - used for calculations later on. 
        
        Parameters:
            - sparse_matrix: a sparse matrix (dok in this case)
            
        Returns:
            Numpy array (shape: vocabulary_size x 1)
        '''
        word_sum = np.array(sparse_matrix.sum(0).tolist())
        total_sum_w = sum(sparse_matrix.sum(0).tolist()[0]) #sum of all the elements in the weighted matrix
        return (word_sum / total_sum_w)[0] #normalized probability
    
    def get_prob_topic_k(self, prob_topic_given_word_transpose, prob_word_i):
        '''
        Get the probability of topic k - used for calculations later on. 
        
        Parameters:
            - prob_topic_given_word_transpose: Numpy array (2d) - the output from self.get_partition_matrix()
            - prob_word_i: Numpy array 1d - the output from self.get_prob_word_i()
        
        Returns:
            Numpy array (shape: 1 x number of topics)
        '''
        return np.matmul(prob_topic_given_word_transpose.T,prob_word_i)

    def get_probability_matrices(self, algorithm, 
                                 prob_topic_given_document_transpose = None, 
                                 prob_topic_given_word_transpose = None, 
                                 local_term_weights = None, 
                                 global_term_weights = None, 
                                 all_matrices = False):
        '''
        Large method that performs matrix multiplications to obtain the desired output matrices. 
        
        Parameters: there is one generic parameter, the other parameters passed into this method depend on the used algorithm.
        
            - generic:
                 - all_matrices: boolean - if false: two matrices will be returned. If true, 8 matrices will be returned. 
        
            - flsa:
                 - prob_topic_given_document_transpose: Numpy array (2d) - the output from self.get_partition_matrix()
                 - global_term_weights: sparse matrix - the output from self.get_sparse_global_term_weights()
            - flsa-w:
                 - prob_topic_given_word_transpose: Numpy array (2d) - the output from self.get_partition_matrix()
                 - global_term_weights: sparse matrix - the output from self.get_sparse_global_term_weights()
            - flsa-v:
                 - prob_topic_given_word_transpose: Numpy array (2d) - the output from self.get_partition_matrix()
                 - local_term_weights: sparse matrix - the output from self.sparse_document_term_matrix()
                
        Returns: 
             - if all_matrices = False (default):
                  - Numpy array: prob_word_given_topic
                  - Numpy array: prob_topic_given_document
            
             - if all_matrices = True:
                  - prob_word_i
                  - prob_document_j
                  - prob_topic_k
                  - prob_document_and_topic
                  - prob_document_given_topic
                  - prob_word_given_document
                  - prob_word_given_topic
                  - prob_topic_given_document
        '''
        #Chech whether the right variable are passed into the method.
        if algorithm == 'flsa':
            if prob_topic_given_document_transpose is None:
                raise ValueError("Please feed the method 'prob_topic_given_document_transpose' to run flsa")
            if global_term_weights is None:
                raise ValueError("Please feed the method 'global_term_weights', to run flsa")
        elif algorithm == 'flsa-w':
            if prob_topic_given_word_transpose is None:
                raise ValueError("Please feed the method 'prob_topic_given_word_transpose' to run flsa-w")
            if global_term_weights is None:
                raise ValueError("Please feed the method 'global_term_weights', to run flsa-w")
        elif algorithm == 'flsa-v':
            if prob_topic_given_word_transpose is None:
                raise ValueError("Please feed the method 'prob_topic_given_word_transpose' to run flsa-v")
            if local_term_weights is None:
                raise ValueError("Please feed the method 'local_term_weights', to run flsa-v")
        else:
            raise ValueError('Your algorithm is currently not supported')
        
        #Calculate the initial probabilities
        if algorithm == 'flsa' or algorithm == 'flsa-w':
            prob_word_i = self.get_prob_word_i(global_term_weights)
            prob_document_j = self.get_prob_document_j(global_term_weights)
            if algorithm == 'flsa-w':
               prob_topic_k = self.get_prob_topic_k(prob_topic_given_word_transpose, prob_word_i)
        elif algorithm == 'flsa-v':
            prob_word_i = self.get_prob_word_i(local_term_weights)
            prob_document_j = self.get_prob_document_j(local_term_weights)
            prob_topic_k = self.get_prob_topic_k(prob_topic_given_word_transpose, prob_word_i)
        
        if algorithm == 'flsa':
            prob_document_and_topic = (prob_topic_given_document_transpose.T * prob_document_j).T
            prob_document_given_topic = prob_document_and_topic / prob_document_and_topic.sum(axis=0)
            prob_word_given_document = np.asarray(global_term_weights / global_term_weights.sum(1))
            self.prob_word_given_topic = np.matmul(prob_word_given_document.T, prob_document_given_topic)
            self.prob_topic_given_document = prob_topic_given_document_transpose.T
            if all_matrices:
                return prob_word_i, prob_document_j, prob_topic_k, prob_document_and_topic, prob_document_given_topic, prob_word_given_document, self.prob_word_given_topic, prob_topic_given_document_transpose.T
            else:
                return self.prob_word_given_topic, self.prob_topic_given_document
                       
        elif algorithm == 'flsa-w' or algorithm == 'flsa-v':
            prob_word_and_topic = (prob_topic_given_word_transpose.T * prob_word_i).T
            self.prob_word_given_topic = prob_word_and_topic / prob_word_and_topic.sum(axis=0)
            
            if algorithm == 'flsa-w':
                prob_word_given_document = np.asarray(global_term_weights / global_term_weights.sum(1)).T
            elif algorithm == 'flsa-v':
                prob_word_given_document = np.asarray(local_term_weights / local_term_weights.sum(1)).T
            
            prob_document_given_word = (prob_word_given_document*prob_document_j).T/np.array(prob_word_i)
            prob_document_given_topic = np.matmul(prob_document_given_word, self.prob_word_given_topic)#Calculate P(D_j | T_k)
            self.prob_topic_given_document = (prob_document_given_topic * prob_topic_k).T/prob_document_j
            
            if all_matrices:
                return prob_word_i, prob_document_j, prob_topic_k, prob_document_and_topic, prob_document_given_topic, prob_word_given_document, self.prob_word_given_topic, self.prob_topic_given_document
            else:
                return self.prob_word_given_topic, self.prob_topic_given_document

    def show_topics(self, prob_word_given_topic = None, num_words = None, index_to_word = None, representation = 'both'):
        '''
        Show the top-n word associated with each topic.
        
        Parameters:
             - prob_word_given_topic: Numpy array (2d)
             - num_words: int (positive) - indicates how many words per topic should be shown.
             - index_to_word: dictionary
                     keys: unique number associated to a word (int)
                     values: vocabulary words (str)
             - representation: str (either 'both' or 'words') - indicates wheter only words are returned or a combination of words and probabilities per word.
        
        Returns:
            - list of tuples (topic_index, all words (and probabilities))
        '''        
        if prob_word_given_topic == None:
            prob_word_given_topic = self.prob_word_given_topic
        if num_words == None:
            num_words = self.num_words
        if index_to_word == None:
            index_to_word = self.index_to_word
        if representation not in ['both', 'words']:
            raise ValueError("Invalid representation. Choose between 'both' or 'words'")
        if not isinstance(prob_word_given_topic,np.ndarray):
            raise TypeError("Please feed the algorithm 'prob_word_given_topic' as a np.ndarray")
        if not isinstance(index_to_word, dict):
            raise TypeError("Please feed the algorithm 'index_to_word' as a dict")
        if not isinstance(num_words, int) or num_words <= 0:
            raise TypeError("Please use a positive int for 'num_words'.")
        if prob_word_given_topic.shape[0] < prob_word_given_topic.shape[1]:
            raise ValueError("'prob_word_given_topic' has more columns then rows, probably you need to take the transpose.")
        if prob_word_given_topic.shape[0] != len(index_to_word.keys()):    
            warnings.warn("It seems like 'prob_word_given_topic' and 'index_to_word' are not aligned. The number of vocabulary words in 'prob_word_given_topic' deviate from the number of words in 'index_to_word'.")
        
        topic_list = []
        if representation == 'both':
            for topic_index in range(prob_word_given_topic.shape[1]):
                weight_words = ""
                sorted_highest_weight_indices = prob_word_given_topic[:,topic_index].argsort()[-num_words:][::-1]
                for word_index in sorted_highest_weight_indices:
                    weight_words += str(round(prob_word_given_topic[word_index,topic_index],4)) + '*"' + index_to_word[word_index] + '" + '
                topic_list.append((topic_index, weight_words[:-3]))
            return topic_list
        elif representation == 'words':
            for topic_index in range(prob_word_given_topic.shape[1]):
                word_list = []
                sorted_highest_weight_indices = prob_word_given_topic[:,topic_index].argsort()[-num_words:][::-1]
                for word_index in sorted_highest_weight_indices:
                    word_list.append(index_to_word[word_index])
                topic_list.append(word_list)
            return topic_list
        
    def get_topic_embedding(self, input_file, prob_word_given_topic=None, method = 'topn', topn = 20, perc=0.05):
        '''
        Create a topic embedding for each input document, to be used as input to predictive models. 
        
        Parameters:
            - input_file: list of list with strings (tokens) - corpus for which topic embeddings are created.
            - prob_word_given_topic: Numpy array (matrix showing the probability that a word appears in a topic)
            - method: method to select words to be included in the embedding (choose from 'topn', 'percentile'):
                    - topn: for each topic the top n words with the highest probability are included
                    - percentile: for each topic all words with highest probabilities are assigned while the cumulative probability is lower than the percentile.
            - topn: int (positive) - top n words to include (needs only to be used when 'method=topn'
            - perc: float (between 0 and 1): benchmark percentile until which words need to be added.
        
        Returns:
            - Numpy array (number of documents x number of topics) - each row gives the topic embedding for the associated document. 
        '''
        self._check_data_format_input_data(input_file)
        if prob_word_given_topic == None:
            prob_word_given_topic = self.prob_word_given_topic
        top_dist = []
        if method not in ['topn','percentile']:
            raise ValueError(method, "is not a valid option for 'method'. Choose either 'topn' or 'percentile'") 
        if method == 'topn':
            create_weights = self._create_dictlist_topn(topn, prob_word_given_topic, self.index_to_word)
        else:
            create_weights = self._create_dictlist_percentile__(perc, prob_word_given_topic, self.index_to_word)
        for doc in input_file:
            topic_weights = [0] * prob_word_given_topic.shape[1]
            for word in doc:
                for i in range(prob_word_given_topic.shape[1]):
                    topic_weights[i] += create_weights[i].get(word, 0)
            top_dist.append(topic_weights)
        return np.array(top_dist)
    
    def _create_dictlist_topn(self, topn, prob_word_given_topic, index_to_word):
        '''
        Create a list with dictionaries. 
         - Keys: all the indices of words from prob_word_given_topic who's weight's are amongst the top percentage. 
         - Values: the probability associated to a word.
        
        Parameters:
             - topn: int (positive) - top n words to include (needs only to be used when 'method=topn'
             - prob_word_given_topic: Numpy array (matrix showing the probability that a word appears in a topic)
             - index_to_word: dictionary
                     keys: unique number associated to a word (int)
                     values: vocabulary words (str) 
        Returns: 
             - list (length: num_topics) with dictionaries with keys: words, values: probabilities
        '''
        if not isinstance(topn, int) and topn >0:
            raise ValueError("Please choose a positive integer for 'topn'")
        top_dictionaries = []       
        for topic_index in range(prob_word_given_topic.shape[1]):
            new_dict = dict()    
            highest_weight_indices = prob_word_given_topic[:,topic_index].argsort()[-topn:]
            for word_index in highest_weight_indices:
                new_dict[index_to_word[word_index]] = prob_word_given_topic[word_index,topic_index]
            top_dictionaries.append(new_dict)
        return top_dictionaries
    

    def _create_dictlist_percentile__(self, perc, prob_word_given_topic, index_to_word):
        '''
        Create a list with dictionaries. 
         - Keys: all the indices of words from prob_word_given_topic who's weight's are amongst the top percentage. 
         - Values: the probability associated to a word.
             
        Parameters:
             - topn: int (positive) - top n words to include (needs only to be used when 'method=topn'
             - prob_word_given_topic: Numpy array (matrix showing the probability that a word appears in a topic)
             - index_to_word: dictionary
                     keys: unique number associated to a word (int)
                     values: vocabulary words (str) 
        Returns: 
             - list (length: num_topics) with dictionaries with keys: words, values: probabilities
        '''
        if not isinstance(perc, float) and perc >= 0 and perc <= 1:
            raise ValueError("Please choose a number between 0 and 1 for 'perc'")
        top_list = []
        for top in range(prob_word_given_topic.shape[1]):
            new_dict = dict()
            count = 0
            i = 0
            weights = np.sort(prob_word_given_topic[:,top])[::-1]
            word_indices = np.argsort(prob_word_given_topic[:,top])[::-1]
            while count < perc:
                new_dict[index_to_word[word_indices[i]]] = weights[i]
                count+=weights[i]
                i+=1
            top_list.append(new_dict)
        return top_list
        
    def get_coherence_value(self, input_file, topics, coherence = 'c_v'):
        '''
        Calculate the coherence score for the generated topic
        
        Parameters:
             - input_file: list of list with strings (tokens) - corpus based on which the topic model is trained.
             - topics: list of lists of tokens - the words per topics, equivalent to self.show_topics(representation='words'))
             - coherence: str - the type of coherence to be calculated (choose from: 'u_mass', 'c_v', 'c_uci', 'c_npmi')
             
        Returns:
             - float - the coherence score
        '''
        id2word = corpora.Dictionary(input_file)
        topics = topics  
        corpus = [id2word.doc2bow(text) for text in input_file]
        self.coherence = CoherenceModel(topics=topics,texts = input_file, corpus=corpus, dictionary=id2word, coherence= coherence, topn=len(topics[0])).get_coherence()
        return self.coherence
    
    def get_diversity_score(self, topics):
        '''''
        Calculate the diversity score for the generated topic (number of unique words / number of total words)
        (see: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00325/96463/Topic-Modeling-in-Embedding-Spaces)
        
        Parameters:
             - topics: list of lists of tokens - the words per topics, equivalent to self.show_topics(representation='words'))
             
        Returns:
             - float - the diversity score
        '''        
        unique_words = set()
        total_words = 0
        for top in topics:
            unique_words.update(top)
            total_words += len(top)
        self.diversity_score = len(unique_words)/total_words
        return self.diversity_score
    
    def get_interpretability_score(self, input_file, topics, coherence='c_v'):
        '''''
        Calculate the interpretability score for the generated topic (coherence * diversity)
        (see: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00325/96463/Topic-Modeling-in-Embedding-Spaces)
        
        Parameters:
             - input_file: list of list with strings (tokens) - corpus based on which the topic model is trained.
             - topics: list of lists of tokens - the words per topics, equivalent to self.show_topics(representation='words'))
             - coherence: str - the type of coherence to be calculated (choose from: 'u_mass', 'c_v', 'c_uci', 'c_npmi')
            
        Returns:
             - float - the interpretability score
        '''        
        
        try:
            self.coherence
        except AttributeError:
            self.coherence = self.get_coherence_value(input_file, topics)
        try:
            self.diversity_score
        except AttributeError:
            self.diversity_score = self.get_diversity_score(topics)
        return self.coherence * self.diversity_score
    
class FLSA(FuzzyTM):
    def __init__(self, input_file, num_topics, num_words=20, word_weighting='normal', svd_factors=2, cluster_method='fcm'):
        
        super().__init__(algorithm = 'flsa', 
                         input_file=input_file, 
                         num_topics=num_topics, 
                         num_words=num_words, 
                         word_weighting = word_weighting, 
                         cluster_method=cluster_method, 
                         svd_factors=svd_factors)
        
    def get_matrices(self, all_matrices = False):
        sparse_document_term_matrix = self.get_sparse_local_term_weights(self.input_file,
                                                                         self.vocabulary_size, 
                                                                         self.word_to_index)
        
        sparse_global_term_weighting = self.get_sparse_global_term_weights(input_file = self.input_file, 
                                                                           word_weighting = self.word_weighting, 
                                                                           vocabulary_size=self.vocabulary_size, 
                                                                           sparse_local_term_weights = sparse_document_term_matrix, 
                                                                           index_to_word = self.index_to_word, 
                                                                           word_to_index = self.word_to_index, 
                                                                           sum_words = self.sum_words)
        projected_data = self.get_projected_data(algorithm = 'flsa',
                                                 sparse_weighted_matrix = sparse_global_term_weighting, 
                                                 svd_factors = self.svd_factors)
        
        partition_matrix = self.get_partition_matrix(data = projected_data, 
                                                          number_of_clusters = self.num_topics, 
                                                          method = self.cluster_method)
        return self.get_probability_matrices(algorithm='flsa', 
                                             prob_topic_given_document_transpose = partition_matrix, 
                                             global_term_weights = sparse_global_term_weighting, 
                                             all_matrices = all_matrices )

class FLSA_W(FuzzyTM):
    def __init__(self, input_file, num_topics, num_words = 20, word_weighting = 'normal', svd_factors = 2, cluster_method = 'fcm'):
        '''
        Method to run after the FLSA class has been initialized to obtain the output matrices. 
        
        Parameters:
             - all_matrices: boolean (default: False) - indicates whether all matrices should be returned or only the two default matrices. 
        
        Returns: 
             - if all_matrices = False (default):
                  - Numpy array: prob_word_given_topic
                  - Numpy array: prob_topic_given_document
            
             - if all_matrices = True:
                  - prob_word_i
                  - prob_document_j
                  - prob_topic_k
                  - prob_document_and_topic
                  - prob_document_given_topic
                  - prob_word_given_document
                  - prob_word_given_topic
                  - prob_topic_given_document        
        '''        
        super().__init__(algorithm = 'flsa-w', 
                         input_file=input_file, 
                         num_topics=num_topics, 
                         num_words=num_words, 
                         word_weighting = word_weighting, 
                         cluster_method=cluster_method, 
                         svd_factors=svd_factors)

    def get_matrices(self, all_matrices = False):
        '''
        Method to run after the FLSA_W class has been initialized to obtain the output matrices. 
        
        Parameters:
             - all_matrices: boolean (default: False) - indicates whether all matrices should be returned or only the two default matrices. 
        
        Returns: 
             - if all_matrices = False (default):
                  - Numpy array: prob_word_given_topic
                  - Numpy array: prob_topic_given_document
            
             - if all_matrices = True:
                  - prob_word_i
                  - prob_document_j
                  - prob_topic_k
                  - prob_document_and_topic
                  - prob_document_given_topic
                  - prob_word_given_document
                  - prob_word_given_topic
                  - prob_topic_given_document        
        '''        
        sparse_document_term_matrix = self.get_sparse_local_term_weights(self.input_file, 
                                                                              self.vocabulary_size, 
                                                                              self.word_to_index)
        
        sparse_global_term_weighting = self.get_sparse_global_term_weights(input_file = self.input_file, 
                                                                           word_weighting = self.word_weighting, 
                                                                           vocabulary_size=self.vocabulary_size, 
                                                                           sparse_local_term_weights = sparse_document_term_matrix, 
                                                                           index_to_word = self.index_to_word, 
                                                                           word_to_index = self.word_to_index, 
                                                                           sum_words = self.sum_words)
        
        projected_data = self.get_projected_data(algorithm = 'flsa-w',
                                                      sparse_weighted_matrix = sparse_global_term_weighting, 
                                                      svd_factors = self.svd_factors)
        
        partition_matrix = self.get_partition_matrix(data = projected_data, 
                                                     number_of_clusters = self.num_topics, 
                                                     method = self.cluster_method)
        
        return self.get_probability_matrices(algorithm='flsa-w', 
                                             prob_topic_given_word_transpose = partition_matrix, 
                                             global_term_weights = sparse_global_term_weighting, 
                                             all_matrices = all_matrices )

class FLSA_V(FuzzyTM):
    def __init__(self, 
                 input_file, 
                 map_file, 
                 num_topics, 
                 num_words = 20, 
                 cluster_method='fcm'):
        self.map_file = map_file
        self.input_file = input_file
        if not isinstance(self.map_file, pd.DataFrame):
            raise TypeError("'map_file' should be a pd.DataFrame, but should be")
            
        if not 'x' in self.map_file.columns:
            raise ValueError("map_file has no 'x' column")
        if not 'y' in self.map_file.columns:
            raise ValueError("map_file has no 'y' column")
        if not 'id' in self.map_file.columns:
            raise ValueError("map_file has no 'id' column")
            
        self.filtered_input = self.get_filtered_input(self.input_file, self.map_file)
        super().__init__(algorithm = 'flsa-v', 
                         input_file=self.filtered_input, 
                         num_topics=num_topics, 
                         num_words=num_words, 
                         cluster_method=cluster_method)
        
        vocab = self.create_vocabulary(self.filtered_input)[0]
        self.coordinates = self.get_coordinates_from_map_file(self.map_file, vocab)

    def get_filtered_input(self, input_file, map_file):
        '''
        Filter out words from the input_file that are not present in the map_file
        
        Parameters:
             - input_file: list of list with strings (tokens) - corpus to train topic models on.
             - map_file: The output file from Vosviewer.
                 Format: pd.DataFrame (The Dataframe needs to contain the following columns: 'id','x','y')
        
        Returns:
             - list of list of tokens - a filtered input_file
        '''
        filtered_input_file = []
        for doc in input_file:
            filtered_input_file.append(list(np.intersect1d(map_file['id'], doc)))
        return filtered_input_file

        
    def get_coordinates_from_map_file(self, map_file, vocab):
        '''
        Filter words that are in the map_file but not in the input file, 
        retrieve the coordinates from the map_file and convert them into a Numpy array.
        
        Parameters:
             - map_file: The output file from Vosviewer.
                 Format: pd.DataFrame (The Dataframe needs to contain the following columns: 'id','x','y')
             - vocab: set 
        '''
        for word in map_file['id']:
            if word not in vocab:
                map_file = map_file[map_file.id != word]
        x = np.array(map_file['x'])
        y = np.array(map_file['y'])
        return np.array([x,y]).T


    def get_matrices(self, all_matrices = False):
        '''
        Method to run after the FLSA_V class has been initialized to obtain the output matrices. 
        
        Parameters:
             - all_matrices: boolean (default: False) - indicates whether all matrices should be returned or only the two default matrices. 
        
        Returns: 
             - if all_matrices = False (default):
                  - Numpy array: prob_word_given_topic
                  - Numpy array: prob_topic_given_document
            
             - if all_matrices = True:
                  - prob_word_i
                  - prob_document_j
                  - prob_topic_k
                  - prob_document_and_topic
                  - prob_document_given_topic
                  - prob_word_given_document
                  - prob_word_given_topic
                  - prob_topic_given_document        
        '''
        sparse_document_term_matrix = self.get_sparse_local_term_weights(self.input_file, 
                                                                              self.vocabulary_size, 
                                                                              self.word_to_index)
        
        partition_matrix = self.get_partition_matrix(data = self.coordinates, 
                                                          number_of_clusters = self.num_topics, 
                                                          method = self.cluster_method)
        
        return self.get_probability_matrices(algorithm='flsa-v', 
                                             prob_topic_given_word_transpose = partition_matrix, 
                                             local_term_weights = sparse_document_term_matrix, 
                                             all_matrices = all_matrices)