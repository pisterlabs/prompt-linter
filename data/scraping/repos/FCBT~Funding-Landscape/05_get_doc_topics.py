#!/usr/bin/env python3

""" Calculates document topic probability matrix

argv[1]: where to get the corpus from
argv[2]: where to get the LDA from
argv[3]: where to save the topic probabilities

Example:  python3 code/05_get_doc_topics.py ./clean-data/fine-scale/'country_name' ./results/fine-scale/'country_name'/lda-models/'model_name' ./results/fine-scale/'country_name'/ 
"""


__appname__ = '[05_get_doc_topics.py]'
__author__ = 'Flavia C. Bellotto-Trigo (flaviacbtrigo@gmail.com)'
__version__ = '0.0.1'

## imports ##
import sys
import gensim
import os
import numpy as np
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from gensim.models import TfidfModel
from gensim.models import CoherenceModel

def main(argv):
    #read LDA + corpus
    corpus = gensim.corpora.MmCorpus(os.path.join(argv[1],"corpus.mm"))
    lda = gensim.models.ldamulticore.LdaMulticore.load(argv[2], "model_topic-number_topics")
    
    lda.minimum_probability = 0.0

    #allocate matrix
    N = 10
    doc_topic_matrix = np.zeros((lda.num_topics, N))

    #calculate distances
    for i in range(0,N):
        topic_probs = lda.get_document_topics(corpus[i])
        for j in topic_probs:
            doc_topic_matrix[j[0],i] = j[1]

    np.save(os.path.join(argv[3], "doc_topic_mat.npy"), doc_topic_matrix)

    return(0)


if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)