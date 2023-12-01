#!/usr/bin/env python3

""" Loads processed corpuses and tokenised text from the clean-data/fine-scale folder, loads fitted LDA models from models/fine-scale folder and 
saves coherence results into the results folder.
Corpuses and LDA model are needed to calculate the UMass coherence.
Corpuses, LDA model and tokenised text are necessary to calculate the other 3 measures of coherence. 

If run as a script, it takes three arguments: 
    1) the file path to the corpus
    2) the file path to the fitted LDA models
    3) the file path to save the results
Example:  python3 code/04_calculate_coherence.py ./clean-data/fine-scale/training-data ./results/fine-scale/lda-models/training-model ./results/fine-scale/coherence-scores/training-data
"""

__appname__ = '[04_calculate_coherence.py]'
__author__ = 'Flavia C. Bellotto-Trigo (flaviacbtrigo@gmail.com)'
__version__ = '0.0.2'

## imports ##
import sys
import gensim
import os
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from gensim.models import TfidfModel
from gensim.models import CoherenceModel
import pandas as pd
import psutil
import ast
import re
import logging

#logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


## functions ##
class LoadFiles(object):
    def __init__(self, dirname, corpus):
        self.dirname = dirname
        self.corpus = corpus
        
    
    def __iter__(self):
        # print((self.dirname))
        i = 0        
        for fname in os.listdir(self.dirname):

            if fname.endswith("topics"):
                print(i, " ", len(os.listdir(self.dirname)))
                i += 1
                
                lda = gensim.models.ldamulticore.LdaMulticore.load(os.path.join(self.dirname, fname))
                print("Calculating coherences for", fname)

                # UMass
                cm_umass = CoherenceModel(model=lda, corpus=self.corpus, coherence='u_mass', processes= 40)
                c_umass = cm_umass.get_coherence()
            
                yield(lda.num_topics, c_umass)
            



def main(argv):
    print(psutil.cpu_count())

    #load data
    print("Loading data")
    loaded_corpus = corpora.MmCorpus(os.path.join(argv[1], 'corpus.mm'))
    
    coherence_results = {"Topics":[], "umass":[]}
    
    #loop over
    for i in LoadFiles(argv[2], loaded_corpus):
        coherence_results["Topics"].append(i[0])
        coherence_results["umass"].append(i[1])




    # print("Fitting")
    df = pd.DataFrame.from_dict(coherence_results)
    df.to_csv(os.path.join(argv[3], 'calculated_coherence.csv'))


if __name__ == "__main__": 
    """Makes sure the "main" function is called from command line"""  
    status = main(sys.argv)
    sys.exit(status)
