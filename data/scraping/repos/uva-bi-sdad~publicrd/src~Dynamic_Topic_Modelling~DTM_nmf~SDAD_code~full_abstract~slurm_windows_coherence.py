import pandas as pd
import numpy
import pickle
import time
import joblib
import gensim
import matplotlib.pyplot as plt

from itertools import islice
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger
from gensim.models.coherencemodel import CoherenceModel

# Remove warnings
import warnings
warnings.filterwarnings("ignore")


# Load the dataset.
df = pd.read_pickle("/project/biocomplexity/sdad/projects_data/ncses/prd/Paper/FR_meta_and_final_tokens_23DEC21.pkl")
df.head()

# Compute the time variable
year = df['FY'].unique()
del df


# Find topics that maximise the coherence for each windows
path = '/project/biocomplexity/sdad/projects_data/ncses/prd/Dynamic_Topics_Modelling/nmf_fullabstract/'
n_topics = list(range(20,61,5))
model_index = []
coherence = []
max_coherence = []
    
for fy in year:
    # upload the result that are necessary for the coherence
    topics_list = joblib.load( path+'nmf_out/windows_nmf'+str(fy)+'.pkl' )[1]
    docs = joblib.load( path+'Term_docs_'+str(fy)+'.pkl' )[2]
    dictionary = joblib.load( path+'Term_docs_'+str(fy)+'.pkl' )[3]
    
    for t in range(0,len(n_topics)):
        term_rankings = topics_list[t]
        cm = CoherenceModel(topics=term_rankings, dictionary=dictionary, texts=docs, coherence='c_v', topn=10, processes=1)
        
        # get the coherence value
        coherence.append(cm.get_coherence())
        print("one step")
    
    # find the topics that maximize the coherence
    max_value = numpy.nanmax(coherence)
    index = coherence.index(max_value)
    model_index.append(index)
    max_coherence.append(max_value)
    print('------- solve for a year -------')
    
# Save the result from the first step
joblib.dump((model_index, max_coherence), path+'first_stage.pkl' )
    