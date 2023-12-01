import pandas as pd
import numpy as np
import pickle
import time
import gensim
import decimal
import csv

# setting up our imports
from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger
from gensim.models.coherencemodel import CoherenceModel


# Import the final tokens
f = open('/project/biocomplexity/sdad/projects_data/ncses/prd/Tech-Report/case_studies/coronavirus_corpus.pkl', 'rb')
df = pickle.load(f)
f.close()

# Create the dictionary and the corpus
def createLDAvars(docs):

    # Create the variables needed for LDA from df[final_frqwds_removed]: dictionary (id2word), corpus
    
    # Create Dictionary
    id2word = gensim.corpora.Dictionary(docs)

    #Filter words to only those found in at least a set number of documents (min_appearances)
    id2word.filter_extremes(no_below=20, no_above=0.6)
    
    # filter out stop words - "use" already filtered out by previous line
    id2word.filter_tokens(bad_ids=[id2word.token2id['research'], id2word.token2id['study'], \
                               id2word.token2id['project']])

    # Create Corpus (Term Document Frequency)

    #Creates a count for each unique word appearing in the document, where the word_id is substituted for the word
    # corpus not need for c_v coherence
    corpus = [id2word.doc2bow(doc) for doc in docs]

    return id2word, corpus
    
# build the dictionary id2word
docs = df["final_tokens"]
[dictionary, corpus] = createLDAvars(docs)

# Create the time slice using the fiscal year
df['Year'] = df['FY']
time_slice = df['PROJECT_ID'].groupby(df['Year']).count()
time = list(time_slice.index)

# Find best chain variance parameter
chain_vec = list(range(0.01,0.2,10)) 
coherence_mean = []

for chain in chain_vec:
    # Run a DTM-LDA with the specifc chain variance
    ldaseq_chain = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=time_slice, num_topics=30, chain_variance=chain)
    
    # Compute the coherence for each model
    time_coherence = []
    
    for t in range(0,len(time)):
        topics_dtm = ldaseq_chain.dtm_coherence(time=t)
        cm = CoherenceModel(topics=topics_dtm, dictionary=dictionary, texts=docs, coherence='c_v', processes=30) 
        time_coherence.append(cm.get_coherence())
        
    # Compute the coherence serie for the model    
    coherence_tm = pd.Series(time_coherence, index =time)
    
    # Compute and store the average/median coherence
    coherence_mean.append(coherence_tm.mean())
    coherence_median.append(coherence_tm.median())

# Save the value
coherence_serie = pd.Series(coherence_mean, coherence_median, index =chain_vec)

# save the global coherence value in a pickle file
pickle.dump(coherence_serie, open('/project/biocomplexity/sdad/projects_data/ncses/prd/Dynamic_Topics_Modelling/LDA/Coherence_chain_30.pkl','wb'))