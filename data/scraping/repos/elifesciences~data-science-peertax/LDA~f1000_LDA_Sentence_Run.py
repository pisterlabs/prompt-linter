# Configuration
import csv
import sys
import pathlib
from pprint import pprint
import numpy as np
import pandas as pd
import re
from collections import defaultdict  # For word frequency
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.test.utils import datapath
import logging  # Setting up the loggings to monitor gensim
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
    
import pyLDAvis.gensim 

# Define iter
ITER = '0'

# Load and output folder settings
path_load_tsv = '../pickles/f1000_tokenized_LDA_sentence_'+ITER+'.tsv'
out_folder = './f1000_LDA_Sentence_Run_'+ITER+'/'
# Create folder if it doesn't exist.
pathlib.Path(out_folder).mkdir(exist_ok=True) 

# Define number of topics
num_topics = sys.argv[1]

# Logger settings
log_file_name = out_folder + 'gensim_' + num_topics +'.log'

handlers = [logging.FileHandler(log_file_name,mode='w'), logging.StreamHandler()]
logging.basicConfig(handlers = handlers,
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)
                    
# Load data from tsv
df = pd.read_csv(path_load_tsv,sep='\t',quoting=csv.QUOTE_NONE)
df.drop(columns=['Unnamed: 0'],inplace=True)
# Split the token column
df['token']=df['token'].str.split(',')

txt_dist = defaultdict(int)
j=0
for sent in df.token:
    for i in sent:
        txt_dist[i] += 1
print('\nNumber of unique words in dataset: ',len(txt_dist))

# Create Dictionary
id2word = corpora.Dictionary(df['token'])
print('\nDict size (no filter): ',len(id2word))
id2word.filter_extremes(no_below=20, no_above=0.75)
print('\nDict size (after filter): ',len(id2word))
# Create Corpus
texts = df['token']
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
print('\nNumber of unique tokens: %d' % len(id2word))
print('Number of documents: %d' % len(corpus))

# Set training parameters.
passes = 30
iterations = 1000
eval_every = 10  # For monitoring

lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=int(num_topics), 
                                           random_state=100,
                                           eval_every = eval_every,
                                           passes=passes,
                                           iterations = iterations,
                                           per_word_topics=True)
                                           
# Save model
file = out_folder + 'Model_' + num_topics
lda_model.save(file)
print('\nModel saved at location '+ file)

# Compute complexity score of model
print('\nPerplexity: ', lda_model.log_perplexity(corpus)) 
# Compute coherence scores (c_v and umass)
cv_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
cv_lda = cv_model_lda.get_coherence()
umass_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence="u_mass")
umass_lda = umass_model_lda.get_coherence()
logging.info('Coherence Score (C_V): %s' % cv_lda) 
logging.info('Coherence Score (UMass): %s' % umass_lda) 
print('\n')

#Get top topics and average topic coherence
top_topics = lda_model.top_topics(corpus)
pprint(top_topics)

# Save visualization as an html file
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
file_vis = out_folder + 'Model_' + num_topics + '.html'
pyLDAvis.save_html(vis, file_vis)
