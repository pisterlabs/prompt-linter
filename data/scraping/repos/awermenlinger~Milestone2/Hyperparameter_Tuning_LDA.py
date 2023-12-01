import numpy as np
import pandas as pd
import tqdm
import gensim
from gensim.models import CoherenceModel
import pickle
import logging
#import nltk
#nltk.download('wordnet')
from multiprocessing import Process, freeze_support

#Some code inspired from https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 & 
# https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
# https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0

if __name__ == '__main__':
   freeze_support()
   # SETTINGS FOR MODEL
   RANDOM_SEED = 7245
   #chunk_size = 5000
   dic_file = "models/trained_lda_dictionary.sav"
   corp_file = "models/trained_lda_corpus.sav"
   model_file = "models/trained_lda.sav"
   text_file = "models/trained_lda_texts.sav"

   print ("Loading the dic, corpus and model")
   dictionary = pickle.load(open(dic_file, 'rb'))
   corpus = pickle.load(open(corp_file, 'rb')) 
   texts = pickle.load(open(text_file, 'rb')) 

   # supporting function
   def compute_coherence_values(corpus, dictionary, k, a, b):
      
      lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                             id2word=dictionary,
                                             num_topics=k, 
                                             random_state=100,
                                             chunksize=100,
                                             passes=10,
                                             alpha=a,
                                             eta=b)
      
      coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
      
      return coherence_model_lda.get_coherence()

   grid = {}
   grid['Validation_Set'] = {}
   # Topics range
   min_topics = 15
   max_topics = 23
   step_size = 2
   topics_range = range(min_topics, max_topics, step_size)
   # Alpha parameter
   alpha = [0.01, 0.5, 1, 'symmetric', 'asymmetric']
   # Beta parameter
   beta = [0.01, 0.5, 1,'symmetric']
   # Validation sets
   num_of_docs = len(corpus)
   corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
                  # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
                  gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
                  corpus]
   corpus_title = ['75% Corpus', '100% Corpus']
   model_results = {'Validation_Set': [],
                  'Topics': [],
                  'Alpha': [],
                  'Beta': [],
                  'Coherence': []
                  }
   # Can take a long time to run
   if 1 == 1:
      pbar = tqdm.tqdm(total=60)
      
      # iterate through validation corpuses
      for i in range(len(corpus_sets)):
         # iterate through number of topics
         for k in topics_range:
               # iterate through alpha values
               for a in alpha:
                  # iterare through beta values
                  for b in beta:
                     # get the coherence score for the given parameters
                     cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=dictionary, 
                                                   k=k, a=a, b=b)
                     # Save the model results
                     model_results['Validation_Set'].append(corpus_title[i])
                     model_results['Topics'].append(k)
                     model_results['Alpha'].append(a)
                     model_results['Beta'].append(b)
                     model_results['Coherence'].append(cv)
                     
                     pbar.update(1)
      pd.DataFrame(model_results).to_csv('results/lda_tuning_results.csv', index=False)
      pbar.close()