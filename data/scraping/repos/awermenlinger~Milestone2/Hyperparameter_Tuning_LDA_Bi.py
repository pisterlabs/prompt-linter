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
   dic_file = "drive/MyDrive/Colab Notebooks/assets/models/bi_trained_lda_dictionary.sav"
   corp_file = "drive/MyDrive/Colab Notebooks/assets/models/bi_trained_lda_corpus.sav"
   model_file = "drive/MyDrive/Colab Notebooks/assets/models/bi_trained_lda.sav"
   text_file = "drive/MyDrive/Colab Notebooks/assets/models/bi_trained_lda_texts.sav"
   
   print ("Loading the dic, corpus and model")
   dictionary = pickle.load(open(dic_file, 'rb'))
   corpus = pickle.load(open(corp_file, 'rb')) 
   texts = pickle.load(open(text_file, 'rb')) 

   # supporting function
   def compute_coherence_values(corpus, dictionary, k, a, b):
      
      lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                             id2word=dictionary,
                                             num_topics=k, 
                                             random_state=RANDOM_SEED,
                                             chunksize=2000,
                                             passes=10)
      
      coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
      
      return coherence_model_lda.get_coherence()

   grid = {}
   grid['Validation_Set'] = {}
   # Topics range
   topics_range = [15,17,19,21,23] #testing to see if HDP is really giving good results
   # Alpha parameter - used the best from standard hypertuning (took too much time to retrain)
   alpha = [0.5]
   # Beta parameter - used the best from standard hypertuning (took too much time to retrain)
   beta = [0.5]
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
      pbar = tqdm.tqdm(total=8)
      
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
                     print(corpus_title[i], " | ", k, " | ", a, " | ", b, " | ", cv)


                     pbar.update(1)
      pd.DataFrame(model_results).to_csv('drive/MyDrive/Colab Notebooks/assets/results/lda_tuning_results_bigram.csv', index=False)
      pbar.close()