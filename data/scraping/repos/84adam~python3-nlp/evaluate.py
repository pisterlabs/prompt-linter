import os
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
import pandas as pd
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

# load pre-trained model and corpus (from current working directory)

def load_model():
  filepath = os.getcwd()  
  filename_model = filepath + '/' + 'tf-lda.model'
  filename_dict = filepath + '/' + 'tf-lda.dict'
  model = gensim.models.LdaModel.load(filename_model)
  dictionary = corpora.Dictionary.load(filename_dict)
  return model, dictionary

def load_corpus():
  # where processed_docs.pkl contains the lemmatized/tokenized version of each document in the corpus
  processed_docs = pd.read_pickle('processed_docs.pkl')
  dictionary = load_model()[1]
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
  tfidf = models.TfidfModel(bow_corpus)
  corpus = tfidf[bow_corpus]
  return corpus, processed_docs

def perplex(lda_model, corpus):
  perplexity = lda_model.log_perplexity(corpus)
  return perplexity

def cohere_umass(lda_model, corpus, dictionary):
  coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
  coherence_lda = coherence_model_lda.get_coherence()
  return coherence_lda

def cohere_cv(lda_model, texts, dictionary):
  coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  return coherence_lda

if __name__ == '__main__':
  topic_model = load_model()
  lda_model = topic_model[0]
  dictionary = topic_model[1]
  read_corpus = load_corpus()
  corpus = read_corpus[0]
  processed_docs = read_corpus[1]
  p_score = perplex(lda_model, corpus)
  c_umass_score = cohere_umass(lda_model, corpus, dictionary)
  c_cv_score = cohere_cv(lda_model, processed_docs, dictionary)
  print(f'Model Log Perplexity Score: {p_score}')
  print(f'Model Coherence Score (u_mass): {c_umass_score}')
  print(f'Model Coherence Score (c_v): {c_cv_score}')
