import sys
import os
import gensim
import pandas as pd  
import numpy as np
from itertools import chain
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import subprocess
import shlex
import matplotlib.pyplot as plt
# %matplotlib inline

def pickle_df(df, pname):
  df.to_pickle(pname)
  print(f'Saved dataframe as "{pname}".')

def unpickle_df(pname, df):
  new_df = pd.read_pickle(pname)
  print(f'Loaded dataframe from "{pname}".')
  return new_df

def load_corpus(pkl):
  corpus = pd.read_pickle(pkl)
  corpus = corpus['tokens']
  return corpus

# build dictionary
def build_dict(corpus, no_below, no_above, keep_n):
  dictionary = gensim.corpora.Dictionary(corpus)
  dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
  return dictionary

# build Bag-of-Words corpus
def build_bow(corpus, dictionary):
  bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
  return bow_corpus

# build Term Frequency, Inverse Document Frequency corpus (TF-IDF)
def build_tfidf(bow_corpus):
  tfidf = models.TfidfModel(bow_corpus)
  corpus_tfidf = tfidf[bow_corpus]
  return corpus_tfidf

# build topic model
def train_lda(corpus, dictionary, n_workers, n_passes, n_topics):
  # Build Bag-of-Words corpus
  bow_corpus = build_bow(corpus, dictionary)
  # Build TF-IDF corpus
  corpus_tfidf = build_tfidf(bow_corpus)  
  n_features = len(list(dictionary.values()))
  print(f'Training model with {n_topics} topics, {n_passes} passes, and {n_features} features...')
  # lda_model = gensim.models.LdaModel(corpus_tfidf, num_topics=n_topics, id2word=dictionary, passes=n_passes)
  lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=n_topics, id2word=dictionary, passes=n_passes, workers=n_workers)
  return lda_model

# save the trained model
def subprocess_cmd(command):
  process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
  proc_stdout = process.communicate()[0].strip()
  print(proc_stdout)

def save_model(lda_model, dictionary):
  print("Saving trained model...")
  filename_model = 'tf-lda.model'
  filename_dict = 'tf-lda.dict'
  lda_model.save(filename_model)
  dictionary.save(filename_dict)
  subprocess_cmd('bash tar_model.sh')

# evaluate topic model metrics of cohesion and log perplexity
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

# perform grid training and compare using evaluation metrics
def grid_train(corpus_filename, n_topics, no_below, no_above, n_passes, n_workers, keep_n):
  corpus = load_corpus(corpus_filename)
  dictionary = build_dict(corpus, no_below, no_above, keep_n)
  lda_model = train_lda(corpus, dictionary, n_workers, n_passes, n_topics)
  save_model(lda_model, dictionary)

  processed_docs = pd.read_pickle(corpus_filename)
  processed_docs = processed_docs['tokens']
  filename_model = 'tf-lda.model'
  model = gensim.models.LdaModel.load(filename_model)
  filename_dict = 'tf-lda.dict'
  dictionary = corpora.Dictionary.load(filename_dict)
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
  tfidf = models.TfidfModel(bow_corpus)
  corpus_tfidf = tfidf[bow_corpus]
  n_features = len(list(dictionary.values()))
  print(f'Evaluating model trained with {n_topics} topics, n_passes={n_passes}, no_below={no_below}, no_above={no_above}, and {n_features} features...')
  print("Calculating Log Perplexity...")
  p_score = perplex(model, corpus_tfidf)
  print("Calculating Coherence (u_mass)...")
  c_umass_score = cohere_umass(model, corpus_tfidf, dictionary)
  print("Calculating Coherence (c_v)...")
  c_cv_score = cohere_cv(model, processed_docs, dictionary)
  
  # RESULTS FOR THIS TRIAL
  results = (p_score, c_umass_score, c_cv_score)
  return results

if __name__ == '__main__':

  grid_results = []
  trained_topics = []
  trained_passes = []
  trained_no_below = []

  # import training data
  corpus_filename = input("\nEnter name of training corpus including 'tokens' column with preprocessed text [.pkl]: ")

  # define range of number of topics to use for grid training
  topics_start = int(input("Enter lowest number of topics for model training, e.g. '3': "))
  topics_end = int(input("Enter highest number of topics for model training, e.g. '20': ")) 
  range_n_topics = range(topics_start, topics_end + 1)

  # get values for other hyperparameters
  nb = int(input("Enter value for `no_below` [terms must appear in >= {n} docs], e.g. '30': "))
  no_below = nb
  na = float(input("Enter value for `no_above` [terms must not appear in > {0.n}->{n}% of docs], e.g. '0.5' (50%): "))
  no_above = na
  nps = int(input("Enter value for `n_passes` [perform {n} passes over the corpus], e.g. '5': "))
  n_passes = nps
  nwk = int(input("Enter value for `n_workers` [use {n} CPUs], e.g. '2': "))
  n_workers = nwk
  kpn = int(input("Enter value for `keep_n` [max {n} features for dict], e.g. '100000': "))
  keep_n = kpn

  # print selected hyperparameters
  print("\nHYPERPARAMETERS SELECTED: ")
  print(f'range_n_topics = range({topics_start}, {topics_end+1})')
  print(f'no_below = {no_below}')
  print(f'no_above = {no_above}')
  print(f'n_passes = {n_passes}')
  print(f'n_workers = {n_workers}')
  print(f'keep_n = {keep_n}')

  # begin grid training
  print("\nInitializing grid training...\n")
  for i in range_n_topics:
  # for i in range_n_passes:
  # for i in range_no_below:
  # for i in range_no_above:
    n_topics = i
    # no_above = i
    # no_below = i
    # n_passes = i
    trial = grid_train(corpus_filename, n_topics, no_below, no_above, n_passes, n_workers, keep_n)
    print(f'RESULTS: {trial}\n')
    grid_results.append(trial)
    trained_topics.append(i)
    trained_passes.append(n_passes)

  # print results
  print("GRID SEARCH COMPLETE:")
  print(grid_results)

  # analyze results
  trained_params = [x for x in range_n_topics]
  for x, y in zip(trained_params, grid_results):	
    print(f'param_value={x}; results: {y}')	

  print()
  
  x_params = []
  y_results = []
  za = []
  zb = []
  zc = []

  # weigh and scale results from each metric for visualization
  for x, y in zip(trained_params, grid_results):
    avg_c = sum([y[2] for y in grid_results])/len(trained_params)
    a = abs(y[0]) / (avg_c *27)
    b = abs(y[1]) / (avg_c *9)
    c = abs(y[2])
    calc = c*2.5 - a - b
    x_params.append(x)
    y_results.append(calc)
    za.append(a)
    zb.append(b)
    zc.append(c)
  
  x = x_params
  y = y_results
  
  # display plot of evaluation metrics by parameter type (default: n_topics)
  plt.figure(figsize=(26,8))
  plt.plot(x,y, label='aggregate scores [HIGHER=BETTER]', linewidth=2)
  plt.plot(x,za, label='perplexity [ABS; LOWER=BETTER]')
  plt.plot(x,zb, label='u_mass [ABS; LOWER=BETTER]')
  plt.plot(x,zc, label='c_v [HIGHER=BETTER]', linewidth=4)
  plt.title('Topic Model Coherence & Perplexity')
  plt.xlabel('Parameters Tested')
  plt.ylabel('Scores')
  plt.grid(True)
  plt.legend(loc=4)
  plt.show()

  for a, b, c in zip(x_params, zc, y_results):
    if b == max(zc):
      print(f'best cv_score = param_value: {a}, score = {b}')
    if c == max(y_results):
      print(f'best agg_score = param_value: {a}, score = {c}')
