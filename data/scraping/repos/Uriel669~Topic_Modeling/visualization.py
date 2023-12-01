# %%
from collections import Counter
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import matplotlib.pyplot as plt
import wordcloud as WordCloud
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import os
import umap.plot
import matplotlib.patches as mpatches

# %%
#Defining all functions

def get_topic_words(token_list, labels, k=None):
  ''' Get topic within each topic form clustering results '''
  if k is None:
    k = len(np.unique(labels))
  #index_k_list = [*range(0,k,1)]
  topics = ['' for _ in range(k)]
  for i, c in enumerate(token_list):
    topics[labels[i]] += (' ' + ' '.join(c))
  word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
  # get sorted word counts
  word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True),word_counts))
  # get topics
  topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))
  
  return topics

def get_coherence(model, token_lists, measure='c_v'):
  ''' Get model coherence from gensim.models.coherencemodel
  : param model: Topic_Model object
  : param token_lists: token list of docs
  : param topics: topics as top words 
  : param measure: coherence metrics
  : return: coherence score '''

  if model.method == 'LDA':
    cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus = model.corpus, dictionary=model.dictionary, coherence = measure)
  else:
    topics = get_topic_words(token_lists, model.cluster_model.labels_)
    cm = CoherenceModel(topics=topics, texts = token_lists, corpus=model.corpus, dictionary=model.dictionary, coherence = measure)
    return cm.get_coherence()

def get_silhouette(model):
  ''' Get silhoulette score from model
  :param_model: Topic_model score
  :return: silhoulette score '''            
  if model.method == 'LDA':
    return
  lbs = model.cluster_model.labels_
  vec = model.vec[model.method]
  return silhouette_score(vec, lbs)

def plot_proj(embedding, lbs):
  '''
  Plot UMAP embeddings
  :param embedding: UMAP (or other) embeddings
  :param lbs: labels
  '''
  n = len(embedding)
  counter = Counter(lbs)
  fig1 = plt.figure("Clustered statements", figsize=(8, 6), dpi=80)
  for i in range(len(np.unique(lbs))):
    plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5, label='cluster {}: {:.2f}%'.format(i, counter[i] / n*100))
    plt.legend(loc='best')
    plt.grid(color='grey', linestyle='-', linewidth=0.25)
  plt.show()
  
  

def visualize(model):  ## added token_list, topic
  '''
  Visualize the result for the topic model by 2D embedding (UMAP)
  :param model: Topic_Model object
  '''
  if model.method == 'LDA':
      
    return
  reducer = umap.UMAP(n_components=2, metric='hellinger')
  print('Calculating UMAP projection...')
  vec_umap = reducer.fit_transform(model.vec[model.method])  
  print(vec_umap)             
  print('Calculating the Umap projection. Done!')
  plot_proj(vec_umap, model.cluster_model.labels_) 

 
  
  #umap.plot.points(vec_umap, labels= model.cluster_model.labels_ , theme='fire')   
  #plt.legend()

def get_wordcloud(model, token_list, topic):
  """
  Get word cloud of each topic from fitted model
  :param model: Topic_Model object
  :param sentences: preprocessed sentences from docs
  """
  if model.method == 'LDA':
    return
  lbs = model.cluster_model.labels_
  tokens = ' '.join([' '.join(_) for _ in np.array(token_list)[lbs == topic   ]])
  wordcloud = WordCloud.WordCloud(width=800, height=560, background_color='white', collocations=False, min_font_size=10).generate(tokens)
  # plot the WordCloud image
  print('Word cloud for topic {}... '.format(topic))
  plt.figure(figsize=(8, 5.6), facecolor=None)
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout(pad=0)
  print('Getting wordcloud for topic {}. Done!'.format(topic))


