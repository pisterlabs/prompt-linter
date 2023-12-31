# -*- coding: utf-8 -*-
"""SO Expertise Model Evaluation - optimizer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k4hpuumxgJ5pERHWEt2bHzzqTQw7irDF
"""
from __future__ import absolute_import
import funcy as fp
import numpy as np
from scipy.sparse import issparse
from sklearn.metrics import precision_recall_fscore_support
from gensim.models.keyedvectors import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, strip_numeric, preprocess_string
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle
import pyLDAvis.gensim
from joblib import Parallel, delayed, cpu_count
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from sklearn.metrics.pairwise import cosine_similarity
from gensim.matutils import jaccard_distance
from joblib import Parallel, delayed, cpu_count
from gensim.sklearn_api import LdaTransformer
from gensim.models import CoherenceModel, LdaModel
from gensim.corpora.mmcorpus import MmCorpus
from gensim.test.utils import datapath
from gensim.corpora.dictionary import Dictionary
from skopt import dump
from skopt import gp_minimize
from skopt.utils import use_named_args
import skopt
import csv
import time
import statistics

"""# Load in human annotations"""

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

def load_annotations():
  SO_annotation = pd.read_csv('/home/norberteke/PycharmProjects/Thesis/data/SO_annotations_processed.csv', header = 0,
                        names = ["sample_ID","profile_url","unified_Id","internal_ID","Annotator_1","Annotator_2", "Processed_Annotator_1", "Processed_Annotator_2"])

  GH_annotation = pd.read_csv('/home/norberteke/PycharmProjects/Thesis/data/GH_annotations_processed.csv', header = 0, 
                        names = ["sample_ID","profile_url","unified_Id","internal_ID","Annotator_1","Annotator_2", "Processed_Annotator_1", "Processed_Annotator_2"])
  
  GH_IDs = GH_annotation["internal_ID"]
  SO_IDs = SO_annotation["internal_ID"]

  GH_annotation_intersect = {}
  GH_annotation_union = {}
  SO_annotation_intersect = {}
  SO_annotation_union = {}

  for index, row in SO_annotation.iterrows():
    a1 = row['Processed_Annotator_1'].split(";")
    a2 = row['Processed_Annotator_2'].split(";")

    if '' in a1:
      a1.remove('')
    if '' in a2:
      a2.remove('')

    SO_annotation_intersect[row['internal_ID']] = intersect(a1, a2)
    SO_annotation_union[row['internal_ID']] = union(a1, a2)

  for index, row in GH_annotation.iterrows():
    a1 = row['Processed_Annotator_1'].split(";")
    a2 = row['Processed_Annotator_2'].split(";")

    if '' in a1:
      a1.remove('')
    if '' in a2:
      a2.remove('')

    GH_annotation_intersect[row['internal_ID']] = intersect(a1, a2)
    GH_annotation_union[row['internal_ID']] = union(a1, a2)

  return GH_IDs, SO_IDs, GH_annotation_intersect, GH_annotation_union, SO_annotation_intersect, SO_annotation_union

"""# Create topic info data frame by calling get_topic_info(lda_model, corpus, dictionary)"""

def get_topic_info(topic_model, corpus, dictionary, doc_topic_dist=None):
  opts = fp.merge(pyLDAvis_prepare(topic_model, corpus, dictionary, doc_topic_dist))
  return my_prepare(**opts)
  
def _chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _job_chunks(l, n_jobs):
    n_chunks = n_jobs
    if n_jobs < 0:
        # so, have n chunks if we are using all n cores/cpus
        n_chunks = cpu_count() + 1 - n_jobs

    return _chunks(l, n_chunks)


def _find_relevance(log_ttd, log_lift, R, lambda_):
    relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
    return relevance.T.apply(lambda s: s.sort_values(ascending=False).index).head(R)


def _find_relevance_chunks(log_ttd, log_lift, R, lambda_seq):
    return pd.concat([_find_relevance(log_ttd, log_lift, R, l) for l in lambda_seq])



def _df_with_names(data, index_name, columns_name):
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name):
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)


def _topic_info(topic_term_dists, topic_proportion, term_frequency, term_topic_freq,
                vocab, lambda_step, R, n_jobs):
    # marginal distribution over terms (width of blue bars)
    term_proportion = term_frequency / term_frequency.sum()

    # compute the distinctiveness and saliency of the terms:
    # this determines the R terms that are displayed when no topic is selected
    topic_given_term = topic_term_dists / topic_term_dists.sum()
    kernel = (topic_given_term * np.log((topic_given_term.T / topic_proportion).T))
    distinctiveness = kernel.sum()
    saliency = term_proportion * distinctiveness
    # Order the terms for the "default" view by decreasing saliency:
    default_term_info = pd.DataFrame({
        'saliency': saliency,
        'Term': vocab,
        'Freq': term_frequency,
        'Total': term_frequency,
        'Category': 'Default'})
    default_term_info = default_term_info.sort_values(
        by='saliency', ascending=False).head(R).drop('saliency', 1)
    # Rounding Freq and Total to integer values to match LDAvis code:
    default_term_info['Freq'] = np.floor(default_term_info['Freq'])
    default_term_info['Total'] = np.floor(default_term_info['Total'])
    ranks = np.arange(R, 0, -1)
    default_term_info['logprob'] = default_term_info['loglift'] = ranks

    # compute relevance and top terms for each topic
    log_lift = np.log(topic_term_dists / term_proportion)
    log_ttd = np.log(topic_term_dists)
    lambda_seq = np.arange(0, 1 + lambda_step, lambda_step)

    def topic_top_term_df(tup):
        new_topic_id, (original_topic_id, topic_terms) = tup
        term_ix = topic_terms.unique()
        return pd.DataFrame({'Term': vocab[term_ix],
                             'Freq': term_topic_freq.loc[original_topic_id, term_ix],
                             'Total': term_frequency[term_ix],
                             'logprob': log_ttd.loc[original_topic_id, term_ix].round(4),
                             'loglift': log_lift.loc[original_topic_id, term_ix].round(4),
                             'Category': 'Topic%d' % new_topic_id})

    top_terms = pd.concat(Parallel(n_jobs=n_jobs)
                          (delayed(_find_relevance_chunks)(log_ttd, log_lift, R, ls)
                          for ls in _job_chunks(lambda_seq, n_jobs)))
    topic_dfs = map(topic_top_term_df, enumerate(top_terms.T.iterrows(), 1))
    return pd.concat([default_term_info] + list(topic_dfs), sort=True)

def pyLDAvis_prepare(topic_model, corpus, dictionary, doc_topic_dists=None):
    import gensim
    if not gensim.matutils.ismatrix(corpus):
        corpus_csc = gensim.matutils.corpus2csc(corpus, num_terms=len(dictionary))
    else:
        corpus_csc = corpus
        # Need corpus to be a streaming gensim list corpus for len and inference functions below:
        corpus = gensim.matutils.Sparse2Corpus(corpus_csc)

    vocab = list(dictionary.token2id.keys())
    # TODO: add the hyperparam to smooth it out? no beta in online LDA impl.. hmm..
    # for now, I'll just make sure we don't ever get zeros...
    beta = 0.01
    fnames_argsort = np.asarray(list(dictionary.token2id.values()), dtype=np.int_)
    term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
    term_freqs[term_freqs == 0] = beta
    doc_lengths = corpus_csc.sum(axis=0).A.ravel()

    if hasattr(topic_model, 'lda_alpha'):
        num_topics = len(topic_model.lda_alpha)
    else:
        num_topics = topic_model.num_topics

    if doc_topic_dists is None:
        # If its an HDP model.
        if hasattr(topic_model, 'lda_beta'):
            gamma = topic_model.inference(corpus)
        else:
            gamma, _ = topic_model.inference(corpus)
        doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]
    else:
        if isinstance(doc_topic_dists, list):
            #doc_topic_dists = np.matrix(gensim.matutils.corpus2dense(doc_topic_dists, num_topics).T)
            doc_topic_dists = gensim.matutils.corpus2dense(doc_topic_dists, num_topics).T
        elif issparse(doc_topic_dists):
            doc_topic_dists = doc_topic_dists.T.todense()
        doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)

    # get the topic-term distribution straight from gensim without
    # iterating over tuples
    if hasattr(topic_model, 'lda_beta'):
        topic = topic_model.lda_beta
    else:
        topic = topic_model.state.get_lambda()
    topic = topic / topic.sum(axis=1)[:, None]
    topic_term_dists = topic[:, fnames_argsort]

    topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
    topic_proportion = (topic_freq / topic_freq.sum())

    term_topic_freq = (topic_term_dists.T * topic_freq).T

    return {'topic_term_dists': topic_term_dists, 'doc_topic_dists': doc_topic_dists, 
            'doc_lengths': doc_lengths, 'vocab': vocab, 'term_frequency': term_freqs}

def my_prepare(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency,
            R=30, lambda_step=0.01, n_jobs=-1, sort_topics=True):

    topic_term_dists = _df_with_names(topic_term_dists, 'topic', 'term')
    doc_topic_dists = _df_with_names(doc_topic_dists, 'doc', 'topic')
    term_frequency = _series_with_name(term_frequency, 'term_frequency')
    doc_lengths = _series_with_name(doc_lengths, 'doc_length')
    vocab = _series_with_name(vocab, 'vocab')

    topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
    if (sort_topics):
        topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
    else:
        topic_proportion = (topic_freq / topic_freq.sum())

    topic_order = topic_proportion.index
    # reorder all data based on new ordering of topics
    topic_freq = topic_freq[topic_order]
    topic_term_dists = topic_term_dists.iloc[topic_order]
    doc_topic_dists = doc_topic_dists[topic_order]

    # token counts for each term-topic combination (widths of red bars)
    term_topic_freq = (topic_term_dists.T * topic_freq).T
    term_frequency = np.sum(term_topic_freq, axis=0)

    topic_info = _topic_info(topic_term_dists, topic_proportion,
                             term_frequency, term_topic_freq, vocab, lambda_step, R, n_jobs)
    return topic_info

"""# Define techniques

## Word2Vec user and topic Embeddings using Max and Avg pooling
"""

def run_Word2Vec_emb(lda, threshold, maxPool):
  user_vectors = get_user_emb()
  avg_topic_vectors, max_topic_vectors = get_topic_emb(lda)
  
  if maxPool:
    topic_vectors = max_topic_vectors
  else:
    topic_vectors = avg_topic_vectors
  
  cos_sim = cosine_similarity(user_vectors, topic_vectors)
  user_topic_mapping = create_user_topic_mapping(cos_sim, threshold)
  return user_topic_mapping

"""Create user and topic embeddings using SO_Word2Vec_200"""

word_vectors = KeyedVectors.load("/home/norberteke/PycharmProjects/Thesis/data/SO_pre-trained_vectors.kv", mmap='r')

CUSTOM_FILTERS = [lambda x: strip_numeric, remove_stopwords]

def word2vec_embedding_lookup(words):
  vectors = []
  for w in words:
    try:
      vec = word_vectors[w]
      vectors.append(vec)
    except:
      try:
        w_transformed = w.replace(".", "").replace("=", "").replace("-", "").replace("*", "").replace("'", "").replace("`", "").replace("|", "").replace('\\', "").replace("/", "").replace("$", "").replace("^", "").replace("#", "").replace("&", "").replace("@", "")
        vec = word_vectors[w_transformed]
        vectors.append(vec)
      except:
        try:
          w_stripped = preprocess_string(w_transformed, CUSTOM_FILTERS)
          vec = word_vectors[w_stripped]
          vectors.append(vec)
        except:
          continue
  return np.array(vectors)

"""Get topic and user emb from pre-trained Word2Vec model"""

def get_user_emb():
  user_embeddings = []
  for i in range(0,len(texts)):
    word_vectors = word2vec_embedding_lookup(list(set(texts[i]).intersection(terms)))
    try:
      feature_vector = np.max(word_vectors, axis=0)
      user_embeddings.append(feature_vector)
    except ValueError:
      user_embeddings.append(np.zeros((200,)))  # 200 x 1 vector of 0's, since the word2vec model is 200 dimensional
  return np.array(user_embeddings)

def get_topic_emb(lda):
  avg_topic_emb = []
  max_topic_emb = []
  number_of_topicWords = 20
  topic_num = lda.num_topics 

  for topic in range(0, topic_num):  # for each topic inside a specific model
    results = lda.show_topic(topic, topn=number_of_topicWords)

    topic_words = []
    for i in range(0,number_of_topicWords): # for each topic word inside a topic
      topic_words.append(results[i][0])

    word_vectors = word2vec_embedding_lookup(topic_words)
    avg_feature_vector = np.average(word_vectors, axis=0)
    max_feature_vector = np.max(word_vectors, axis=0)

    avg_topic_emb.append(avg_feature_vector)
    max_topic_emb.append(max_feature_vector)
  return np.asarray(avg_topic_emb), np.asarray(max_topic_emb)

"""## LDA_topicEmbedding using Max-pooling and Avg-pooling"""

def embedding_lookup(term_embeddings, word):
  return np.array(term_embeddings[word])

def get_LDA_user_emb(topic_num, term_emb):
  user_embeddings = []
  for i in range(0,len(texts)):
    word_vectors = embedding_lookup(term_emb, list(set(texts[i]).intersection(terms)))
    try:
      feature_vector = np.max(word_vectors, axis=1)
      user_embeddings.append(feature_vector)
    except ValueError:
      user_embeddings.append(np.zeros((topic_num,)))
  return np.array(user_embeddings)

def get_LDA_topic_emb(lda_model, term_emb):
  avg_topic_emb = []
  max_topic_emb = []
  number_of_topicWords = 20

  for topic in range(0, lda_model.num_topics):  # for each topic inside a specific model
    results = lda_model.show_topic(topic, topn=number_of_topicWords)

    topic_words = []
    for i in range(0,number_of_topicWords):
      topic_words.append(results[i][0])

    word_vectors = embedding_lookup(term_emb, topic_words)
    avg_feature_vector = np.average(word_vectors, axis=1)
    max_feature_vector = np.max(word_vectors, axis=1)

    avg_topic_emb.append(avg_feature_vector)
    max_topic_emb.append(max_feature_vector)

  avg_topic_vectors = np.array(avg_topic_emb)
  max_topic_vectors = np.array(max_topic_emb)
  return avg_topic_vectors, max_topic_vectors

def jaccard_sim(A, B):
  return 1 - jaccard_distance(set(A), set(B))     # jaccard sim = 1 - jaccard distance

def run_LDA_emb(lda_model, term_emb, threshold, maxPool):
  user_vectors = get_LDA_user_emb(lda_model.num_topics, term_emb)
  avg_topic_vectors, max_topic_vectors = get_LDA_topic_emb(lda_model, term_emb)

  if maxPool:
    topic_vectors = max_topic_vectors
  else:
    topic_vectors = avg_topic_vectors

  cos_sim = cosine_similarity(user_vectors, topic_vectors)
  user_topic_mapping = create_user_topic_mapping(cos_sim, threshold)
  return user_topic_mapping

def create_user_topic_mapping(cos_sims, threshold):
  user_topic_mapping = {}
  for user_i in range(0, 83550):   # counting for users 0 --> 83549
    user_topic_mapping[user_i] = ['Topic' + str(index+1) for index, value in enumerate(cos_sims[user_i]) if value > threshold]
  return user_topic_mapping

def get_user_expertise(topicInfo, user_i, topic_terms = 20):
  optimal_lambda_val = 0.6
  expertise = get_relevant_terms(topicInfo, user_i, optimal_lambda_val, topic_terms)
  return expertise

def get_relevant_terms(topic_info, topics, _lambda, term_num):
  """Retuns a list of top-n keywords (where n = term_num) that have the highest relevance score for the topics the the user is in."""

  tdf = pd.DataFrame(topic_info[topic_info.Category.isin(topics)])
  stdf = tdf.assign(relevance=_lambda * tdf['logprob'] + (1 - _lambda) * tdf['loglift'])
  new_df = stdf.sort_values('relevance', ascending=False)

  term_list = new_df['Term'].tolist()
  if '-PRON-' in term_list:
    term_list.remove('-PRON-')
  if ' ' in term_list:
    term_list.remove(' ')
  return term_list

"""## LDA Topic Distribution based Expertise"""

def LDA_topicDistr(lda_model, topicInfo, user_i, threshold, topic_terms):
  user_topic_membership = create_user_topic_thresholding(lda_model, user_i, threshold)
  user_expertise = get_expertise_for_user_i(topicInfo, user_topic_membership, topic_terms)
  return user_expertise

def create_user_topic_thresholding(lda_model, user_i, threshold):
  user_i_topic_distr = lda_model.get_document_topics(bow = corpus[user_i], minimum_probability = threshold)

  topic_memberships = []
  for topic in user_i_topic_distr:
    topic_memberships.append('Topic' + str(topic[0]+1))   # topics are 0 to k-1, so offset by 1, since pyLDAvis indexes from 1 to k

  return topic_memberships

def get_expertise_for_user_i(topicInfo, user_topic_membership, topic_terms):
  optimal_lambda_val = 0.6
  expertise = get_relevant_terms(topicInfo, user_topic_membership, optimal_lambda_val, topic_terms)
  return expertise

"""# Create Evaluation functions"""

def getExistingWordsFromModel(words):
  """ Checks if a list of words are in the dictionary of the word2vec model """
  CUSTOM_FILTERS = [lambda x: strip_numeric, remove_stopwords]
  res = []
  for w in words:
    try:
      vec = word_vectors[w]
      res.append(w)
    except:
      try:
        w_transformed = w.replace(".", "").replace("=", "").replace("-", "").replace("*", "").replace("'", "").replace("`", "").replace("|", "").replace('\\', "").replace("/", "").replace("$", "").replace("^", "").replace("&", "").replace("@", "").replace("%", "")
        vec = word_vectors[w_transformed]
        res.append(w_transformed)
      except:
         try:
          w_stripped = preprocess_string(w_transformed, CUSTOM_FILTERS)
          vec = word_vectors[w_stripped]
          res.append(w_stripped)
         except:
           continue
  return res

def evaluate_LDA_topicDistr(lda_model, topicInfo, threshold_t):
  BLEU_scores = []
  jacc_similarity = []
  cos_similarity = []

  recall = []
  precision = []
  fscore = []

  for user_i in SO_IDs:
    # or if you want intersect, use SO_annotation_intersect
    annotation = SO_annotation_union[user_i]

    model_hypothesis = LDA_topicDistr(lda_model, topicInfo, user_i, threshold_t, topic_terms = len(annotation))

    # 1-gram individual BLEU with smoothing function
    smooth = SmoothingFunction()
    BLEU_score = sentence_bleu(references = [annotation], hypothesis = model_hypothesis, weights = (1, 0, 0, 0), smoothing_function = smooth.method1)
    BLEU_scores.append(BLEU_score)

    # calculate precision, recall, fscore
    if len(annotation) == len(model_hypothesis):
      micro_res = precision_recall_fscore_support(y_true = np.array(annotation), y_pred = np.array(model_hypothesis), average='micro')
      precision.append(micro_res[0])
      recall.append(micro_res[1])
      fscore.append(micro_res[2])

    # calculate Jaccard similarity between annotation and model hypothesis 
    jaccard_simm = jaccard_sim(annotation, model_hypothesis)
    jacc_similarity.append(jaccard_simm)

    # Compute cosine similarity between annotation and model hypothesis 
    a = getExistingWordsFromModel(annotation)
    b = getExistingWordsFromModel(model_hypothesis)

    if len(a) > 0 and len(b) > 0:
      cos_sim = word_vectors.n_similarity(a, b)
      cos_similarity.append(cos_sim)

  return BLEU_scores, jacc_similarity, cos_similarity, precision, recall, fscore

def evaluate_LDA_topicEmb(lda_model, topicInfo, term_emb, threshold, maxPool):
  user_topic_mapping = run_LDA_emb(lda_model, term_emb, threshold, maxPool)

  BLEU_scores = []
  jacc_similarity = []
  cos_similarity = []

  recall = []
  precision = []
  fscore = []

  for user_i in SO_IDs:
    # or if you want intersect, use SO_annotation_intersect
    annotation = SO_annotation_union[user_i]

    model_hypothesis = get_user_expertise(topicInfo, user_topic_mapping[user_i], topic_terms = len(annotation))

    # 1-gram individual BLEU with smoothing function
    smooth = SmoothingFunction()
    BLEU_score = sentence_bleu(references = [annotation], hypothesis = model_hypothesis, 
                               weights = (1, 0, 0, 0), smoothing_function = smooth.method1)
    BLEU_scores.append(BLEU_score)

    # calculate precision, recall, fscore
    if len(annotation) == len(model_hypothesis):
      micro_res = precision_recall_fscore_support(y_true = np.array(annotation), y_pred = np.array(model_hypothesis), average='micro')
      precision.append(micro_res[0])
      recall.append(micro_res[1])
      fscore.append(micro_res[2])

    # calculate Jaccard similarity between annotation and model hypothesis 
    jaccard_simm = jaccard_sim(annotation, model_hypothesis)
    jacc_similarity.append(jaccard_simm)

    # Compute cosine similarity between annotation and model hypothesis 
    a = getExistingWordsFromModel(annotation)
    b = getExistingWordsFromModel(model_hypothesis)
    
    if len(a) > 0 and len(b) > 0:
      cos_sim = word_vectors.n_similarity(a, b)
      cos_similarity.append(cos_sim)

  return BLEU_scores, jacc_similarity, cos_similarity, precision, recall, fscore

def evaluate_Word2Vec_Emb(lda, topicInfo, threshold, maxPool):
  user_topic_mapping = run_Word2Vec_emb(lda, threshold, maxPool)

  BLEU_scores = []
  jacc_similarity = []
  cos_similarity = []

  recall = []
  precision = []
  fscore = []

  for user_i in SO_IDs:
    # or if you want intersect, use SO_annotation_intersect
    annotation = SO_annotation_union[user_i]
    model_hypothesis = get_user_expertise(topicInfo, user_topic_mapping[user_i], topic_terms = len(annotation))

    # 1-gram individual BLEU with smoothing function
    smooth = SmoothingFunction()
    BLEU_score = sentence_bleu(references = [annotation], hypothesis = model_hypothesis, 
                               weights = (1, 0, 0, 0), smoothing_function = smooth.method1)
    BLEU_scores.append(BLEU_score)

    # calculate precision, recall, fscore
    if len(annotation) == len(model_hypothesis):
      micro_res = precision_recall_fscore_support(y_true = np.array(annotation), y_pred = np.array(model_hypothesis), average='micro')
      precision.append(micro_res[0])
      recall.append(micro_res[1])
      fscore.append(micro_res[2])

    # calculate Jaccard similarity between annotation and model hypothesis 
    jaccard_simm = jaccard_sim(annotation, model_hypothesis)
    jacc_similarity.append(jaccard_simm)

    # Compute cosine similarity between annotation and model hypothesis 
    a = getExistingWordsFromModel(annotation)
    b = getExistingWordsFromModel(model_hypothesis)
    
    if len(a) > 0 and len(b) > 0:
      cos_sim = word_vectors.n_similarity(a, b)
      cos_similarity.append(cos_sim)

  return BLEU_scores, jacc_similarity, cos_similarity, precision, recall, fscore

"""# Main

## Run Experiments for LDA_topic distribution based expertise on SO data, using SO_past model
"""

def main_topicDistr(lda_model, topic_info):
  threshold = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
             0.22, 0.24, 0.26, 0.28, 0.30,
             0.32, 0.34, 0.36, 0.38, 0.40,
             0.42, 0.44, 0.46, 0.48, 0.50]
  mean_jaccard = []
  mean_bleu = []
  mean_cos = []
  mean_fscore = []

  for t in threshold:
    BLEU_scores, jacc_sim, cos_sim, precision, recall, fscore = evaluate_LDA_topicDistr(lda_model, topic_info, t)
    bleu_np = np.asarray(BLEU_scores)
    jacc_np = np.asarray(jacc_sim)
    cos_np = np.asarray(cos_sim)
    fscore_np = np.asarray(fscore)

    mean_jaccard.append( np.mean(jacc_np) )
    mean_bleu.append( np.mean(bleu_np) )
    mean_cos.append( np.mean(cos_np) )
    mean_fscore.append( np.mean(fscore_np) )
  return np.max( np.asarray(mean_bleu) ), np.max( np.asarray(mean_jaccard) ), np.max( np.asarray(mean_cos) ), np.max( np.asarray(mean_fscore) )

"""## Run Experiments for LDA_topicEmbedding using Avg-pooling on SO data, using SO_full model"""

def main_LDA_avgEmb(lda_model, topic_info, term_emb):
  threshold_values = [0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 
                    0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 
                    0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]
  mean_jaccard = []
  mean_bleu = []
  mean_cos = []
  mean_fscore = []
  
  for threshold in threshold_values:
    BLEU_scores, jacc_sim, cos_sim, precision, recall, fscore = evaluate_LDA_topicEmb(lda_model, topic_info, term_emb, threshold, maxPool=False)
    bleu_np = np.asarray(BLEU_scores)
    jacc_np = np.asarray(jacc_sim)
    cos_np = np.asarray(cos_sim)
    fscore_np = np.asarray(fscore)

    mean_jaccard.append( np.mean(jacc_np) )
    mean_bleu.append( np.mean(bleu_np) )
    mean_cos.append( np.mean(cos_np) )
    mean_fscore.append( np.mean(fscore_np) )
  return np.max( np.asarray(mean_bleu) ), np.max( np.asarray(mean_jaccard) ), np.max( np.asarray(mean_cos) ), np.max( np.asarray(mean_fscore) )

"""## Run Experiments for LDA_topicEmbedding using Max-pooling on SO data, using SO_past model"""

def main_LDA_maxEmb(lda_model, topic_info, term_emb):
  threshold_values = [0.20, 0.25, 0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48, 
                      0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66,
                      0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90] 
  mean_jaccard = []
  mean_bleu = []
  mean_cos = []
  mean_fscore = []

  for threshold in threshold_values:
    BLEU_scores, jacc_sim, cos_sim, precision, recall, fscore = evaluate_LDA_topicEmb(lda_model, topic_info, term_emb, threshold, maxPool=True)
    bleu_np = np.asarray(BLEU_scores)
    jacc_np = np.asarray(jacc_sim)
    cos_np = np.asarray(cos_sim)
    fscore_np = np.asarray(fscore)

    mean_jaccard.append( np.mean(jacc_np) )
    mean_bleu.append( np.mean(bleu_np) )
    mean_cos.append( np.mean(cos_np) )
    mean_fscore.append( np.mean(fscore_np) )
  return np.max( np.asarray(mean_bleu) ), np.max( np.asarray(mean_jaccard) ), np.max( np.asarray(mean_cos) ), np.max( np.asarray(mean_fscore) )

"""##Run Experiments for Word2vec user and topic Embedding using Avg-pooling on SO data"""

def main_Word2Vec_AvgEmb(lda_model, topic_info):
  threshold_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
  mean_jaccard = []
  mean_bleu = []
  mean_cos = []
  mean_fscore = []

  for threshold in threshold_values:
    BLEU_scores, jacc_sim, cos_sim, precision, recall, fscore = evaluate_Word2Vec_Emb(lda=lda_model, topicInfo=topic_info, threshold=threshold, maxPool=False)
    bleu_np = np.asarray(BLEU_scores)
    jacc_np = np.asarray(jacc_sim)
    cos_np = np.asarray(cos_sim)
    fscore_np = np.asarray(fscore)

    mean_jaccard.append( np.mean(jacc_np) )
    mean_bleu.append( np.mean(bleu_np) )
    mean_cos.append( np.mean(cos_np) )
    mean_fscore.append( np.mean(fscore_np) )
  return np.max( np.asarray(mean_bleu) ), np.max( np.asarray(mean_jaccard) ), np.max( np.asarray(mean_cos) ), np.max( np.asarray(mean_fscore) )

"""## Run Experiments for Word2vec user and topic Embedding using Max-pooling on SO data"""

def main_Word2Vec_MaxEmb(lda_model, topic_info):
  threshold_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                    0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
                    0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30,
                    0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]
  mean_jaccard = []
  mean_bleu = []
  mean_cos = []
  mean_fscore = []

  for threshold in threshold_values:
    BLEU_scores, jacc_sim, cos_sim, precision, recall, fscore = evaluate_Word2Vec_Emb(lda_model, topicInfo=topic_info, threshold = threshold, maxPool=True)
    bleu_np = np.asarray(BLEU_scores)
    jacc_np = np.asarray(jacc_sim)
    cos_np = np.asarray(cos_sim)
    fscore_np = np.asarray(fscore)

    mean_jaccard.append( np.mean(jacc_np) )
    mean_bleu.append( np.mean(bleu_np) )
    mean_cos.append( np.mean(cos_np) )
    mean_fscore.append( np.mean(fscore_np) )
  return np.max( np.asarray(mean_bleu) ), np.max( np.asarray(mean_jaccard) ), np.max( np.asarray(mean_cos) ), np.max( np.asarray(mean_fscore) )

GH_IDs, SO_IDs, GH_annotation_intersect, GH_annotation_union, SO_annotation_intersect, SO_annotation_union = load_annotations()
path = '/home/norberteke/PycharmProjects/Thesis/data/'

dictionary = Dictionary.load(path + 'SO_full_processed_Dictionary.dict')
corpus = MmCorpus(datapath(path + 'corpus_processed_SO_full.mm'))

texts = []
with open(path + 'new_SO_full_processed_corpus.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)


terms = []
for (key, value) in dictionary.iteritems():
  terms.append(value)

def write_results_to_file(path, lda_model, max_bleu, max_jaccard, max_cos, max_fscore):
  with open(path, 'a') as f:
    writer = csv.writer(f, delimiter = ',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
    writer.writerow([str(lda_model.num_topics), str(lda_model.eta), str(max_bleu), str(max_jaccard), str(max_cos), str(max_fscore)])


def evaluateModel(lda_model, topic_info, term_emb, mode):
  if mode == 1:
    max_bleu, max_jaccard, max_cos, max_fscore = main_topicDistr(lda_model, topic_info)
  elif mode == 2:
    max_bleu, max_jaccard, max_cos, max_fscore = main_LDA_avgEmb(lda_model, topic_info, term_emb)
  elif mode == 3:
    max_bleu, max_jaccard, max_cos, max_fscore = main_LDA_maxEmb(lda_model, topic_info, term_emb)
  elif mode == 4:
    max_bleu, max_jaccard, max_cos, max_fscore = main_Word2Vec_AvgEmb(lda_model, topic_info)
  elif mode == 5:
    max_bleu, max_jaccard, max_cos, max_fscore = main_Word2Vec_MaxEmb(lda_model, topic_info)
  write_results_to_file("/home/norberteke/PycharmProjects/Thesis/data/SO_simulation_results_4.csv", lda_model, max_bleu, max_jaccard, max_cos, max_fscore)


k = []
for i in range(10,51):
	k.append(i)
	
beta = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

for topic_num in k:
  print("----- Progress: k= ", topic_num ,"----")
  for b in beta:
    model = LdaTransformer(id2word=dictionary, num_topics = topic_num, alpha='auto', eta = b, iterations=100, random_state=2019)
    lda = model.fit(corpus)
    term_topic_matrix = lda.gensim_model.get_topics()
    term_emb = pd.DataFrame(term_topic_matrix, columns=terms)
    topic_info = get_topic_info(lda.gensim_model, corpus, dictionary)
    evaluateModel(lda.gensim_model, topic_info, term_emb, mode = 4)