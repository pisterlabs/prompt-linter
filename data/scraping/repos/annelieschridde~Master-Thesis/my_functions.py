# The following packages, modules and functions are used for processing the data

#files
import csv
import pickle

# OS for path
import os
import requests
import gzip
import io

# General packages
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import Normalizer

# Models
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
#from sklearn.decomposition import MiniBatchNMF
from sklearn.manifold import TSNE

# Metrics
from scipy.spatial.distance import euclidean, cosine
from gensim.models import CoherenceModel # Compute Coherence Score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

# Parallelization and Time Complexity
from joblib import Parallel, delayed
from tqdm import tqdm
from time import time
from tqdm.notebook import trange, tqdm
from tqdm import tqdm
tqdm.pandas()

# Plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as xp
import matplotlib.pyplot as plt
import seaborn as sns

# spaCy 
import spacy
from spacy import displacy # visualization
nlp = spacy.load("en_core_web_sm")
#import en_core_web_sm

# NLTK
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
tokenizer = RegexpTokenizer(r'\b\w{3,}\b')
from nltk import FreqDist

# Gensim: Preprocessing and Modelling
from gensim.parsing.preprocessing import STOPWORDS, strip_tags, strip_numeric, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short, stem_text
from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.corpora.dictionary import Dictionary


# Other Preprocessing Modules
from autocorrect import Speller
import contractions
import re
import demoji
from datetime import datetime
import math


#_______________________________________________________________________________________________________________________________________________________
#### DATA CLEANING AND PREPROCESSING #####

def autocorrection(review):
    '''function to autocorrect words in a review
        Review: String, for example from a row in a dataframe which contains text
    '''
    corrected_words = []

    spell_corrector = Speller(lang="en")
    for word in word_tokenize(review):
        correct_word = spell_corrector(word)
        corrected_words.append(correct_word)
    
    correct_spelling = " ".join(corrected_words)
    return correct_spelling

def expand_contractions(review):
    '''
    expanding contractions in a string
    example: doesn't --> does not
    '''
    expanded_words = []
    for word in review.split():
        expanded_words.append(contractions.fix(word))
    expanded_review = " ".join(expanded_words)
    return expanded_review

def rm_single_char(review):
    regex_pattern = r'\s+[a-zA-Z]\s+'
    removed_char = re.sub(pattern=regex_pattern, repl = " ", string=review)
    return removed_char

def clean_complete(review):
    """
    review: pandas series
    prepares reviews complete cleaning for further lemmatization and bag-of-words
    """
    pat = r"(\\n)|(@\w*)|((www\.[^\s]+)|(https?://[^\s]+))"
    review =  review.str.replace(pat, '')
    
    #replace emoticons with words

    review =  review.str.replace(r':-\)', ' smile')
    review =  review.str.replace(r':-\(', ' sad')
    review =  review.str.replace(r':-\/', ' confused')
    review =  review.str.replace(r':-P', ' playfullness')

    #delete \xa
    review =  review.str.replace('\xa0', '')
    review =  review.str.replace('&amp', '')
    review =  review.str.replace('\n', '')
    review =  review.str.replace('"', '')
    review =  review.str.replace("'", '')
    review =  review.str.replace(r'$|\u200d|—', '')
    
    #to lower case
    review =  review.str.lower()

    review =  review.str.replace(r'book', '') # do not want product category in topics
    review =  review.str.replace(r'books', '') # do not want product category in topics
    review =  review.str.replace(r'star', '') # do not want star in topics
    review =  review.str.replace(r'stars', '') # do not want stars in topics

    #convert hashtags to the normal text
    review =  review.str.replace(r'#([^\s]+)', r'\1')

    #delete numbers
    review = [strip_numeric(c) for c in   review]

    #replacing emojies with descriptions '❤️-> red heart'
    review = [demoji.replace_with_desc(c, ' ') for c in   review]

    #replacing emojies with ''
    review = [demoji.replace(c, ' ') for c in review]

    #delete punctuation
    review = [strip_punctuation(c) for c in   review]

    #remove stop words
    review = [remove_stopwords(c) for c in    review]

    #remove short words
    review = [strip_short(c) for c in review]

    #remove mult whitespaces
    review = [strip_multiple_whitespaces(c) for c in  review]
    return  review

def lemmatize(review):
    '''
    review: pandas series
    should be applied on the cleaned review to transform words to their initial base form.
    For example: suggests -> suggest, deliveries -> delivery
    '''
    nlp = spacy.load("en_core_web_sm")
    review = [nlp(c) for c in review]
    review = [" ".join([token.lemma_ for token in t]) for t in review]
    return review

def remove_frequent_words(review, stop_words:list):
    words = word_tokenize(review)
    review = [word for word in words if word not in stop_words]
    return " ".join(review)
 

#_______________________________________________________________________________________________________________________________________________________
##### EVALUATION FUNCTION OF ALL METRICS AND FOR ALL TOPIC MODELS  ######
#https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py

# Getting Topics
def get_topics(model, feature_names, no_top_words):
    """ model: Clustering Model
        feature_names: all tokenized words from preprocessed data
        no_top_words: Count of words that should be used for topic, sorted descending
    """
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
    return topics

#Print topics as df
def display_topics(model, feature_names, no_top_words):
    """a function which takes in our model object model, the order of the words in our matrix tf_feature_names and the number of words we would like to show. Use this function, which returns a dataframe, to show you the topics we created. Remember that each topic is a list of words/tokens and weights"""
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
    
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di


def evaluation(model, bow_sparse,bow_sparse_array, feature_names, no_top_words, original_docs, dictionary):


    """ combines all possible metrics for a topic model, output are 8 different measures
        model: Topic Model
        bow_sparse: Bag-of-Words Sparse-Matrix (TF or TFIDF)
        bow_sparse_array: Bag-of-Words Sparse-Matrix (TF or TFIDF) as an array
        feature_names: all tokenized words from preprocessed data
        no_top_words: Count of words that should be used for topic, sorted descending
    """
    #prepare input for coherence model parameter "texts"
    texts =[]
    for doc in original_docs:
        words = tokenizer.tokenize(doc)
        texts.append(words)
  

    #Assigning topics to documents
    if hasattr(model, 'components_')==True:
        doc_topic_mat = model.transform(bow_sparse)
        labels_ = []
        for line in doc_topic_mat:
            idx = np.where(line == line.max())
            labels_.append(idx[0][0])
    else: 
        labels_ = model.labels_

    #creating corpus for gensim models
    #corpus = []
    #for line in range(len(bow_sparse_array)):
     #   idx = np.where(bow_sparse_array[line] != 0)
      #  doc = []
       # for id in range(len(idx[0])):
        #    weight = bow_sparse_array[line][idx[0][id]]
         #   item = (idx[0][id], weight)
          #  doc.append(item)
        #corpus.append(doc)

    topics = []
    if hasattr(model, 'components_')==True:
        for topic_idx, topic in enumerate(model.components_):
            topics.append([feature_names[i]
                    for i in topic.argsort()[:-no_top_words - 1:-1]])
    else: 
        for topic_idx, topic in enumerate(model.cluster_centers_):
            topics.append([feature_names[i]
                    for i in topic.argsort()[:-no_top_words - 1:-1]])
    # Coherence Cv
    coherence_model_lda = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics, coherence='c_v', topn = 10)
    cv = coherence_model_lda.get_coherence()
    # Coherence UMass
    coherence_umass = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='u_mass', topn = 10)
    c_umass = coherence_umass.get_coherence()
    # Coherence NPMI
    coherence_npmi = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_npmi', topn = 10)
    c_npmi = coherence_npmi.get_coherence()
    # Coherence UCI
    coherence_uci = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_uci', topn = 10)
    c_uci = coherence_uci.get_coherence()

    # Dunn Index
    #di = dunn_fast(bow_sparse_array,labels_)

    # Silhouette Score
    sil_score = silhouette_score(bow_sparse_array, labels= labels_, metric='euclidean')

    # Calinski-Harabasz Score
    ch_index = calinski_harabasz_score(bow_sparse_array, labels=labels_)

    # Davies-Bouldin Score
    db_index = davies_bouldin_score(bow_sparse_array, labels=labels_ )


    print('\nCoherence Score CV: ', cv)
    print('\nCoherence Score UMass: ', c_umass)
    print('\nCoherence Score NPMI: ', c_npmi)
    print('\nCoherence Score UCI: ', c_uci)
    #print('\nDunn Index: ', di)
    print('\nSilhouette Score: ', sil_score)
    print('\nCalinski-Harabasz Score: ', ch_index)
    print('\nDavies-Bouldin Score: ', db_index)
    return cv, c_umass, c_npmi, c_uci,  sil_score, ch_index, db_index


#______________________________________________________________________________________________________________________________________________________________________
#### FINDING BEST K AND PLOT RESULTS ####

#getting topics
def get_topics(model, feature_names):
    topics = []
    if hasattr(model, 'components_')==True:
        for topic_idx, topic in enumerate(model.components_):
            topics.append(feature_names[topic.argsort()[-1]])
    else: 
        for topic_idx, topic in enumerate(model.cluster_centers_):
            topics.append([feature_names[i]
                for i in topic.argsort()[-1]])
    return topics

#Finding best k for each model

def find_best_k(n_components, estimator,sparse_matrix, sparse_toarray, feature_names, top_n_words, original_docs, dictionary):
    bestk_dict = {}
    for k in n_components:
        if estimator == "LSA":
            model_best_k = TruncatedSVD(n_components=k, algorithm = "arpack")
            model_best_k.fit(sparse_toarray.astype('float64'))
        if estimator == "LDA":
            model_best_k = LatentDirichletAllocation(n_components=k,random_state=0, learning_method="online", doc_topic_prior = (50 / 10), topic_word_prior = 200/len(dictionary))
            model_best_k.fit(sparse_toarray)
        if estimator == "NMF":
            model_best_k = MiniBatchNMF(n_components=k,init ='nndsvd')
            model_best_k.fit(sparse_toarray)
        cv_tf, c_umass_tf, c_npmi_tf, c_uci_tf, sil_score_tf, ch_index_tf, db_index_tf = evaluation(model_best_k, sparse_matrix, sparse_toarray, feature_names, top_n_words, original_docs,dictionary)
        bestk_dict["{} Topics".format(k)] = [cv_tf, c_umass_tf, c_npmi_tf, c_uci_tf,  sil_score_tf, ch_index_tf, db_index_tf]
    Topic_Score_Mat = pd.DataFrame(bestk_dict).T
    Topic_Score_Mat.columns = ["cv", "c_umass", "c_npmi", "c_uci",  "sil_score", "ch_index", "db_index"] # di excluded
    print("FINISHED! Evaluation of 20.000 Documents for topic modelling are evaluated!")
    return Topic_Score_Mat




def plot_results(Topic_Score_Mat, Title):
        fig = make_subplots(rows=8, cols=1, subplot_titles= ["Cv",  "UMass",  "NPMI", "UCI","Silhouette Score",  "Calinski-Harabasz Score", "Davies-Bouldin Score",])

        for i in range(1,8):
 
                fig.add_trace(
                go.Scatter(y=Topic_Score_Mat.iloc[:,i-1], x=Topic_Score_Mat.index, name=Topic_Score_Mat.columns[i-1]),
                row=i, col=1)

                if Topic_Score_Mat.columns[i-1] != "db_index":
                        fig.add_shape(type = "line",
                        x0=Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].max())[0])], y0 =Topic_Score_Mat.iloc[:,i-1].min(), 
                        x1=Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].max())[0])], y1 = Topic_Score_Mat.iloc[:,i-1].max(),
                        line=dict(
                        color="SeaGreen",
                        width=10), opacity = 0.5,
                        row = i, col= 1
                        )
                        fig.add_annotation(
                        x=Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].max())[0])],
                        y=Topic_Score_Mat.iloc[:,i-1].max(), 
                        xref="x",
                        yref="y",
                        text="Best K", 
                        showarrow=False,
                        yshift=20, row=i, col = 1)
                        fig.add_trace(go.Scatter(x=[Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].max())[0])]], y=[Topic_Score_Mat.iloc[:,i-1].max()], mode = 'markers',
                         marker_symbol = 'star', marker_color = "Red", 
                         marker_size = 15), row = i, col = 1)

                if Topic_Score_Mat.columns[i-1] == "db_index":
                        fig.add_shape(type = "line",
                        x0=Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].min())[0])], y0 =Topic_Score_Mat.iloc[:,i-1].min()-1, 
                        x1=Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].min())[0])], y1 = Topic_Score_Mat.iloc[:,i-1].min(),
                        line=dict(
                        color="SeaGreen",
                        width=10), opacity = 0.5, 
                        row = i, col= 1
                        ) 
                        fig.add_annotation(
                        x=Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].min())[0])],
                        y=Topic_Score_Mat.iloc[:,i-1].min(), 
                        xref="x",
                        yref="y",
                        text="Best K", 
                        showarrow=False,
                        yshift=20, row=i, col = 1)
                        fig.add_trace(go.Scatter(x=[Topic_Score_Mat.index[int(np.where(Topic_Score_Mat.iloc[:,i-1] == Topic_Score_Mat.iloc[:,i-1].min())[0])]], y=[Topic_Score_Mat.iloc[:,i-1].min()], mode = 'markers',
                         marker_symbol = 'star',marker_color = "Red",
                         marker_size = 15), row = i, col = 1)
                         



        fig.update_layout(height=1600, width=900, title_text=Title, showlegend = False
        )
        fig.update_yaxes(title='Metric Values')
        fig.update_xaxes()
        fig.update_traces()
        fig.update_scenes()
        fig.show()

#______________________________________________________________________________________________________________________________________________________________________

#### STABILITY ANALYSIS ####

def parallel_npmi(data, estimator, k, B, tfidf = True, no_top_words=20):
        #results = {}
    # get num _runs bootstrapped samples of unlabeled and labeled data
        if tfidf == True:
            bootstrapped_data = data.sample(n= 10000, random_state = B, replace = True)
            tfidf_vec = TfidfVectorizer( 
                #max_df=0.8,
                min_df=20,
                token_pattern='\w+|\$[\d\.]+|\S+'
                    )
            bow_sparse = tfidf_vec.fit_transform(bootstrapped_data)
            sparse_toarray = bow_sparse.toarray()
            feature_names = tfidf_vec.get_feature_names_out()


        else:
            bootstrapped_data = data.sample(n=10000, random_state = B, replace = True)
            # the vectorizer object will be used to transform text to vector form
            vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+', min_df=20) # max_df=0.8, min_df=20, 

            # apply transformation
            bow_sparse = vectorizer.fit_transform(bootstrapped_data)
            sparse_toarray = bow_sparse.toarray()
            # tf_feature_names tells us what word each column in the matric represents
            feature_names = vectorizer.get_feature_names_out()
 
        #prepare input for coherence model parameter "texts"
        texts =[]
        for doc in bootstrapped_data:
            words = tokenizer.tokenize(doc)
            texts.append(words)

        dictionary = Dictionary(texts)
        
        if estimator == "LDA":
            print('Running LDA...')
            model = LatentDirichletAllocation(n_components=k, random_state = 0, learning_method="online", doc_topic_prior=50/10, topic_word_prior=200/2046)
            model.fit(bow_sparse.toarray())

            topics = []
            for topic_idx, topic in enumerate(model.components_):
                topics.append([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])
                    
                # Coherence Cv
            coherence_model_lda = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics, coherence='c_v', topn = 10)
            cv = coherence_model_lda.get_coherence()
            # Coherence UMass
            coherence_umass = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='u_mass', topn = 10)
            c_umass = coherence_umass.get_coherence()
            # Coherence NPMI
            coherence_npmi = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_npmi', topn = 10)
            c_npmi = coherence_npmi.get_coherence()
            # Coherence UCI
            coherence_uci = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_uci', topn = 10)
            c_uci = coherence_uci.get_coherence()

        if estimator == "KM":
            print("Running K-Means...")
            model_best_k = MiniBatchKMeans(n_clusters=k, random_state=0)
            model_best_k.fit(sparse_toarray)
            topics = []
            for topic_idx, topic in enumerate(model_best_k.cluster_centers_):
                topics.append([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])
                    
                # Coherence Cv
            coherence_model_lda = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics, coherence='c_v', topn = 10)
            cv = coherence_model_lda.get_coherence()
            # Coherence UMass
            coherence_umass = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='u_mass', topn = 10)
            c_umass = coherence_umass.get_coherence()
            # Coherence NPMI
            coherence_npmi = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_npmi', topn = 10)
            c_npmi = coherence_npmi.get_coherence()
            # Coherence UCI
            coherence_uci = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_uci', topn = 10)
            c_uci = coherence_uci.get_coherence()

        if estimator == "LSA":
            print('Running LSA...')
            model_best_k = TruncatedSVD(n_components=k, algorithm = "arpack")
            model_best_k.fit(bow_sparse.astype('float64'))

            topics = []
            for topic_idx, topic in enumerate(model_best_k.components_):
                topics.append([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])

     
                # Coherence Cv
            coherence_model_lda = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics, coherence='c_v', topn = 10)
            cv = coherence_model_lda.get_coherence()
            # Coherence UMass
            coherence_umass = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='u_mass', topn = 10)
            c_umass = coherence_umass.get_coherence()
            # Coherence NPMI
            coherence_npmi = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_npmi', topn = 10)
            c_npmi = coherence_npmi.get_coherence()
            # Coherence UCI
            coherence_uci = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_uci', topn = 10)
            c_uci = coherence_uci.get_coherence()

            #results['sample {} and {} topic(s)'.format(j, k)] = [npmi]
                
        if estimator == 'NMF':
            print('Running NMF...')
            model_best_k = NMF(n_components=k,init ='nndsvd')
            model_best_k.fit(bow_sparse.toarray())
            topics = []
            for topic_idx, topic in enumerate(model_best_k.components_):
                topics.append([feature_names[i]
                for i in topic.argsort()[:-no_top_words - 1:-1]])
                    
                # Coherence Cv
            coherence_model_lda = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics, coherence='c_v', topn = 10)
            cv = coherence_model_lda.get_coherence()
            # Coherence UMass
            coherence_umass = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='u_mass', topn = 10)
            c_umass = coherence_umass.get_coherence()
            # Coherence NPMI
            coherence_npmi = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_npmi', topn = 10)
            c_npmi = coherence_npmi.get_coherence()
            # Coherence UCI
            coherence_uci = CoherenceModel(texts=texts, dictionary=dictionary, topics=topics,  coherence='c_uci', topn = 10)
            c_uci = coherence_uci.get_coherence()

        #result_mat = pd.DataFrame(results)
        print('Task finished!')
        yield ((k, cv, c_umass, c_npmi, c_uci)) # yield, because we are creating a generator object!

d_cv = 0
d_umass = 0
d_npmi = 0
d_uci = 0


#### BOXPLOTS!!! #####
def plot_box(Title_box, d_cv=d_cv, d_uci=d_uci, d_npmi=d_npmi, d_umass=d_umass):
    norm = Normalizer()
    norm_cv = pd.DataFrame(norm.fit_transform(pd.DataFrame(d_cv)),columns = pd.DataFrame(d_cv).columns)
    norm_umass = pd.DataFrame(norm.fit_transform(pd.DataFrame(d_umass)),columns = pd.DataFrame(d_umass).columns)
    norm_npmi = pd.DataFrame(norm.fit_transform(pd.DataFrame(d_npmi)),columns = pd.DataFrame(d_npmi).columns)
    norm_uci = pd.DataFrame(norm.fit_transform(pd.DataFrame(d_uci)),columns = pd.DataFrame(d_uci).columns)

    fig = make_subplots(rows=4, cols=1, subplot_titles= ["Stability Coherence Cv",  "Stability Coherence UMass", "Stability Coherence NPMI", "Stability Coherence UCI"])
    

    for col in norm_cv: 
        
        if col == norm_cv.columns[[int(np.where(norm_cv.median()
        == norm_cv.median().max())[0])]][0] and col != norm_cv.columns[[int(np.where(norm_cv.var()
        == norm_cv.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_cv[col], name = col, fillcolor="LightSalmon",  line=dict(color='darkred')),
            row=1, col=1)  
            fig.add_annotation(
                        x=norm_cv.columns[[int(np.where(norm_cv.median()== norm_cv.median().max())[0])]][0],
                        y=norm_cv[col].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=1, col = 1)


        if col != norm_cv.columns[[int(np.where(norm_cv.median()
        == norm_cv.median().max())[0])]][0] and col == norm_cv.columns[[int(np.where(norm_cv.var()== norm_cv.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_cv[col], name = col, fillcolor="Yellow",  line=dict(color='orange')),
            row=1, col=1)  
            fig.add_annotation(
                        x=norm_cv.columns[[int(np.where(norm_cv.var()== norm_cv.var().min())[0])]][0],
                        y=norm_cv[col].max(), 
                        xref="x",
                        yref="y",
                        text="Lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=1, col = 1)

        if col == norm_cv.columns[[int(np.where(norm_cv.median()
        == norm_cv.median().max())[0])]][0] and col == norm_cv.columns[[int(np.where(norm_cv.var()== norm_cv.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_cv[col], name = col, fillcolor="Lightgreen",  line=dict(color='Seagreen')),
            row=1, col=1)  
            fig.add_annotation(
                        x=norm_cv.columns[[int(np.where(norm_cv.var()== norm_cv.var().min())[0])]][0],
                        y=norm_cv[col].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median and lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=1, col = 1)


        if col != norm_cv.columns[[int(np.where(norm_cv.median()== norm_cv.median().max())[0])]][0] and col != norm_cv.columns[[int(np.where(norm_cv.var()== norm_cv.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_cv[col], name = col, fillcolor="slategray",  line=dict(color='black')),
            row=1, col=1)
        


    for col2 in norm_umass: 
        
        if col2 == norm_umass.columns[[int(np.where(norm_umass.median()
        == norm_umass.median().max())[0])]][0] and col2 != norm_umass.columns[[int(np.where(norm_umass.var()
        == norm_umass.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_umass[col2], name = col2, fillcolor="LightSalmon",  line=dict(color='darkred')),
            row=2, col=1)  
            fig.add_annotation(
                        x=norm_umass.columns[[int(np.where(norm_umass.median()== norm_umass.median().max())[0])]][0],
                        y=norm_umass[col2].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=2, col = 1)


        if col2 != norm_umass.columns[[int(np.where(norm_umass.median()
        == norm_umass.median().max())[0])]][0] and col2 == norm_umass.columns[[int(np.where(norm_umass.var()== norm_umass.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_umass[col2], name = col2, fillcolor="Yellow",  line=dict(color='orange')),
            row=2, col=1)  
            fig.add_annotation(
                        x=norm_umass.columns[[int(np.where(norm_umass.var()== norm_umass.var().min())[0])]][0],
                        y=norm_umass[col2].max(), 
                        xref="x",
                        yref="y",
                        text="Lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=2, col = 1)

        if col2 == norm_umass.columns[[int(np.where(norm_umass.median()
        == norm_umass.median().max())[0])]][0] and col2 == norm_umass.columns[[int(np.where(norm_umass.var()== norm_umass.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_umass[col2], name = col2, fillcolor="Lightgreen",  line=dict(color='Seagreen')),
            row=2, col=1)  
            fig.add_annotation(
                        x=norm_umass.columns[[int(np.where(norm_umass.var()== norm_umass.var().min())[0])]][0],
                        y=norm_umass[col2].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median and lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=2, col = 1)


        if col2 != norm_umass.columns[[int(np.where(norm_umass.median()== norm_umass.median().max())[0])]][0] and col2 != norm_umass.columns[[int(np.where(norm_umass.var()== norm_umass.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_umass[col2], name = col2, fillcolor="slategray",  line=dict(color='black')),
            row=2, col=1)


    for col3 in norm_npmi: 
        
        if col3 == norm_npmi.columns[[int(np.where(norm_npmi.median()
            == norm_npmi.median().max())[0])]][0] and col3 != norm_npmi.columns[[int(np.where(norm_npmi.var()
            == norm_npmi.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_npmi[col3], name = col3, fillcolor="LightSalmon",  line=dict(color='darkred')),
            row = 3, col=1)  
            fig.add_annotation(
                        x=norm_npmi.columns[[int(np.where(norm_npmi.median()== norm_npmi.median().max())[0])]][0],
                        y=norm_npmi[col3].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row = 3, col = 1)


        if col3 != norm_npmi.columns[[int(np.where(norm_npmi.median()
        == norm_npmi.median().max())[0])]][0] and col3 == norm_npmi.columns[[int(np.where(norm_npmi.var()== norm_npmi.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_npmi[col3], name = col3, fillcolor="Yellow",  line=dict(color='orange')),
            row = 3, col=1)  
            fig.add_annotation(
                        x=norm_npmi.columns[[int(np.where(norm_npmi.var()== norm_npmi.var().min())[0])]][0],
                        y=norm_npmi[col3].max(), 
                        xref="x",
                        yref="y",
                        text="Lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row = 3, col = 1)

        if col3 == norm_npmi.columns[[int(np.where(norm_npmi.median()
        == norm_npmi.median().max())[0])]][0] and col3 == norm_npmi.columns[[int(np.where(norm_npmi.var()== norm_npmi.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_npmi[col3], name = col3, fillcolor="Lightgreen",  line=dict(color='Seagreen')),
            row = 3, col=1)  
            fig.add_annotation(
                        x=norm_npmi.columns[[int(np.where(norm_npmi.var()== norm_npmi.var().min())[0])]][0],
                        y=norm_npmi[col3].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median and lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row = 3, col = 1)


        if col3 != norm_npmi.columns[[int(np.where(norm_npmi.median()== norm_npmi.median().max())[0])]][0] and col3 != norm_npmi.columns[[int(np.where(norm_npmi.var()== norm_npmi.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_npmi[col3], name = col3, fillcolor="slategray",  line=dict(color='black')),
            row = 3, col=1)


    for col4 in norm_uci: 
        
        if col4 == norm_uci.columns[[int(np.where(norm_uci.median()
            == norm_uci.median().max())[0])]][0] and col4 != norm_uci.columns[[int(np.where(norm_uci.var()
            == norm_uci.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_uci[col4], name = col4, fillcolor="LightSalmon",  line=dict(color='darkred')),
            row=4, col=1)  
            fig.add_annotation(
                        x=norm_uci.columns[[int(np.where(norm_uci.median()== norm_uci.median().max())[0])]][0],
                        y=norm_uci[col4].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=4, col = 1)


        if col4 != norm_uci.columns[[int(np.where(norm_uci.median()
        == norm_uci.median().max())[0])]][0] and col4 == norm_uci.columns[[int(np.where(norm_uci.var()== norm_uci.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_uci[col4], name = col4, fillcolor="Yellow",  line=dict(color='orange')),
            row=4, col=1)  
            fig.add_annotation(
                        x=norm_uci.columns[[int(np.where(norm_uci.var()== norm_uci.var().min())[0])]][0],
                        y=norm_uci[col4].max(), 
                        xref="x",
                        yref="y",
                        text="Lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=4, col = 1)

        if col4 == norm_uci.columns[[int(np.where(norm_uci.median()
        == norm_uci.median().max())[0])]][0] and col4 == norm_uci.columns[[int(np.where(norm_uci.var()== norm_uci.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_uci[col4], name = col4, fillcolor="Lightgreen",  line=dict(color='Seagreen')),
            row=4, col=1)  
            fig.add_annotation(
                        x=norm_uci.columns[[int(np.where(norm_uci.var()== norm_uci.var().min())[0])]][0],
                        y=norm_uci[col4].max(), 
                        xref="x",
                        yref="y",
                        text="Highest Median and lowest Variance", 
                        showarrow=True,
                        yshift=10, 
                        font=dict(
            family="Arial, monospace",
            size=16,
            color="Black"
            ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver",
        opacity=0.8, row=4, col = 1)


        if col4 != norm_uci.columns[[int(np.where(norm_uci.median()== norm_uci.median().max())[0])]][0] and col4 != norm_uci.columns[[int(np.where(norm_uci.var()== norm_uci.var().min())[0])]][0]:
            fig.add_trace(
            go.Box(y = norm_uci[col4], name = col4, fillcolor="slategray",  line=dict(color='black')),
            row=4, col=1)


    fig.update_layout( height=1200, width=1200, title_text=Title_box, boxgap = 0, showlegend=False,  boxmode = "overlay")
    fig.show()


def plot_hist(Title_hist, d_cv=d_cv, d_uci=d_uci, d_npmi=d_npmi, d_umass=d_umass):
    norm = Normalizer()
    fig = make_subplots(rows=4, cols=1, subplot_titles= ["Stability Coherence Cv",  "Stability Coherence UMass", "Stability Coherence NPMI", "Stability Coherence UCI",])
   
# best k in each row Cv
    best_k_cv = []
    for row in range(len(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_cv)),columns = pd.DataFrame(d_cv).columns))):
        best = np.where(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_cv)),columns = pd.DataFrame(d_cv).columns).iloc[row,:] == pd.DataFrame(norm.fit_transform(pd.DataFrame(d_cv)),columns = pd.DataFrame(d_cv).columns).iloc[row,:].max())
        best_k_cv.append(int(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_cv)),columns = pd.DataFrame(d_cv).columns).columns[best].tolist()[0].split(" ")[0]))

# best k in each row UMass
    best_k_umass = []
    for row in range(len(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_umass)),columns = pd.DataFrame(d_umass).columns))):
        best = np.where(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_umass)),columns = pd.DataFrame(d_umass).columns).iloc[row,:] == pd.DataFrame(norm.fit_transform(pd.DataFrame(d_umass)),columns = pd.DataFrame(d_umass).columns).iloc[row,:].max())
        best_k_umass.append(int(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_umass)),columns = pd.DataFrame(d_umass).columns).columns[best].tolist()[0].split(" ")[0]))

# best k in each row Npmi
    best_k_npmi = []
    for row in range(len(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_npmi)),columns = pd.DataFrame(d_npmi).columns))):
        best = np.where(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_npmi)),columns = pd.DataFrame(d_npmi).columns).iloc[row,:] == pd.DataFrame(norm.fit_transform(pd.DataFrame(d_npmi)),columns = pd.DataFrame(d_npmi).columns).iloc[row,:].max())
        best_k_npmi.append(int(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_npmi)),columns = pd.DataFrame(d_npmi).columns).columns[best].tolist()[0].split(" ")[0]))

# best k in each row Uci
    best_k_uci = []
    for row in range(len(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_uci)),columns = pd.DataFrame(d_uci).columns))):
        best = np.where(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_uci)),columns = pd.DataFrame(d_uci).columns).iloc[row,:] == pd.DataFrame(norm.fit_transform(pd.DataFrame(d_uci)),columns = pd.DataFrame(d_uci).columns).iloc[row,:].max())
        best_k_uci.append(int(pd.DataFrame(norm.fit_transform(pd.DataFrame(d_uci)),columns = pd.DataFrame(d_uci).columns).columns[best].tolist()[0].split(" ")[0]))



    fig.add_trace(
    go.Bar(x = pd.Series(best_k_cv).value_counts().index.unique(), y= pd.Series(best_k_cv).value_counts(), name = "cv", text= pd.Series(best_k_cv).value_counts(), textposition="auto", marker = dict(color = 'seagreen'),textangle=360,textfont=dict(
        color='black',
        size=14, #can change the size of font here
        family='Arial'
     )),row=1, col=1)

    fig.add_trace(
    go.Bar(x = pd.Series(best_k_umass).value_counts().index.unique(), y= pd.Series(best_k_umass).value_counts(),name = "umass", text= pd.Series(best_k_umass).value_counts(),textposition="auto", marker = dict(color = 'skyblue'),textangle=360,textfont=dict(
        color='black',
        size=14, #can change the size of font here
        family='Arial'
     )),row=2, col=1)

    fig.add_trace(
    go.Bar(x = pd.Series(best_k_npmi).value_counts().index.unique(),y= pd.Series(best_k_npmi).value_counts(), name = "npmi", text= pd.Series(best_k_npmi).value_counts(),textposition="auto", marker = dict(color = 'slategray'),textangle=360,textfont=dict(
        color='black',
        size=14, #can change the size of font here
        family='Arial'
     )),row=3, col=1)

    fig.add_trace(
    go.Bar(x = pd.Series(best_k_uci).value_counts().index.unique(), y= pd.Series(best_k_uci).value_counts(),name = "uci", text= pd.Series(best_k_uci).value_counts(),textposition="auto", marker = dict(color = 'steelblue'), textangle=360, textfont=dict(
        color='black',
        size=14, #can change the size of font here
        family='Arial'
     )),row =4, col=1)


    fig.add_vrect(x0 = pd.Series(best_k_cv).value_counts().index[int(np.where(pd.Series(best_k_cv).value_counts() == pd.Series(best_k_cv).value_counts().max())[0])]-1, x1=pd.Series(best_k_cv).value_counts().index[int(np.where(pd.Series(best_k_cv).value_counts() == pd.Series(best_k_cv).value_counts().max())[0])]+1
    ,annotation_text="Best K: {}".format(pd.Series(best_k_cv).value_counts().index[int(np.where(pd.Series(best_k_cv).value_counts() == pd.Series(best_k_cv).value_counts().max())[0])]), annotation_position="outside right",
    fillcolor="red", opacity=0.25, line_width=0,  annotation=dict(font_size=14, font_family="Arial",align="center",showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver"), layer="above", row = 1, col = 1)

    fig.add_vrect(x0 = pd.Series(best_k_umass).value_counts().index[int(np.where(pd.Series(best_k_umass).value_counts() == pd.Series(best_k_umass).value_counts().max())[0])]-1, x1=pd.Series(best_k_umass).value_counts().index[int(np.where(pd.Series(best_k_umass).value_counts() == pd.Series(best_k_umass).value_counts().max())[0])]+1
    ,annotation_text="Best K: {}".format(pd.Series(best_k_umass).value_counts().index[int(np.where(pd.Series(best_k_umass).value_counts() == pd.Series(best_k_umass).value_counts().max())[0])]), annotation_position="outside right",
    fillcolor="red", opacity=0.25, line_width=0,  annotation=dict(font_size=14, font_family="Arial",align="center",showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver"), layer="above", row = 2, col = 1)

    fig.add_vrect(x0 = pd.Series(best_k_npmi).value_counts().index[int(np.where(pd.Series(best_k_npmi).value_counts() == pd.Series(best_k_npmi).value_counts().max())[0])]-1, x1=pd.Series(best_k_npmi).value_counts().index[int(np.where(pd.Series(best_k_npmi).value_counts() == pd.Series(best_k_npmi).value_counts().max())[0])]+1
    ,annotation_text="Best K: {}".format(pd.Series(best_k_npmi).value_counts().index[int(np.where(pd.Series(best_k_npmi).value_counts() == pd.Series(best_k_npmi).value_counts().max())[0])]), annotation_position="outside right",
    fillcolor="red", opacity=0.25, line_width=0,  annotation=dict(font_size=14, font_family="Arial",align="center",showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver"), layer="above", row = 3, col = 1)

    fig.add_vrect(x0 = pd.Series(best_k_uci).value_counts().index[int(list(np.where(pd.Series(best_k_uci).value_counts() == pd.Series(best_k_uci).value_counts().max())[0]) [-1])]-1, x1=pd.Series(best_k_uci).value_counts().index[int(list(np.where(pd.Series(best_k_uci).value_counts() == pd.Series(best_k_uci).value_counts().max())[0]) [-1])]+1
    ,annotation_text="Best K: {}".format(pd.Series(best_k_uci).value_counts().index[int(list(np.where(pd.Series(best_k_uci).value_counts() == pd.Series(best_k_uci).value_counts().max())[0]) [-1])]), annotation_position="outside right",
    fillcolor="red", opacity=0.25, line_width=0,  annotation=dict(font_size=14, font_family="Arial",align="center",showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=20,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="silver"), layer="above", row = 4, col = 1)



    fig.update_layout( height=1000, width=1200, title_text=Title_hist, showlegend=False, xaxis_title_text='Coherence Value', #uniformtext_mode='hide',
    yaxis_title_text='Count', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1)
    fig.show()


def visualize(path, Title_box , Title_hist, d_cv=d_cv, d_uci=d_uci, d_npmi=d_npmi, d_umass=d_umass):
    plot_box(Title_box=Title_box,d_cv=d_cv, d_uci=d_uci, d_npmi=d_npmi, d_umass=d_umass)
    plot_hist(Title_hist=Title_hist,d_cv=d_cv, d_uci=d_uci, d_npmi=d_npmi, d_umass=d_umass)

def prepare_sets(dfs):

    df_cv = pd.DataFrame([])
    df_umass = pd.DataFrame([])
    df_npmi = pd.DataFrame([])
    df_uci = pd.DataFrame([])


    for i in dfs:
        values_cv = []
        values_umass = []
        values_npmi = []
        values_uci = []

        for name in i.columns:
            if " c_v" in name:
                values_cv.append(name)
            if " c_umass" in name:
                values_umass.append(name)
            if " c_npmi" in name:
                values_npmi.append(name)
            if " c_uci" in name:
                values_uci.append(name)

        d_cv = {}
        for col in values_cv:
            d_cv[col]=list(i[col])
        df_cv = pd.concat([df_cv, pd.DataFrame(d_cv, columns = values_cv)], axis = 1)


        d_umass = {}
        for col in values_umass:
            d_umass[col]=list(i[col])
        df_umass = pd.concat([df_umass, pd.DataFrame(d_umass, columns = values_umass)], axis = 1)



        d_npmi = {}
        for col in values_npmi:
            d_npmi[col]=list(i[col])
        df_npmi = pd.concat([df_npmi, pd.DataFrame(d_npmi, columns = values_npmi)], axis = 1)




        d_uci = {}
        for col in values_uci:
            d_uci[col]=list(i[col])
        df_uci = pd.concat([df_uci, pd.DataFrame(d_uci, columns = values_uci)], axis = 1)

    
    norm = Normalizer()
    df_cv = pd.DataFrame(norm.fit_transform(df_cv))
    df_umass = pd.DataFrame(norm.fit_transform(df_umass))
    df_npmi = pd.DataFrame(norm.fit_transform(df_npmi))
    df_uci = pd.DataFrame(norm.fit_transform(df_uci))
    return df_cv, df_umass, df_npmi, df_uci

    
def show_var( path,d_cv=d_cv, d_umass=d_umass, d_npmi=d_npmi, d_uci=d_uci):

    variances_cv = {}
    variances_umass = {}
    variances_npmi = {}
    variances_uci = {}
    
    ## CV
    
    for i in range(len(d_cv.columns)):
        if int(d_cv.columns[i].split(" ")[0]) not in variances_cv.keys() :
            variances_cv[int(d_cv.columns[i].split(" ")[0])] = []

        variances_cv[int(d_cv.columns[i].split(" ")[0])].append(round(d_cv.iloc[:,i].var(), 5))

    ## UMass
    
    for i in range(len(d_umass.columns)):
        if int(d_umass.columns[i].split(" ")[0]) not in variances_umass.keys() :
            variances_umass[int(d_umass.columns[i].split(" ")[0])] = []
 
        variances_umass[int(d_umass.columns[i].split(" ")[0])].append(round(d_umass.iloc[:,i].var(), 5)) 

    ## NPMI
    
    for i in range(len(d_npmi.columns)):
        if int(d_npmi.columns[i].split(" ")[0]) not in variances_npmi.keys() :
            variances_npmi[int(d_npmi.columns[i].split(" ")[0])] = []

        variances_npmi[int(d_npmi.columns[i].split(" ")[0])].append(round(d_npmi.iloc[:,i].var(), 5))

    ## UCI
    
    for i in range(len(d_uci.columns)):
        if int(d_uci.columns[i].split(" ")[0]) not in variances_uci.keys() :
            variances_uci[int(d_uci.columns[i].split(" ")[0])] = []

        variances_uci[int(d_uci.columns[i].split(" ")[0])].append(round(d_uci.iloc[:,i].var(), 5))


    res_cv = pd.DataFrame(variances_cv).T
    res_umass = pd.DataFrame(variances_umass).T
    res_npmi = pd.DataFrame(variances_npmi).T
    res_uci = pd.DataFrame(variances_uci).T
    res_cv.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]
    res_umass.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]
    res_npmi.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]
    res_uci.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]

    res_cv.to_excel("../Stability/cv_var.xlsx", engine='openpyxl')
    res_umass.to_excel("../Stability/umass_var.xlsx", engine='openpyxl')
    res_npmi.to_excel("../Stability/npmi_var.xlsx", engine='openpyxl')
    res_uci.to_excel("../Stability/uci_var.xlsx", engine='openpyxl')
    return res_cv, res_umass, res_npmi, res_uci


def show_median( path,d_cv=d_cv, d_umass=d_umass, d_npmi=d_npmi, d_uci=d_uci):

    medians_cv = {}
    medians_umass = {}
    medians_npmi = {}
    medians_uci = {}
    
    ## CV
    
    for i in range(len(d_cv.columns)):
        if int(d_cv.columns[i].split(" ")[0]) not in medians_cv.keys() :
            medians_cv[int(d_cv.columns[i].split(" ")[0])] = []

        medians_cv[int(d_cv.columns[i].split(" ")[0])].append(round(d_cv.iloc[:,i].median(), 5))

    ## UMass
    
    for i in range(len(d_umass.columns)):
        if int(d_umass.columns[i].split(" ")[0]) not in medians_umass.keys() :
            medians_umass[int(d_umass.columns[i].split(" ")[0])] = []
 
        medians_umass[int(d_umass.columns[i].split(" ")[0])].append(round(d_umass.iloc[:,i].median(), 5)) 

    ## NPMI
    
    for i in range(len(d_npmi.columns)):
        if int(d_npmi.columns[i].split(" ")[0]) not in medians_npmi.keys() :
            medians_npmi[int(d_npmi.columns[i].split(" ")[0])] = []

        medians_npmi[int(d_npmi.columns[i].split(" ")[0])].append(round(d_npmi.iloc[:,i].median(), 5))

    ## UCI
    
    for i in range(len(d_uci.columns)):
        if int(d_uci.columns[i].split(" ")[0]) not in medians_uci.keys() :
            medians_uci[int(d_uci.columns[i].split(" ")[0])] = []

        medians_uci[int(d_uci.columns[i].split(" ")[0])].append(round(d_uci.iloc[:,i].median(), 5))


    res_cv = pd.DataFrame(medians_cv).T
    res_umass = pd.DataFrame(medians_umass).T
    res_npmi = pd.DataFrame(medians_npmi).T
    res_uci = pd.DataFrame(medians_uci).T
    res_cv.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]
    res_umass.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]
    res_npmi.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]
    res_uci.columns = ["LSA TF", "LSA TF-IDF", "LDA TF", "LDA TF-IDF", "NMF TF", "NMF TF-IDF"]

    res_cv.to_excel("../Stability/cv_median.xlsx", engine='openpyxl')
    res_umass.to_excel("../Stability/umass_median.xlsx", engine='openpyxl')
    res_npmi.to_excel("../Stability/npmi_median.xlsx", engine='openpyxl')
    res_uci.to_excel("../Stability/uci_median.xlsx", engine='openpyxl')
    return res_cv, res_umass, res_npmi, res_uci




# Visualizations
#import octis
#import pyLDAvis











#Combining Data 

def combine_files(file_path, number_of_rows:int = 100):
    list_of_org_files = list(os.listdir(path=file_path))
    combined_files = []
    for file in list_of_org_files:
        print("Is reading " + file + "...")
        data = pd.read_csv(file_path + "/" + file, sep="\t", compression = 'gzip', nrows=number_of_rows)
        combined_files.append(data)
        pd.concat(combined_files)
        print("File added!")


