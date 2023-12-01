# Import Analysis Libraries
import numpy as np
import pandas as pd
import random
import joblib
import json

# Import Visualization Libraries
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models

# Import Word Processing Libraries
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")
from nltk.stem.porter import *
import nltk
nltk.download('wordnet')

# Import Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel

# Configure Seed Generation
np.random.seed(400)
random.seed(400)


"==============================================================================================================================="
## Text Preprocessing Functions

def lem_stem(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text, deacc=True) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lem_stem(token))
    return result
"==============================================================================================================================="




"==============================================================================================================================="
## TFIDF Function

def tfidf(data, tokenizer_function, ngram_range):
    """
    ngram_range must be in tuple format. Ex. (1,2,3,...N)

    """
    tfidf_model = TfidfVectorizer(
                                    # max_df : maximum document frequency for the given word
                                    max_df=0.90,
                                    # max_features: maximum number of words -- ADJUST FOR Corpus Size
                                    max_features=500, 
                                    # min_df : minimum document frequency for the given word
                                    min_df=0.10,
                                    # use_idf: if not true, we only calculate tf
                                    use_idf=True, 
                                    tokenizer=tokenizer_function,
                                    # ngram_range: (min, max), eg. (1, 2) including 1-gram, 2-gram
                                    ngram_range=ngram_range)

    # Fit the TfidfVectorizer to the data
    tfidf_matrix = tfidf_model.fit_transform(data) 
    print(f"Total {tfidf_matrix.shape[0]} reviews and {tfidf_matrix.shape[1]} terms.")
    return tfidf_matrix
"==============================================================================================================================="




"==============================================================================================================================="
## Automate Best Number of Clusters Function

def get_cluster_labels(data, evaluate, n):
    """
    This method will automate the n_cluster selection
    data: 2-D Array
    evaluate: 0 if to be evaluated else 1 if best n_cluster if fixed
    n: if evaluate=0, number of clusters to try else best fit cluster if evaluate =1
    """
    score_k={}
    best_n_cluster = None
    best_n_cluster_val =None

    if not evaluate:
        for n_clusters in range(2,n+1):
            cluster = KMeans(n_clusters=n_clusters)
            preds = cluster.fit_predict(data)
            centers = cluster.cluster_centers_
            score = silhouette_score(data, preds)
            score_k[n_clusters] = score
            
        for key, val in score_k.items():
            if val == max(list(score_k.values())):
                best_n_cluster = key
                best_n_cluster_val=val
        print(f"Best N Cluster: {best_n_cluster}, score :{best_n_cluster_val}")
        get_cluster_labels(data,1,best_n_cluster)
        
    if evaluate:
        cluster = KMeans(n_clusters=n)
        preds = cluster.fit_predict(data)
        #print(preds)
        centers = cluster.cluster_centers_
        score = silhouette_score(data, preds)
        
    return best_n_cluster
"==============================================================================================================================="




"==============================================================================================================================="
## Kmeans Clustering Function

def kmeans_clustering(data,n_clusters, tfidf_matrix):
    K_model = KMeans(n_clusters=n_clusters)
    K_model.fit(tfidf_matrix)
    K_result = data.rename(columns={'review_body':'review'})
    K_result['clusters'] = K_model.labels_.tolist()

    cluster_size = K_result['clusters'].value_counts().to_frame()
    print(cluster_size,'\n')
    return K_result
"==============================================================================================================================="




"==============================================================================================================================="
## Create Dictionary for Cluster-Specific data Function

def create_cluster_dict(cluster_dataframe):
    cluster_dict = {}
    for x in range(len(cluster_dataframe.clusters.unique())):
        # Get each respective cluster data per loop
        data = cluster_dataframe.loc[cluster_dataframe['clusters']==x]
        # Assign respective cluster data to dictionary key as a list
        cluster_dict[f"Cluster_{x}"] = data['review'].to_list()
        print(f"Number of messages in Cluster {x}:", len(cluster_dict[f'Cluster_{x}']))
    print(f"\n{cluster_dict.keys()}")
    return cluster_dict
"==============================================================================================================================="





"==============================================================================================================================="
## Bag Of Word Corpus and Dictionary Creation Function

def produce_dict_corpus(data_list):
    # Preprocess each cluster's data and Store in List format
    processed_docs = [preprocess(document) for document in data_list]        
    
    # Create Dictionary from processed data containing the number of times a word appears
    id2word = gensim.corpora.Dictionary(processed_docs)
    # Filter out tokens that appear in
    # (1)- less than no_below documents (absolute number) or
    # (2)- more than no_above documents (fraction of total corpus size, not absolute number).
    # - after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
#     word_count_dict.filter_extremes(no_below=1, no_above=0.9, keep_n=None)

    ### Bag Of Words on Data
    bow_corpus = [id2word.doc2bow(doc) for doc in processed_docs]

#     # Test 1st Document from original Data
#     document_num = 0
#     bow_doc_x = bow_corpus[document_num]
#     for i in range(len(bow_doc_x)):
#         print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
#                                                          word_count_dict[bow_doc_x[i][0]], 
#                                                          bow_doc_x[i][1]))

    # Evaluate Coherence Score for each
    return processed_docs,bow_corpus,id2word
"==============================================================================================================================="




"==============================================================================================================================="
## Create LDA Model Function

"""
- **num_topics : number of requested latent topics to be extracted from the corpus.
- **id2word : mapping from word ids (integers) to words (strings). It is used to determine the vocabulary size, debugging and topic printing.
- **workers : number of extra processes to use for parallelization. Default: All.
- **alpha & **eta: hyperparameters that affect sparsity of the document-topic (theta) and topic-word (lambda) distributions. Default:1/num_topics

    - Alpha is the per document topic distribution.
        - High alpha: Every document has a mixture of all topics(documents appear similar to each other).
        - Low alpha: Every document has a mixture of very few topics
        
    - Eta is the per topic word distribution.
        - High eta: Each topic has a mixture of most words(topics appear similar to each other).
        - Low eta: Each topic has a mixture of few words.
        
**passes** : number of training passes through the corpus
"""
def create_lda_model(corpus, id2word, num_topics):
    # Training the LDA Multicore Model
    lda_model = gensim.models.LdaMulticore(
                                            corpus = corpus,
                                            num_topics = num_topics, 
                                            id2word = id2word,
                                            chunksize=len(corpus),
                                            passes = 100,
                                            workers = None,
                                            random_state=42
                                            )
    return lda_model
"==============================================================================================================================="




"==============================================================================================================================="
# Save & Load Model

def save_model(model,dir_path):
    joblib.dump(model, dir_path)

# Load Model
def load_model(dir_path):
    lda_model = joblib.load('Models/lda_model.jl')
    return lda_model
"==============================================================================================================================="





"==============================================================================================================================="
# Save/Load Dictionary as JSON file for easy access
def save_dict(dictionary,filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(dictionary))
        
def load_dict(filename):
    with open(filename) as f:
        dictionary = json.loads(f.read())
    return dictionary
"==============================================================================================================================="




"==============================================================================================================================="
# Save Visualized Topics with PyLDAvis Module as HTML

def visualize(lda_model,corpus,id2word,key): 
    # pyLDAvis.enable_notebook()
    plot = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(plot, f'Images/LDA_{key}.html')
"==============================================================================================================================="




"==============================================================================================================================="
# Generate list of topic words Function
def get_topic_words(lda_model,id2word,num_topics):
    topic_dict = {}
    topic_words = []
    for i in range(num_topics):
        # get top 10 words for each topic
        tt = lda_model.get_topic_terms(i,10)
        topic_words.append([id2word[pair[0]] for pair in tt])
        
#     print(f"Number of Topics: {len(topic_words)}")
    count = 0
    for i in topic_words:
        topic_dict[f'Topic_{count}'] = i
        count+=1
    return topic_dict
"==============================================================================================================================="




"==============================================================================================================================="
# Data format: dict{key: [list of sentences]}
def execute_loop(data, n_topics):
    # create empty dictionary to store each cluster's results
    stored_dict = {}
    # Outer Loop through each Cluster
    for key in data.keys():
#         print(key)
        # prepare BOW corpus and word_count_dictionary for cluster-specific LDA
        processed_docs,bow_corpus,word_count_dict = produce_dict_corpus(data_list=data[key])

        # Build LDA model to analyze data list
        lda_model = create_lda_model(corpus=bow_corpus, 
                                     id2word=word_count_dict,
                                     num_topics=n_topics)
        # Evaluate Coherence Score for model
        coherence_score = calc_coherence(model=lda_model, 
                                         texts=processed_docs, 
                                         dictionary=word_count_dict)
        # Get Cluster-specific keywords
        results = get_topic_words(lda_model=lda_model,   
                                  id2word=word_count_dict, 
                                  num_topics=n_topics)
        # Store Results from clustering
        stored_dict.update({key: results})
        # print Keywords, Weights, corpus Perplexity & Coherence Score from LDA Model per cluster, per topic
        print_scores(model=lda_model, 
                     corpus=bow_corpus, 
                     key=key, 
                     coherence=coherence_score)
        # Save pyLDAvis as html ---- Open in Browser Tab
        visualize(lda_model=lda_model, 
                  corpus=bow_corpus, 
                  id2word=word_count_dict,
                  key=key)

        print(f"{key} Complete")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("\n")
        
    print("Cluster Analysis Complete")
    return stored_dict, lda_model
"==============================================================================================================================="




"==============================================================================================================================="
# Evaluate Model Performance using Coherence & Perplexity Scores
def calc_coherence(model,texts,dictionary):
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model = model, 
                                         texts=texts, 
                                         dictionary=dictionary, 
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    coherence_lda = round(coherence_lda, 4)
    return coherence_lda
"==============================================================================================================================="




"==============================================================================================================================="
# Show LDA Topics with Weights
def print_scores(model,corpus,key,coherence):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    for topic in model.show_topics(num_topics=-1, num_words=10, log=False, formatted=False):
        print(f"Topic: {topic[0]}")
        print(f"Keywords: {topic[1]}")
        print(f'Perplexity: {round(model.log_perplexity(chunk=corpus),4)}')
        print(f"Coherence Score: {coherence}", "\n")
"==============================================================================================================================="




"==============================================================================================================================="
# Function that creates csv from cluster data dictionary for export
def create_cluster_datacsv(cluster_dict):
    cluster_frame = pd.DataFrame.from_dict(cluster_dict,orient='index')
    cluster_frame = cluster_frame.transpose()
    cluster_frame.to_csv('data/cluster_data/cluster_data.csv', index=False)
"==============================================================================================================================="




"==============================================================================================================================="
# Function to Extract Keywords from Clustered_dictionary data
def get_keywords(multindex_dictionary):
    
    for topic in multindex_dictionary:
        print(topic)
        keyword_list = []
        for keywords in multindex_dictionary[topic]:
            keyword_list.append(keywords)
        print(keyword_list)
    return keyword_list
"==============================================================================================================================="



