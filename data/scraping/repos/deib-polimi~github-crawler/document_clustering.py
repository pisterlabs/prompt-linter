from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn.externals import joblib
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mpld3
from nltk.tag import pos_tag
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import ward, dendrogram
import string
from nltk.tag import pos_tag
from sklearn.manifold import MDS
from gensim import corpora, models, similarities 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import CoherenceModel
from sklearn.cluster import KMeans
import json 
import logging

#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s: %(name)s: %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')
logger = logging.getLogger('document-clustering')

RESULT_FOLDER="ansible-results"

logger.info("Loading input text corpus.")

commits = open(os.path.join(RESULT_FOLDER,'commits.txt')).read().split('\n')

messages = open(os.path.join(RESULT_FOLDER,'deletions-cleaned.txt')).read().split('\nBREAKS HERE\n')
    
messages=[m.decode('utf-8') for m in messages]

# clean empty commit meggases (or code snippets, when working with code clustering)
bad_indexes = []

for i in range(len(commits)):
    if(messages[i] == ""):
        bad_indexes.append(i)

commits=list(commits[i] for i in range(len(commits)) if not i in bad_indexes)
messages=list(messages[i] for i in range(len(messages)) if not i in bad_indexes)

print(str(len(commits)) + ' commits')
print(str(len(messages)) + ' messages')

# generates index for each item in the corpora (in this case it's just rank) that we'll be used for scoring later
ranks = []

for i in range(0,len(commits)):
    ranks.append(i)
    
# load nltk's English stopwords 
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# load nltk's SnowballStemmer 
stemmer = SnowballStemmer("english")

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in messages:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
# create a data frame that associates to each word resulting from the tokenization 
# of all the input text samples with the corresponding stemmed version
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

# define the TFIDF vectorizer
# all the parameters (maybe excluded "stop_words") could be tuned
# it should be reasonable to use tokenize_only for code and tokenize_and_stem for text (like commmit messages)

logger.info("Computing TFIDF matrix.")

TOKENIZE_ONLY=True
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=20000,
                                 min_df=0.005, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_only if TOKENIZE_ONLY else tokenize_and_stem, ngram_range=(1,1))


tfidf_matrix = tfidf_vectorizer.fit_transform(messages)

terms = tfidf_vectorizer.get_feature_names()
logger.info("Resulting number of features: " + str(len(terms)))

dist = 1 - cosine_similarity(tfidf_matrix)

#################### CLUSTERING

logger.info("Run Kmeans clustering.")
num_clusters = 2
 
for k in range(2, num_clusters + 1):
    km = KMeans(n_clusters=k)
     
    km.fit(tfidf_matrix)
     
    clusters = km.labels_.tolist()
     
    joblib.dump(km,  os.path.join(RESULT_FOLDER,'doc_cluster_' + str(k) + '.pkl'))
    km = joblib.load(os.path.join(RESULT_FOLDER,'doc_cluster_' + str(k) + '.pkl'))
    clusters = km.labels_.tolist()
     
    commit_messages = { 'commit': commits, 'rank': ranks, 'message': messages, 'cluster': clusters }
     
    frame = pd.DataFrame(commit_messages, index = [clusters] , columns = ['rank', 'commit', 'cluster'])
     
    frame['cluster'].value_counts()
     
    grouped = frame['rank'].groupby(frame['cluster'])
    
    clusters_words={}
    
    print("Top terms per cluster:")
    print()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(frame.index.levels[0].size):
        print("Cluster %d words:" % i, end='')
        words = []
        for ind in order_centroids[i, :10]:
            if not TOKENIZE_ONLY:
                print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
                words.append(vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
            else:
                print(' %s' % terms[ind].split(' ')[0].encode('utf-8', 'ignore'), end=',')
                words.append(terms[ind].split(' ')[0].encode('utf-8', 'ignore'))
        clusters_words.update({i: words})
         
    # dump clusters words to file
    with open(os.path.join(RESULT_FOLDER,'clusters-words-' + str(k) +'.json'), 'wb') as outfile:
        json.dump(clusters_words, outfile)

    # dumping assignment of commits to clusters
    frame.set_index("rank").to_csv("/home/warmik/eclipse-workspace/iac-crawler/ml/ansible-results/commits-clustering.csv", index= False)
    
    logger.info("Scaling down document vectors to plot clusters in 2 dimensions.")
    MDS()
     
    # two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
     
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
     
    xs, ys = pos[:, 0], pos[:, 1]
     
    #set up colors per clusters using a dict
    #cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
     
    #create data frame that has the result of the MDS plus the cluster numbers and commits
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, commit=commits)) 
     
    #group by cluster
    groups = df.groupby('label')
     
    # set up plot
     
    logger.info("Plotting clusters.")
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
     
    #iterate through groups to layer the plot
    #I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')
         
    ax.legend(numpoints=1)  #show legend with only 1 point
     
    #add label in x,y position with the label as the commit
    # for i in range(len(df)):
    #     ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['commit'], size=8)  
     
         
         
    #plt.show() #show the plot
     
    #uncomment the below to save the plot if need be
    logger.info("Saving output clustering image.")
    plt.savefig(os.path.join(RESULT_FOLDER,'clusters_small_noaxes' + str(k) + '.png'), dpi=200)
     
    plt.close()

###################### HIERARCHICAL

logger.info("Computing dendrogram (hierarchical clustering)")
linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
  
fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=commits);
  
plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
  
plt.tight_layout() #show plot with tight layout
  
#uncomment below to save figure
logger.info("Saving dendrogram to file.")
plt.savefig(os.path.join(RESULT_FOLDER,'ward_clusters.png'), dpi=200) #save figure as ward_clusters
  
plt.close()

##################### LDA

#strip any proper names from a text...unfortunately right now this is yanking the first word from a sentence too.
def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.islower()]
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
 
#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns
 
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super(DecimalEncoder, self).default(o)
     
#Latent Dirichlet Allocation implementation with Gensim
 
# to tune number of topics
n_topics=2

for t in range(2,n_topics + 1):
    #remove proper names
    logger.info("Running Latent Dirichlet Allocation for n_topics: " + str(t) + ".")
    preprocess = [strip_proppers(doc) for doc in messages]
    
    if not TOKENIZE_ONLY:
        tokenized_text = [tokenize_and_stem(text) for text in preprocess]
    else:
        tokenized_text = [tokenize_only(text) for text in preprocess]
    
    texts = [[word for word in text if word not in stopwords] for text in tokenized_text]
     
    dictionary = corpora.Dictionary(texts)
     
    dictionary.filter_extremes(no_below=1, no_above=0.8)
     
    corpus = [dictionary.doc2bow(text) for text in texts]
     
    len(corpus)
     
    lda = models.LdaModel(corpus, num_topics=t, id2word=dictionary, update_every=5, chunksize=100, passes=5)
     
    # to tune number of words per topics
    topics = lda.print_topics(t, num_words=3)
     
    topics_matrix = lda.show_topics(formatted=False, num_words=10)
    for topic in topics_matrix:
        for ii in range(len(topic[1])):
            w=topic[1][ii]
            w=(w[0],float(w[1]))
            topic[1][ii] = w
            
    # compute model perplexity and coherence score to evaluate goodness of identified topics        
    # Compute Perplexity (lower the better)
    perplexity = lda.log_perplexity(corpus)
    
    # Compute Coherence Score (higher the better)
    coherence_model_lda = CoherenceModel(model=lda, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()

    output = {}
    output.update({"topics_matrix": topics_matrix})
    output.update({"perplexity": perplexity})
    output.update({"coherence": coherence})
                
                
    with open(os.path.join(RESULT_FOLDER,'topics_' + str(t) +'.json'), 'wb') as outfile:
        json.dump(output, outfile)
        

# ##################### DOC2VEC
# 
# tokenize_and_stemmed_messages=[]
# for m in messages:
#     tokenize_and_stemmed_messages.append(tokenize_and_stem(m))
#     
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenize_and_stemmed_messages)]
# model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
# trained_vectors = model.docvecs
# # K-MEANS ON DOC2VEC
# 
# km = KMeans(n_clusters=num_clusters)
# num_clusters = 3
# 
# docvecs=[]
# for v in trained_vectors:
#     docvecs.append(v)
# 
# doc2vec_dist = 1 - cosine_similarity(tfidf_matrix)
# 
# 
# km.fit(docvecs)
# 
# joblib.dump(km,  'doc2vec_cluster.pkl')
# km = joblib.load('doc2vec_cluster.pkl')
# 
# doc2vec_clusters = km.labels_.tolist()
# 
# commit_messages = { 'commit': commits, 'rank': ranks, 'message': messages, 'cluster': doc2vec_clusters }
# 
# frame = pd.DataFrame(commit_messages, index = [doc2vec_clusters] , columns = ['rank', 'commit', 'cluster'])
# 
# frame['cluster'].value_counts()
# 
# grouped = frame['rank'].groupby(frame['cluster'])
# 
# grouped.mean()
# 
# 
# print("Top terms per cluster (doc2vec):")
# print()
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
# for i in range(num_clusters):
#     print("Cluster %d words:" % i, end='')
#     for ind in order_centroids[i, :6]:
#         print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
#     print()
#     print()
#     print("Cluster %d commits:" % i, end='')
#     for commit in frame.ix[i]['commit'].values.tolist():
#         print(' %s,' % commit, end='')
#     print()
#     print()
# 
# # plotting as before
# 
# MDS()
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
# pos = mds.fit_transform(doc2vec_dist)  # shape (n_components, n_samples)
# xs, ys = pos[:, 0], pos[:, 1]
# #create data frame that has the result of the MDS plus the cluster numbers and commits
# df = pd.DataFrame(dict(x=xs, y=ys, label=doc2vec_clusters, commit=commits)) 
# groups = df.groupby('label')
# fig, ax = plt.subplots(figsize=(17, 9)) # set size
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, mec='none')
#     ax.set_aspect('auto')
#     ax.tick_params(\
#         axis= 'x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom='off',      # ticks along the bottom edge are off
#         top='off',         # ticks along the top edge are off
#         labelbottom='off')
#     ax.tick_params(\
#         axis= 'y',         # changes apply to the y-axis
#         which='both',      # both major and minor ticks are affected
#         left='off',      # ticks along the bottom edge are off
#         top='off',         # ticks along the top edge are off
#         labelleft='off')
#     
# ax.legend(numpoints=1)  #show legend with only 1 point
# 
# for i in range(len(df)):
#     ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['commit'], size=8)  
# 
# plt.show() #show the plot
# 
# plt.close()

# HIERACHICAL ON DOC2VEC



