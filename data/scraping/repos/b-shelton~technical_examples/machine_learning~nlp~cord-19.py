# This script uses an aggregation of COVID-19 research papers to create ML-generate topic-modeling metadata associated with each paper. 

import os
import zipfile
import tempfile
import json
import numpy as np
import pandas as pd
import re
from langdetect import detect
from time import process_time
import multiprocessing as mp
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# SET THIS APPROPRIATELY to analyze either abstracts or full texts
focus = 'abstract' # 'abstract' or 'body_text'


# If not in a Kaggle notebook, configure environment to download data from Kaggle API (one time activity)
# Follow instructions here: https://medium.com/@ankushchoubey/how-to-download-dataset-from-kaggle-7f700d7f9198
#os.system('kaggle datasets download -d allen-institute-for-ai/CORD-19-research-challenge')
zippath = 'CORD-19-research-challenge.zip'

# Create the temporary directory to store the zip file's content
temp_dir = tempfile.TemporaryDirectory()

# Extract the zip file's content into the temporary directory
with zipfile.ZipFile(zippath, 'r') as zip_ref:
    zip_ref.extractall(temp_dir.name)

# Read the metadata.csv file
md = pd.read_csv(temp_dir.name + '/metadata.csv')



###############################################################################
# Read all of the text from the research papers
###############################################################################

sources = ['biorxiv_medrxiv',
           'comm_use_subset',
           'noncomm_use_subset',
           'custom_license']

papers_source = []
papers_sha = []
papers_text = []
for h in sources:
    paper_path = '/' + h + '/' + h + '/'
    for i in range(0, len(os.listdir(temp_dir.name + paper_path))):
        # read json file
        sha = os.listdir(temp_dir.name + paper_path)[i]
        json_path = (temp_dir.name + paper_path + sha)
        with open(json_path) as f:
            d = json.load(f)

        if len(d[focus]) == 0:
            continue
        else:
            # get text
            paper_text = []
            for j in range(0, len(d[focus])):
                if len(paper_text) == 0:
                    paper_text = d[focus][j]['text']
                else:
                    paper_text += d[focus][j]['text']

            # append to the rest of the extracted papers
            papers_source.append(h)
            papers_sha.append(re.sub('.json', '', sha))
            papers_text.append(paper_text)


df = pd.DataFrame({'sha': papers_sha,
                   'source': papers_source,
                   'text': papers_text})
df = df[df['text'].str.len() > 5]

# Only retain research papers in English (for now)
df['language'] = df['text'].apply(detect)

df.groupby('language')['sha'].count() \
    .reset_index().sort_values('sha', ascending = False)

df = df[df['language'] == 'en']

###############################################################################
# Pre-Processing
###############################################################################
'''
 This section will clean the text to prepare if for analysis, including transformation
 to all lowercase, tokenization, stemming (PortStemmer), and removing stop words.

 This section uses the multiprocessing package, which takes advantage of all the
 operating system's cores. I've hard coded the number of cores to 4, but the user
 can identify how many cores they have available by running `mp.cpu_count()`.

 Even with the multiprocessing package, it takes a long time to stem every word
 in the 30k~ papers.
'''

# make every word lowercase
papers_lower = [x.lower() for x in df['text'].tolist()]


# tokenize every paper, using multiprocessing
tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')

def token_a_paper(paper_lower):
    return tokenizer.tokenize(paper_lower)

t1_start = process_time()
pool = mp.Pool(4)
token_papers = list(pool.map(token_a_paper, papers_lower))
t1_end = process_time()
print('Time to tokenize:', round(t1_end - t1_start, 2), 'seconds')
pool.close()


# remove stop words (including customs stop words)
custom_to_exclude = {'et', 'al', 'al.', 'preprint', 'copyright', 'peer-review',
                     'author/fund', 'http', 'licens', 'biorxiv', 'fig',
                     'figure', 'medrxiv', 'i.e.', 'e.g.', 'e.g.,', '\'s', 'doi',
                     'author', 'funder', 'https', 'license'}
stop_words = set(stopwords.words('english')) | custom_to_exclude

st_words = []
for i in token_papers:
     t = [word for word in i if (not word in stop_words)]
     st_words.append(t)


# stem every remaining word, using multiprocessing
stemmer = PorterStemmer()
def stem_tokens(st_paper):
    return [stemmer.stem(word) for word in st_paper]

t1_start = process_time()
pool = mp.Pool(4)
stemmed_words = pool.map(stem_tokens, st_words)
t1_end = process_time()
print('Time to stem:', round((t1_end - t1_start) / 60, 2), 'minutes')
pool.close()


# count how many words after stop words are removed
# and put the tokenized words back into papers
counter = 0
stemmed_papers = []
for i in stemmed_words:
    paper = " ".join(i)
    stemmed_papers.append(paper)
    counter += len(i)
print('Number of total words:', counter)


# show top words in corpus
flat_words =[]
for i in stemmed_words:
    flat_words += i

fw = pd.DataFrame({'words': flat_words,
                   'occurences': 1})
gfw = fw.groupby('words')['occurences'].count() \
    .reset_index().sort_values('occurences', ascending = False)
gfw.head(25)



###############################################################################
# Topic Modeling with Latent Dirichlet Allocation (LDA)
# and NMF
###############################################################################
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


custom_to_exclude = ['et', 'al', 'al.', 'preprint', 'copyright', 'peer-review',
                     'author/fund', 'http', 'licens', 'biorxiv', 'fig',
                     'figure', 'medrxiv', 'i.e.', 'e.g.', 'e.g.,', '\'s', 'doi',
                     'author', 'funder', 'https', 'license']
my_stop_words = text.ENGLISH_STOP_WORDS.union(custom_to_exclude)

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=my_stop_words)
tfidf = tfidf_vectorizer.fit_transform(stemmed_papers)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=my_stop_words)
tf = tf_vectorizer.fit_transform(stemmed_papers)
tf_feature_names = tf_vectorizer.get_feature_names()


from sklearn.decomposition import NMF, LatentDirichletAllocation

no_topics = 10

# Run NMF
nmf = NMF(n_components = no_topics,
          random_state = 1,
          alpha = .1,
          l1_ratio = .5,
          init = 'nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_components = no_topics,
                                max_iter = 5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state=0).fit(tf)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print('Topic %d:' % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 15
#display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)


# based on the top words from each topic, creating a title for each one (manual)
topic_ids = []
for i in range(0, no_topics):
    topic_ids.append(f'topic{i}')

topics = pd.DataFrame({'topic_id': topic_ids,
    'topic_name': ['animal virus',
                   'animal tesing',
                   'outbreak \nmonitoring',
                   'symptoms \nanalyses',
                   'vaccine \ndevelopment',
                   'patient affects',
                   'cellular studies',
                   'genomic studies',
                   'comparison to \nother outbreaks',
                   'disease/drug \ninteraction']})


###############################################################################
# Assign a topic to every research paper
###############################################################################

# collapse the different topic weights for every word into a single dataframe
lda_df = pd.DataFrame({'words': tf_feature_names})
for i in range(0, len(lda.components_)):
    colname = f'topic{i}'
    lda_df[colname] = lda.components_[i].tolist()

# get the summed weights for every topic, for every research paper
t1_start = process_time()
topic_amounts = pd.DataFrame()
for i in range(0, len(stemmed_words)):
    topic0_amount = 0
    df = pd.DataFrame({'words': stemmed_words[i]})
    df_lda = df.merge(lda_df, on = 'words', how = 'inner')
    amounts = df_lda.drop(['words'], axis = 1).sum(axis = 0).reset_index()
    amounts['paper'] = i
    topic_amounts = topic_amounts.append(amounts)
t1_end = process_time()
round(t1_end - t1_start, 2)

idx = topic_amounts.groupby(['paper'])[0] \
    .transform(max) == topic_amounts[0]
paper_topics = topic_amounts[idx]
paper_topics.columns = ['topic_id', 'lda_value', 'paper_loc']

# group paper counts by topic and visualize
topic_count = paper_topics.groupby('topic_id')['paper_loc'].count().reset_index()
topic_count = topic_count.merge(topics, on = 'topic_id', how = 'inner')

import matplotlib.pyplot as plt
def topic_viz():
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    paper_count = list(topic_count['paper_loc'])
    topics = tuple(list(topic_count['topic_name']))
    x_pos = np.arange(len(topics))
    ax.bar(x_pos, paper_count)
    plt.xticks(x_pos, topics, rotation = 45)
    plt.title('Research Paper Categorization by Model Topic')

topic_viz()



###############################################################################
# Exploratory Analysis
###############################################################################

def get_most_freq_words(str, n=None):
    vect = CountVectorizer().fit(str)
    bag_of_words = vect.transform(str)
    sum_words = bag_of_words.sum(axis=0)
    freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=True)
    return freq[:n]

get_most_freq_words([word for word in stemmed_papers for word in word] , 50)

df = pd.DataFrame({'abstract': papers_text1, 'token_stemmed': stemmed_papers})


# build a dictionary where for each tweet, each word has its own id.

# create a single list of all stemmed words from the papers
flat_words =[]
for i in stemmed_papers:
    flat_words += i

# Creating dictionary for the word frequency table
frequency_table = dict()
for wd in flat_words:
    if wd in frequency_table:
        frequency_table[wd] += 1
    else:
        frequency_table[wd] = 1

# build the corpus i.e. vectors with the number of occurence of each word per tweet
corpus_corpus = [frequency_table.doc2bow(word) for word in stemmed_papers]

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel

tweets_dictionary = Dictionary(stemmed_papers)


# compute coherence
tweets_coherence = []
for nb_topics in range(1,36):
    lda = LdaModel(tweets_corpus,
                   num_topics = nb_topics,
                   id2word = tweets_dictionary,
                   passes=10)
    cohm = CoherenceModel(model=lda,
                          corpus=tweets_corpus,
                          dictionary=tweets_dictionary,
                          coherence='u_mass')
    coh = cohm.get_coherence()
    tweets_coherence.append(coh)

# visualize coherence
plt.figure(figsize=(10,5))
plt.plot(range(1,36),tweets_coherence)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score");




# Close the temporary directory
import shutil
shutil.rmtree(temp_dir.name)
