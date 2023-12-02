import pandas as pd
import os
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim import models
from gensim.models import CoherenceModel
import spacy
import numpy as np
import sys
sys.path.append('./')
from utilities import clean_str, del_http_user_tokenize

part = sys.argv[1]
sentiment = sys.argv[2]

if not os.path.exists("../data/topic_modeling"):
    os.makedirs("../data/topic_modeling")

vaccine_wo_dist_path = "../data/vaccine_text_wo_distribution_real_date"
LDA_path = "../data/topic_modeling"

stop_words = stopwords.words('english')
keep = ['who', 'no', 'nor', 'not', "don't", "aren't", "couldn't", "didn't", "doesn't", "hadn't", "hasn't",
        "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't", "weren't",
        "won't", "wouldn't",
        'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
        'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
stop_words = [e for e in stop_words if e not in keep]
stop_words.extend(['pron', 'amp', '-PRON-','datum'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    mallet_path = '/projects/p31384/twitter_vaccine/pycharm_code/mallet-2.0.8/bin/mallet'
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# initialize spacy for lemmatization
nlp = spacy.load('en', disable=['parser', 'ner'])

vaccine_wo_dist_df = \
    pd.read_csv(os.path.join(vaccine_wo_dist_path,
                             'vaccine_text_wo_distribution_real_date_part%s.csv' % (part)),
                usecols = [1,2,3,4],
                dtype = {'text':str, 'tweet_id':str, 'user_id': str, 'sentiment':str})

if sentiment != 'all':
    vaccine_wo_dist_df = vaccine_wo_dist_df[vaccine_wo_dist_df['sentiment'] == sentiment]

vaccine_wo_dist_df = vaccine_wo_dist_df.reset_index().drop_duplicates()

# Remove text, remove http, punctuations and lower case
vaccine_wo_dist_df['text_processed'] = vaccine_wo_dist_df['text'].apply(del_http_user_tokenize)
vaccine_wo_dist_df['text_processed'] = vaccine_wo_dist_df['text_processed'].apply(clean_str)

# lemmatization
text_nlp = vaccine_wo_dist_df['text_processed'].apply(nlp)
text_lst = []
for sen in text_nlp:
    text_lst.append([token.lemma_ for token in sen])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(text_lst, threshold=10)
trigram = gensim.models.Phrases(bigram[text_lst], threshold=10)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# remove stop words
data_words_nostops = remove_stopwords(text_lst)

# Form Bigrams
data_words_bigrams = make_trigrams(data_words_nostops)

# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)

# Create Corpus
texts = data_words_bigrams

# BOW
corpus = [id2word.doc2bow(doc) for doc in texts]

# find the optimal number of topic according to coherence value
n_topic_lst = [2,7,12,17,22]
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words_bigrams,
                                                        start=2, limit=23, step=5)
# optimal model according to best number of topics
optimal_model = model_list[np.argmax(coherence_values)]
num_topics = n_topic_lst[np.argmax(coherence_values)]

df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=vaccine_wo_dist_df['text'])
df_topic_sents_keywords = df_topic_sents_keywords.reset_index()
df_topic_sents_keywords.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts().reset_index()
topic_counts.columns = ['Dominant_Topic', 'Num_Documents']

# Percentage of Documents for Each Topic
topic_counts['Perc_Documents'] = pd.DataFrame(round(topic_counts['Num_Documents'] / topic_counts['Num_Documents'].sum(), 4))

# Concatenate Column wise
df_dominant_topics = pd.merge(df_topic_sents_keywords, topic_counts, on = 'Dominant_Topic')
df_out = df_dominant_topics.sort_values(['Num_Documents','Topic_Perc_Contrib'], ascending=False)

df_out.to_csv(os.path.join(LDA_path,
                           'part%s_ntopics_%s_%s.csv' %(part, str(num_topics), sentiment)))
