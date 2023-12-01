import re
import numpy as np
import pandas as pd
from pprint import pprint
import streamlit as st
# from Engagement import *
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# from app import read_csv
# spacy for lemmatization
import spacy
# import os
# Plotting tools
import pyLDAvis
# from pyLDAvis import gensim
import pyLDAvis.gensim_models
# import en_core_web_sm
# %matplotlib inline

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
# spacy.load('en_core_web_sm')
warnings.filterwarnings("ignore", category=DeprecationWarning)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('words')
# nlp = spacy.load('en', disable=['parser', 'ner'])
negation = ["no", "nor", "not", "don", "don't", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn",
            "doesn't",
            "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "mightn", "mightn't", "mustn",
            "mustn't",
            "needn", "needn't", "shan", "shan't", "shouldn", "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won',
            "won't", 'wouldn', "wouldn't"]
stop_words = set(stopwords.words('english'))
stop_list_2 = ['dm', 'get', 'always', 'have', 'para', 'thought', 'fd', 'get', 'have', 'keep', 'lol', 'email', 'do', 'back', 'cg', 'sl', 'eye', 'from', 'subject', 're', 'edu', 'use', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
     "you're", "you've", "you'll", "you'd", 'your',
     'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
     "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
     'that', "that'll",
     'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
     'does', 'did',
     'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'do', 'reply', 'site', 'fb', 'hour', 'inbox', 's', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
     'with', 'about',
     'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
     'in', 'out', 'go', 'sr','get', 'canva', 'already', 'be', 'take', 'do', 's',
     'on', 'off', 'over', 'under','not','try','look','try','week','write','order','again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'all',
     'any', 'alamat', 'picsart',
     'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too',
     'very', 's', 't', 'can', 'will', 'just', 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'rt',
     'rt', 'qt', 'for',
     'the', 'with', 'in', 'of', 'and', 'its', 'it', 'this', 'i', 'have', 'has', 'would', 'could', 'you', 'a', 'an',
     'be', 'am', 'can', 'edushopper', 'will', 'to', 'on', 'is', 'by', 'ive', 'im', 'your', 'we', 'are', 'at', 'as',
     'any', 'ebay', 'thank', 'hello', 'know',
     'need', 'want', 'look', 'hi', 'sorry', 'http', 'body', 'dear', 'hello', 'hi', 'thanks', 'sir', 'tomorrow', 'sent',
     'send', 'see', 'there', 'welcome', 'what', 'well', 'us', 'do', 'go','be', 'still', 's', 'cv', 'lol',
     'be', 'que', 'have', 'sure', 'theshow', 'also', 'esta']

stoplist = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your',
            'yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',
            "it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",
            'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did',
            'doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about',
            'against','between','into','through','during','before','after','above','below','to','from','up','down','in','out',
            'on','off','over','under','again','further','then','once','here','there','when','where','why','all','any',
            'both','each','few','more','most','other','some','such','only','own','same','so','than','too',
            'very','s','t','can','will','just','should',"should've",'now','d','ll','m','o','re','ve','y','rt','rt','qt','for',
            'the','with','in','of','and','its','it','this','i','have','has','would','could','you','a','an',
            'be','am','can','edushopper','will','to','on','is','by','ive','im','your','we','are','at','as','any','ebay','thank','hello','know',
            'need','want','look','hi','sorry','http','body','dear','hello','hi','thanks','sir','tomorrow','sent','send','see','there','welcome','what','well','us']


stop_words.update(set(stoplist))
stop_words.update(set(negation))
# Reading the dataset
# def reading_csv():
#     data = pd.read_csv()
#     return data

"Cleaning the text "

@st.cache(suppress_st_warning=True)
def preprocessing_text_column(df):
    # Convert to list
    data = df.text.values.tolist()
    data = [x for x in data if str(x) != 'nan']
    data = [re.sub(r'\d', '', sent) for sent in data]  # removing digits
    data = [re.sub(r"(?:\@|https?\://)\S+", "", sent) for sent in data]  # removing mentions and urls
    data = [re.sub(r"\b[a-zA-Z]\b", "", sent) for sent in data] # Removing single letter words
    data = [re.sub(r'/\s+\S{1,2}(?!\S)|(?<!\S)\S{1,2}\s+/', '',sent) for sent in data]
    data = [re.sub('[0-9]+', '', sent) for sent in data]
    # data = [REPLACE_BY_SPACE_RE.sub(" ", sent) for sent in data]  # replace REPLACE_BY_SPACE_RE symbols by space in text
    # data = [BAD_SYMBOLS_RE.sub(" ", sent) for sent in data]  # delete symbols which are in BAD_SYMBOLS_RE from text
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]  # Remove Emails
    data = [re.sub('\s+', ' ', sent) for sent in data]  # Remove new line characters
    data = [re.sub("\'", "", sent) for sent in data]  # Remove distracting single quotes

    return data

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# Creating Bigram and Trigram Models
@st.cache(suppress_st_warning=True)
def data_words(words):
    data_words = list(sent_to_words(words))
    return data_words


# Build the bigram and trigram models

# Faster way to get a sentence clubbed as a trigram/bigram
@st.cache(suppress_st_warning=True)
def bigram_mod(bigram):
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def trigram_mod(trigram):
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return trigram_mod

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts, user_input):
    stop_words.update(set(stoplist))
    stop_words.update(set(stop_list_2))
    stop_words.update(set(negation))
    stop_words.update(user_input)
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    # nlp = en_core_web_sm.load()
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def to_dataframe(lda_model=None):
    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False)
                       for j, (topic, wt) in enumerate(topics) if j < 10]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0, inplace=True)

    return df_top3words

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def pyldavis(lda_model=None, corpus=None):
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    return vis


# Building the Topic Model

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model(corpus, id2word):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=5,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=500,
                                                passes=10,
                                                iterations=200,
                                                alpha= 0.1,
                                                eta='auto',
                                                eval_every=1,
                                                per_word_topics=True)

    return lda_model

from gensim.models.ldamodel import LdaModel
def convertldaGenToldaMallet(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

@st.cache(allow_output_mutation=True)
def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
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
@st.cache(allow_output_mutation=True)
def representative_sentence(df_topic_sents_keywords= None):
    pd.options.display.max_colwidth = 100

    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    return sent_topics_sorteddf_mallet
@st.cache(allow_output_mutation=True)
def relevency(topic_data , lambd = None):
    all_topics = {}
    num_terms = 6  # Adjust number of words to represent each topic
    lambd = lambd  # Adjust this accordingly based on tuning above
    for i in range(1, 6):  # Adjust this to reflect number of topics chosen for final LDA model
        topic = topic_data.topic_info[topic_data.topic_info.Category == 'Topic' + str(i)].copy()
        topic['relevance'] = topic['loglift'] * (1 - lambd) + topic['logprob'] * lambd
        all_topics['Topic ' + str(i)] = topic.sort_values(by='relevance', ascending=False).Term[:num_terms].values
    final_data = pd.DataFrame(all_topics).T
    return final_data

@st.cache(suppress_st_warning= True)
def result(file, user_input):
    data = preprocessing_text_column(file)
    words = data_words(data)
    bigram = gensim.models.Phrases(words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_words_nostops = remove_stopwords(words, user_input)
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = model(corpus, id2word)
    # mallet_path = '/Users/shyam.muralidharan/Desktop/Engagement/mallet-2.0.8/bin/mallet'  # update this path
    # ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=5, id2word=id2word)

    df_topic_sents_keywords = format_topics_sentences(ldamodel= lda_model, corpus=corpus, texts=data_lemmatized)

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
                                               coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)
    # lda_mal = convertldaGenToldaMallet(ldamallet)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    sent_topics_sorteddf_mallet = representative_sentence(df_topic_sents_keywords)
    vis = pyldavis(lda_model, corpus)
    topic_final = relevency(lambd = 0.11, topic_data=vis)
    pyLDAvis.save_html(vis, 'LDA_Visualization.html')
    return topic_final, vis, df_dominant_topic, sent_topics_sorteddf_mallet
