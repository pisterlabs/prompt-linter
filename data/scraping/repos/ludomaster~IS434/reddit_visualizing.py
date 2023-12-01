# Used to visualize csv data (plotting)

import datetime as dt
import json
from collections import defaultdict
import math
import praw
import pandas as pd
import numpy as np
import nltk

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pprint import pprint
from IPython import display
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from nltk.stem import PorterStemmer
# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer

#ps = PorterStemmer()

stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    stop_words = stopwords.words('english')
    #customStopWords = ['iâ','one','want','anyone','today','itâ','suicidal','depressed','would','get','make','really','else','even',
       #'ever','know','think','day','much','going','feeling','person','died','everyone','dead','everything','feel','like',
	   #'life','someone','always','still','way','sometimes','things','thoughts','something','every','back','years','cares','good']
    customStopWords = ['like','thinking','killed','things','want','killing','going','good']
    stop_words.extend(customStopWords)
    for token in gensim.utils.simple_preprocess(text):
        #token = [t.lower() for t in token if t.lower() not in stop_words]
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            #result.append(lemmatize_stemming(token))
            result.append(token)
    return result

from nltk.corpus import wordnet as wn
<<<<<<< HEAD
=======
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
>>>>>>> 7c11b4c536ef98246ed147bd57f5a84181e27ff0

	
# Instantiation
def main():
    """
    Initiation docstring
    """
	
    # Change to whatever you want to plot from
    subreddit = "depression"

	#read suicide-related keywords in csv
    #df = pd.read_csv(f"subreddits/{subreddit}/reddit_depression_submissions.csv", 
			  #sep=',',
			  #encoding='latin-1')
    df = pd.concat(map(pd.read_csv, ['subreddits/depression/reddit_depression_submissions.csv', 
         'subreddits/foreveralone/reddit_foreveralone_submissions.csv',
		 'subreddits/offmychest/reddit_offmychest_submissions.csv',
		 'subreddits/singapore/reddit_singapore_submissions.csv',
		 'subreddits/suicidewatch/reddit_suicidewatch_submissions.csv']))
    #print(df)
	
	##############################################################################
    #####1. PLOTTING BAR CHART of overall sentiment analysis of submissions####
    ##############################################################################

    fig, ax = plt.subplots(figsize=(8, 8))

    counts = df.risk.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    plt.title("Sentiment Analysis on Reddit")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Sentiment Categories")	
    #plt.show()
	##############################################################################
    #####2. PLOTTING Negative keyword frequency####
    ##############################################################################
    neg_lines = list(df[df.risk == -1].submission)
    data_text = df[['submission']]
    data_text['index'] = data_text.index
    documents = data_text
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    customStopWords = ['iâ','one','want','anyone','today','itâ','suicidal','depressed','would','get','make','really','else','even',
       'ever','know','think','day','much','going','feeling','person','died','everyone','dead','everything','feel','like',
	   'life','someone','always','still','way','sometimes','things','thoughts','something','every','back','years','killing','killed'
	   'keep']
    stop_words.extend(customStopWords)
    neg_tokens = []
    doc_clean = []
	
    for line in neg_lines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        #toks = [ps.stem(t) for t in toks]
        neg_tokens.extend(toks)
    
    plt.style.use('ggplot')
	
    neg_freq = nltk.FreqDist(neg_tokens)
    neg_freq.most_common(20)
    #print(neg_freq.most_common(20))
    y_val = [x[1] for x in neg_freq.most_common()]
    y_final = []
    for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
        y_final.append(math.log(i + k + z + t))

    x_val = [math.log(i + 1) for i in range(len(y_final))]
    fig = plt.figure(figsize=(10,5))
    
    plt.xlabel("Words (Log)")
    plt.ylabel("Frequency (Log)")
    plt.title("Negative Word Frequency Distribution on Reddit")
    plt.plot(x_val, y_final)
    #plt.show()
	##############################################################################
    #####3. PLOTTING Negative keyword wordcloud####
    ##############################################################################
    neg_words = ' '.join([text for text in neg_tokens])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(neg_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    #plt.show()
	##############################################################################
    #####3. Topic Analysis####
    ##############################################################################
    processed_docs = documents['submission'].map(preprocess)
    print(processed_docs[:10])
    dictionary = gensim.corpora.Dictionary(processed_docs)
    count = 0
    for k, v in dictionary.iteritems():
        #print(k, v)
        count += 1
        if count > 10:
            break
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    #bow_corpus[4310]
    from gensim import corpora, models
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    from pprint import pprint
    for doc in corpus_tfidf:
        pprint(doc)
        break
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
    topics = lda_model.show_topics(formatted=False)
	
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)
	
    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    # Form Bigrams
    #data_words_bigrams = [bigram_mod[doc] for doc in neg_lines]
    #dictionary = corpora.Dictionary(neg_lines)
    #corpus = [dictionary.doc2bow(text) for text in neg_lines]
	
if __name__ == "__main__":
    main()