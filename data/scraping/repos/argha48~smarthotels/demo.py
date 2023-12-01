#############################################################
#	Copyright (C) 2019  Argha Mondal
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#############################################################
import nltk
import pandas as pd
import numpy as np
import re
import codecs
import json

import re
import pickle
# # Gensim
# import gensim
# import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from gensim.models import CoherenceModel
# from gensim.models import Phrases
# from gensim.corpora import Dictionary, MmCorpus
# from gensim.models.word2vec import LineSentence
# from gensim.models.ldamulticore import LdaMulticore
# # spacy for lemmatization
# import spacy
# # NLTK for text cleaning
# from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
# from nltk.classify import NaiveBayesClassifier
# from nltk.corpus import stopwords, names
# from nltk.tokenize import RegexpTokenizer
# from nltk import tokenize
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')
# # TextBlob package for translation and spelling correction
# from textblob import TextBlob
#
# nlp = spacy.load('en')
# # Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


alldata = pd.read_json('./src/minidemo.json',orient='columns',encoding='utf-8')
star_score_df = pd.read_json('./src/hotel_star_scores.json',encoding='utf-8')
mallet_lda_topics={
                0:'Hotel Staff',
                1:'Accessibility',
                2:'Food',
                3:'Overall Experience',
                4:'Noise',
                5:'Value for Money',
                6:'Room Amenities',
                7:'Location in the city',
                8:'Overall Experience',
                9:'Cleanliness',
                10:'Early Check-in/Late Check-out',
                11:'Health and Wellness Amenities',
                12:'Booking Experience',
                13:'Sleep Quality',
                14:'Parking Facility'
            }

#
# def get_text(rev):
#     if pd.DataFrame(rev).empty:
#         return ''
#     else:
#         return rev[0] if str(rev)!='nan' else ''
#
# def review_cleaned(review_df):
#     df = review_df[['HotelName','PositiveReview','NegativeReview','StayDate']].copy()#.applymap(get_text)
#     df['FullReview'] = [pos+' '+neg for pos,neg in zip(df['PositiveReview'],df['NegativeReview'])]
# #     df['StayDate'] = df['StayDate'].apply(lambda x: x.replace('\n','')).apply(lambda x: x.replace('Stayed in ',''))
#     return df
#
# def review_to_sentence(df):
#     all_sentences = []
#     from nltk.tokenize import sent_tokenize
#     import pandas as pd
#     allreview = df['FullReview']
#     for areview in allreview:
#         all_sentences.extend(sent_tokenize(areview))
#     tokensentence = pd.DataFrame(data=all_sentences,columns=['TokenSentence'])
#     return tokensentence
#
# def sentence_sentiment(text):
#     from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#     analyzer = SentimentIntensityAnalyzer()
#     compound_sentiment = analyzer.polarity_scores(text)['compound']
#     return compound_sentiment
#
# def token_to_sentiment(df):
#     df['CompoundSentiment'] = df['TokenSentence'].apply(sentence_sentiment)
#     return df
#
# # helper functions for text preprocessing & LDA modeling:
#
# def punct_space(token):
#     """
#     helper function to eliminate tokens
#     that are pure punctuation or whitespace
#     """
#
#     return token.is_punct or token.is_space or token.like_num or token.is_digit
#
# def line_review(filename):
#     """
#     generator function to read in reviews from Pandas Series
#     and un-escape the original line breaks in the text
#     """
#
#     #with codecs.open(filename, encoding='utf_8') as f:
#     for review in filename:
#         yield review.replace('\\n', '\n')
#
# def lemmatized_sentence_corpus(filename):
#     """
#     generator function to use spaCy to parse reviews,
#     lemmatize the text, and yield sentences
#     """
#
#     for parsed_review in nlp.pipe(line_review(filename), batch_size=10000, n_threads=10):
#         for sent in parsed_review.sents:
#             yield u' '.join([token.lemma_ for token in sent
#                              if not punct_space(token)])
#
# def trigram_bow_generator(filepath):
#     """
#     generator function to read reviews from a file
#     and yield a bag-of-words representation
#     """
#     # load finished dictionary from disk
#     trigram_dictionary = Dictionary.load('./src/saved_model/models2/trigram_dict_all.dict')
#
#     for review in LineSentence(filepath):
#         yield trigram_dictionary.doc2bow(review)
#
# def nCPU():
#     import multiprocessing
#     N = multiprocessing.cpu_count()-1
#     return N
#
# def topic_extractor(df, min_topic_freq=0.10):
#     from tqdm import tqdm
#     from operator import itemgetter
#     ncpu = nCPU()
#     dfc=df.copy()
#     text = dfc['TokenSentence'].copy()
#     trigram_dictionary = Dictionary.load('./src/saved_model/models2/trigram_dict_all.dict')
#     lda = LdaMulticore.load('./src/saved_model/models2/mallet_lda_model')
#     # trigram_review = LineSentence('./tri_temporary.txt')
#     bigram_model = Phrases.load('./src/saved_model/models2/bigram_model.txt')
#     trigram_model = Phrases.load('./src/saved_model/models2/trigram_model.txt')
#     topic_list = []
#     trigram_list = []
#     freq_list = []
#     # parse the review text with spaCy
#     for parsed_review in tqdm(nlp.pipe(line_review(text),
#                                     batch_size=10000, n_threads=ncpu)):
#         # lemmatize the text, removing punctuation and whitespace
#         unigram_review = [token.lemma_ for token in parsed_review
#                             if not punct_space(token)]
#         # apply the first-order and second-order phrase models
#         bigram_review = bigram_model[unigram_review]
#         trigram_review = trigram_model[bigram_review]
#
#         common_terms = ['-PRON-','hotel'] #'service',
#         # remove any remaining stopwords
#         trigram_review = [term for term in trigram_review
#                             if term not in spacy.lang.en.stop_words.STOP_WORDS]
#         trigram_review = [term for term in trigram_review
#                             if term not in common_terms]
#         if len(trigram_review)==0:
#             topic_number=-1
#             freq = 0.0
#             tri = str([])
#         else:
#             # create a bag-of-words representation
#             review_bow = trigram_dictionary.doc2bow(trigram_review)
#             # create an LDA representation
#             review_lda = lda.get_document_topics(review_bow)
#             # print the most highly related topic name and frequency
#             review_lda = sorted(review_lda, key=itemgetter(1),reverse=True)[0]
#             topic_number = review_lda[0]
#             freq = review_lda[1]
#             if freq < min_topic_freq:
#                 topic_number=-1
#                 freq = 0.0
#
#         topic_list.append(topic_number)
#         freq_list.append(round(freq,2))
#         trigram_list.append(trigram_review)
#     dfc['Topic']=topic_list
#     dfc['TopicFrequency']=freq_list
#     dfc['Trigram']=trigram_list
#     return dfc
#
# def topic_scorer(df):
#     xdf = pd.get_dummies(df,prefix='Topic',
#                      prefix_sep='_', dummy_na=False,
#                      columns=['Topic'])
#     topics = ['Topic_0', 'Topic_1', 'Topic_2','Topic_3',
#               'Topic_4', 'Topic_5', 'Topic_6', 'Topic_7',
#               'Topic_8','Topic_9', 'Topic_10', 'Topic_11',
#               'Topic_12', 'Topic_13','Topic_14']
#     topic_dict = {}
#     for atopic in topics:
#         if atopic in xdf.columns.values:
#             xdf[atopic] = xdf[atopic] * xdf['CompoundSentiment']
#             m = np.mean(list(filter(lambda a: a != 0, xdf[atopic])))
#             topic_dict[mallet_lda_topics[int(atopic.replace('Topic_',''))]] = round(m,2)
#         else:
#             topic_dict[mallet_lda_topics[int(atopic.replace('Topic_',''))]] = 'No information available'
#     return topic_dict
#
# def demo_(hotel_name):
#     new_doc = alldata[alldata['HotelName']==hotel_name]
#     # print(new_doc.shape)
#     text = review_cleaned(new_doc)
#     tokensentence = review_to_sentence(text)
#     sentencesentiment = token_to_sentiment(tokensentence)
#     topicdf = topic_extractor(sentencesentiment)
#     topic_dict = topic_scorer(topicdf)
#     return [(key,int(100*topic_dict[key])) for key in topic_dict]

def score_compare(hotel_name, hotel_star):
    topics = ['Topic_0', 'Topic_1', 'Topic_2','Topic_3',
              'Topic_4', 'Topic_5', 'Topic_6', 'Topic_7',
              'Topic_8','Topic_9', 'Topic_10', 'Topic_11',
              'Topic_12', 'Topic_13','Topic_14']

    df_myhotel = star_score_df[star_score_df['HotelName']==hotel_name]
    df = star_score_df[star_score_df['HotelStar']==str(hotel_star)]
    topic_dict = {}
    for i in range(len(topics)):
        atopic = topics[i]
        if atopic in df_myhotel.columns.values:
            if df_myhotel[atopic].values >= -10.:
                # print(str(df_myhotel[atopic]))
                myscore = int(100.*float(df_myhotel[atopic]))
                otherscore = pd.Series(df[atopic]).dropna().astype('int')
                otherscore = otherscore.dropna()
                # print(otherscore)
                otherscore = 100*otherscore
                othermean = int(100.*np.mean(otherscore))
                if myscore > othermean:
                    topic_dict[mallet_lda_topics[int(atopic.replace('Topic_',''))]] = [myscore, othermean, 'Good']
                else:
                    topic_dict[mallet_lda_topics[int(atopic.replace('Topic_',''))]] = [myscore, othermean, 'Bad']
    topic_dict['Hotel_info'] = [hotel_name, str(int(hotel_star)+1), 'info']
    return [(str(key),topic_dict[key][0],topic_dict[key][1]) for key in topic_dict]



def make_plot(title,data, hist, edges, x):
    import numpy as np
    import scipy.special
    from scipy import stats
    from bokeh.layouts import gridplot
    from bokeh.plotting import figure, show, output_file
    kernel = stats.gaussian_kde(data)
    pdf = kernel(x)
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5, legend="Histogram")
    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend="PDF")
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'Sentiment Score'
    p.yaxis.axis_label = 'Frequency'
    p.grid.grid_line_color="white"
    return p


def compare_plot(hotel_name, hotel_star, atopic):
    from bokeh.embed import components
    from bokeh.models import Span
    # Create the main plot
    df_myhotel = star_score_df[star_score_df['HotelName']==hotel_name]
    df = star_score_df[star_score_df['HotelStar']==str(hotel_star)]
    myscore = 100.*float(df_myhotel[atopic])
    otherscore = 100.*pd.Series(df[atopic].values).dropna()
    hist, edges = np.histogram(otherscore, density=True, bins=50)
    x = np.linspace(-100,100,5000)
    p = make_plot(mallet_lda_topics[int(atopic.replace('Topic_',''))],otherscore, hist, edges, x)

    othermean = int(100.*np.mean(otherscore))
    if myscore > othermean:
        # Vertical line
        vcolor = 'red'
        vline = Span(location=myscore, dimension='height', line_color=vcolor, line_width=5)
        p.add_layout(vline)
        p.title.text_font_size = '18pt'
        p.xaxis.axis_label_text_font_size = "18pt"
        p.yaxis.axis_label_text_font_size = "18pt"
    else:
        # Vertical line
        vcolor = 'red'
        vline = Span(location=myscore, dimension='height', line_color=vcolor, line_width=5)
        p.add_layout(vline)
        p.title.text_font_size = '18pt'
        p.xaxis.axis_label_text_font_size = "18pt"
        p.yaxis.axis_label_text_font_size = "18pt"
    return p
