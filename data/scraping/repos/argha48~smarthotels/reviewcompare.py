import nltk
import pandas as pd
import numpy as np
import re
import codecs
import json

import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec import LineSentence
from gensim.models.ldamulticore import LdaMulticore
# spacy for lemmatization
import spacy
# NLTK for text cleaning
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords, names
from nltk.tokenize import RegexpTokenizer
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# TextBlob package for translation and spelling correction
from textblob import TextBlob

nlp = spacy.load('en')
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)



### Step 1: Initialization.

# Set up tokenizer.
tokenizer = RegexpTokenizer(r'\w+')
# Set up stop words.
en_stop = set(stopwords.words('english'))
# Set up stemmer.
p_stemmer = PorterStemmer()
# For sentiment analysis
senti = SentimentIntensityAnalyzer()


# Load saved models.
# I trained three LDA models from step 5, they have slightly different topics.
# I trained three models because there is randomness in LDA training.
ldamodel1 = pickle.load(open("PATH/model/lda_1.pickle", "rb"))
dict1 = pickle.load(open("PATH/model/dictionary_1.pickle", "rb"))
corpus1 = pickle.load(open("PATH/model/corpus_1.pickle", "rb"))

# # Topic dictionary
# lda_topics={
#                 0:'Parking Service',
#                 1:'Sleep Quality',
#                 2:'WiFi',
#                 3:'Online Service',
#                 4:'Overall Experience',
#                 5:'Value for Money',
#                 6:'Swimming Pool/Spa',
#                 7:'Front Desk Service',
#                 8:'Food', ''
#                 9:'Cleanliness',
#                 10:'Surroundings',
#                 11:'Distance from Transportation',
#                 12:'Booking Experience',
#                 13:'Hotel Staff',
#                 14:'Early Check-in/Late Check-out',
#                 15:'Noise'
#             }

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

# helper functions for text preprocessing & LDA modeling:

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """

    return token.is_punct or token.is_space or token.like_num or token.is_digit

def line_review(filename):
    """
    generator function to read in reviews from Pandas Series
    and un-escape the original line breaks in the text
    """

    #with codecs.open(filename, encoding='utf_8') as f:
    for review in filename:
        yield review.replace('\\n', '\n')

def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """

    for parsed_review in nlp.pipe(line_review(filename), batch_size=10000, n_threads=10):
        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    # load finished dictionary from disk
    trigram_dictionary = Dictionary.load('./models2/trigram_dict_all.dict')

    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)


def LDA_Review(review_df,min_topic_freq=0):
    """
    Takes Pandas series as input,
    consisting of one review as
    text string per row
    """
    from tqdm import tqdm
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """
    text = review_df['FullReview']
    # parse the review text with spaCy
    with codecs.open('./uni_temporary.txt', 'w', encoding='utf_8') as f:
        for sentence in tqdm(lemmatized_sentence_corpus(text)):
            # print(sentence)
            f.write(sentence + '\n')
    f.close()
    # load and apply the first-order and secord-order phrase models
    bigram_model = Phrases.load('./models2/bigram_model.txt')
    trigram_model = Phrases.load('./models2/trigram_model.txt')

    unigram_review = LineSentence('./uni_temporary.txt')
    bigram_review = bigram_model[unigram_review]
    trigram_review = trigram_model[bigram_review]
    # remove any remaining stopwords
    trigram_review = [term for term in trigram_review
                        if term not in spacy.lang.en.stop_words.STOP_WORDS]
    with codecs.open('./tri_temporary.txt', 'w', encoding='utf_8') as ftri:
        for sentence in trigram_review:
            sentence = u' '.join(sentence)
            ftri.write(sentence + '\n')
    ftri.close()

    trigram_dictionary = Dictionary.load('./models2/trigram_dict_all.dict')
    lda = LdaMulticore.load('./models2/lda_model')
    trigram_review = LineSentence('./tri_temporary.txt')
    # create a bag-of-words representation
    review_bow = trigram_dictionary.doc2bow(trigram_review)
    # create an LDA representation
    review_lda = lda.get_document_topics(review_bow)
    review_lda = sorted(review_lda, key=itemgetter(1),reverse=True)
    for topic_number, freq in review_lda:
        if freq < min_topic_freq:
            break
        # print the most highly related topic names and frequencies
        print('{:25} {}'.format(lda_topics[topic_number], round(freq, 3)))


    ### Step 2: Generate the contents of the doctors' snapshots.

    counter = 0
    # The temporary string that stores all of the review highlights in each round of the for loop below.
    big_str = []
    # For every doctor, find two things:
    #     1. The most mentioned FIVE topics in their reviews.
    #         1.1 The sentiments of these topics.
    #     2. The 3 most positive sentences and the 3 most negative sentences.
    #         2.1 Rank all sentences according to sentiment analysis.
    # I do NOT keep info about individual reviews. All sentences are stored in a
    # long list regardless of whether they are from the same reviews or not!
    ###########################################################################
    # Build sentence dataframe for the current doctor.
    ###########################################################################
    this_hotel = pd.DataFrame(columns = ["HotelName","Sentence", "Sentiment_neg",
                            "Sentiment_neu","Sentiment_pos",
                            "Sentiment_compound", "topic_1", "topic1_score",
                            "topic_2", "topic2_score"])
    sent_count = 0

    # For every review sentence
    for sentence in unigram_review:
        # Assess sentiment.
        sentiments = senti.polarity_scores(sentence)
        sentiment_neg = sentiments["neg"]
        sentiment_neu = sentiments["neu"]
        sentiment_pos = sentiments["pos"]
        sentiment_compound = sentiments["compound"]
        # Assign topic.
        # Default topic to -1.
        this_topic = -1
        # Preprocess sentence.
        sent_tokens = tokenizer.tokenize(str(sentence).lower())
        cleaned_sent = [p_stemmer.stem(i) for i in sent_tokens]
        # Evaluate for topic.
        sent_topics = []
        for mod_id in range(0, mod_num):
            model = ldamodel[mod_id]
            dicti = dictionary[mod_id]
            lda_score = model[dicti.doc2bow(cleaned_sent)]
            for item in lda_score:
                sent_topics.append((mod_id, item[0], item[1]))
        sent_topics =  sorted(sent_topics, key=lambda x: x[2], reverse=True)
        # Assign the most relevant topic to a sentence only if the topic is more than 70% dominant.
        if sent_topics[0][2] > 0.7:
            this_topic = topics_matrix[sent_topics[0][0]][sent_topics[0][1]]

        # Add procressed sentence and its meta information to the sentence dataframe.
        this_doc.loc[sent_count] = [sentence, sentiment, this_topic, sent_topics[0][2]]
        sent_count += 1

    ###########################################################################
    # Compiling results for a hotel.
    ###########################################################################
    # Review highlights.
    # Save the most positive and negative sentiments.
    this_doc2 = this_doc.sort_values(["sentiment"], ascending=[0]).reset_index(drop=True)
    this_doc2 = this_doc2.loc[this_doc2["topic"] != -1].reset_index(drop=True)
    this_doc2 = this_doc2.loc[this_doc2["topic_score"] > 0.5].reset_index(drop=True)
    sent_count_2 = len(this_doc2)
    composite = "NONE"
    # Save the most polarizing sentiments only if there are at least 6 sentences.
    if sent_count_2 > 5:
        sent1 = sent2 = sent3 = sent4 = sent5 = sent6 = ""
        # Only keep positive sentiment if its score is above 0.4 (within [-1, 1]).
        if this_doc2.loc[0]["sentiment"] > 0.4:
            sent1 = this_doc2.loc[0]["sentence"]
        if this_doc2.loc[1]["sentiment"] > 0.4:
            sent2 = this_doc2.loc[1]["sentence"]
        if this_doc2.loc[2]["sentiment"] > 0.4:
            sent3 = this_doc2.loc[2]["sentence"]
        # Only keep positive sentiment if its score is below -0.2 (within [-1, 1]).
        if this_doc2.loc[sent_count_2-1]["sentiment"] < -0.2:
            sent4 = this_doc2.loc[sent_count_2-1]["sentence"]
        if this_doc2.loc[sent_count_2-2]["sentiment"] < -0.2:
            sent5 = this_doc2.loc[sent_count_2-2]["sentence"]
        if this_doc2.loc[sent_count_2-3]["sentiment"] < -0.2:
            sent6 = this_doc2.loc[sent_count_2-3]["sentence"]
        composite = sent1 + "SSEEPP" + sent2 + "SSEEPP" + sent3 + "SSEEPP" + sent4 + "SSEEPP" + sent5 + "SSEEPP" + sent6 + "SSEEPP" + str(sent_count)
    # Add review highlights to the doctor dataframe.
    doctor_info.set_value(doctor_id, "summary", composite)

    # Top topics and their ratings.
    # Ratings are the percent positive sentences belonging to a topic.
    doc_topics = [ [ 0 for i in range(2) ] for j in range(topic_num) ]  # [total count, count positive]
    for index2 in range(0, len(this_doc2)):
        topic_index = this_doc2.loc[index2]["topic"]
        if topic_index != -1:
            doc_topics[topic_index][0] += 1
            topic_sentiment = this_doc2.loc[index2]["sentiment"]
            # A topic sentence if positive if its sentiment is bigger than 0.1.
            if topic_sentiment > 0.1:
                doc_topics[topic_index][1] += 1
    # Do not display dentist stuff for non-dentist
    if not is_dentist:
        doc_topics[3][0] = 0
    # Do not output "positive comment" as a topic. It is non-informative.
    doc_topics[0][0] = 0

    # Putting the results into a format to be sparsed by the webapp.
    doc_topic_tuples = []
    for index3, item in enumerate(doc_topics):
        doc_topic_tuples.append((index3, item[0], item[1]))
    doc_topic_tuples =  sorted(doc_topic_tuples, key=lambda x: x[1], reverse=True)
    for index4 in range(0, 5):
        if doc_topic_tuples[index4][1] >= 10:
            topic_name = topics[doc_topic_tuples[index4][0]][0]
            percent_positive = str(int(doc_topic_tuples[index4][2]/doc_topic_tuples[index4][1] * 100))
            composite = topic_name + "SSEEPP" + percent_positive + "SSEEPP" + str(doc_topic_tuples[index4][1])
            doctor_info.set_value(doctor_id, "percent{0}".format(str(index4+1)), composite)

            print(topic_name, "XXXXXX", doctor_info.loc[doctor_id]["specialty"])
            big_str.append(topic_name + "XXXXXX" + str(doctor_info.loc[doctor_id]["specialty"]))
        else:
            doctor_info.set_value(doctor_id, "percent{0}".format(str(index4+1)), "NONE")

    # Print progress.
    print(counter/5088)
    counter += doctor_review_count
    del this_doc
    del this_doc2

# Save the updated doctor dataframe containing snapshot information.
doctor_info.to_csv("PATH/result/all_doctors.csv")
