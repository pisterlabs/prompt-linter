# for data processing
import numpy as np
import itertools

# for nlp
import gensim
from gensim.utils import simple_preprocess
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
import nltk
from nltk import FreqDist
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import joblib


def vader(text):
    sid = SentimentIntensityAnalyzer()

    # words = set(nltk.corpus.words.words())
    # def remove_non_english(text):
    #     return " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or w.isdigit())
    
    # text = remove_non_english(text)

    if text == '':
        return ''

    def tokenize_and_score(text, sentence_wise=True):
        tokenized_sentences = tokenize.sent_tokenize(text)
        print(tokenized_sentences)
        if sentence_wise:
            scores = list(map(lambda x : sid.polarity_scores(x)['compound'], tokenized_sentences))
        else:
            scores = sid.polarity_scores(text)['compound']
        
        return tokenized_sentences, scores

    tokenized_sentences, vader_scores = tokenize_and_score(text)

    return tokenized_sentences, vader_scores

def lda(tokenized_text, lda_model, dictionary, removed_words, tfidf):
    #POS tagging first to utilize sentences
    stemmer = SnowballStemmer('english')

    def pos_tagging(tokenized_text):
        list_of_preprocessed_sentences = list(filter(lambda x: x != [], [preprocess(sentence) for sentence in tokenized_text]))
        return list(itertools.chain.from_iterable(list_of_preprocessed_sentences)), list_of_preprocessed_sentences 
        # return concatenated document and sentences for later

    def preprocess(sentence):
        '''
        0. Convert to lower case
        1. Only allow nouns. 
        2. Remove stopwords
        3. Stem words
        '''
        pos_tag_tuples = nltk.pos_tag(nltk.word_tokenize(sentence.lower())) # lower case to avoid weird nouns
        
        nouns = []
        for pos_tag_tuple in pos_tag_tuples:
            if pos_tag_tuple[1] in [
                'NN', 'NNS', 'NNP', 'NNPS', # nouns
                'JJ', 'JJR', 'JJS', # adjectives
    #             'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' # verbs
            ]:
                # keep the word
                # but only if it is not a stop word
                if pos_tag_tuple[0] not in stopwords.words('english'):
                    nouns.append(stemmer.stem(pos_tag_tuple[0]))
        
        return nouns

    _, prepared_sentences = pos_tagging(tokenized_text)

    def remove_extremes(list_of_documents, removed_words: set):
        new_list_of_documents = []
        for sentence in list_of_documents:
            new_sentence = [w for w in sentence if w not in removed_words]
            if new_sentence != []:
                new_list_of_documents.append(new_sentence)
                
        return new_list_of_documents

    prepared_sentences = remove_extremes(prepared_sentences, removed_words)

    sentence_lda_scores = []

    for sentence in prepared_sentences:
        new_doc_bow = dictionary.doc2bow(sentence)
        doc_tfidf = tfidf[new_doc_bow]

        sentence_lda_scores.append(list(lda_model.get_document_topics(doc_tfidf)))

    return sentence_lda_scores

def score(vader_scores, sentence_lda_scores):
    # topic_assignments = {
    #     'food': [1,3,5,6,7,8,9],
    #     'service': [2, 4]
    # }
    # topic_assignments = {
    #     'food': [1,3,4,6],
    #     'service': [0,2],
    #     'price': [5]
    # }
    # topic_assignments = {
    #     'food': [1,3,5,6, 8, 11, 12, 13],
    #     'service': [0,3,4,9,10],
    #     'ambience': [7,14],
    #     'price': [2]
    # }
    topic_assignments = {
        'food': [0, 1, 11, 13],
        'eating': [8],
        'service': [6, 9, 12],
        'ambience': [2, 4, 5],
        'price': [3],
        'irrelevant': [7,14,8,10]
    } 

    topic_assignments_reversed = {index: key for key, value in topic_assignments.items() for index in value }

    print(vader_scores)
    print(sentence_lda_scores)

    def get_topic_scores(vader_scores, sentence_lda_scores):
        '''
        Calculates the scores per sentence and adds them up for a document. Ends up with score per topic.
        '''
        
        # create vars per topic dynamically
        counters = {}
        ret = {}
        for topic in topic_assignments.keys():
            ret['score_'+topic] = 0
            counters[topic] = 0

        for topic_scores, vader_score in zip(sentence_lda_scores, vader_scores):
            topics, scores = zip(*topic_scores)
            
            topic = topics[np.argmax(scores)]
                
            # check in which subset the key is in.
            assigned_topic = topic_assignments_reversed[topic]
            counters[assigned_topic] += 1
            ret['score_'+assigned_topic] += (vader_score - ret['score_'+assigned_topic]) / counters[assigned_topic]

        return ret

    return get_topic_scores(vader_scores, sentence_lda_scores)