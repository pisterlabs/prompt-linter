import nltk
import numpy as np
import pandas as pd
import spacy 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel




def preprocess(df):
    '''
    Preprocess text from dataframe. Splits reviews into sentences, then each sentence is preprocessed. 
    Returns dataframe with original and preprocessed text.
    '''

    # Split into sentences 
    nlp = spacy.load('en_core_web_sm')
    review_ids = []
    text = []

    # Assuming the reviews aren't split and that reviews columns are 'review',
    # for review_id, review in zip(df['review_id'], df['review']):
    #     sentences= [i for i in nlp(review).sents]
    #     for sentence in sentences:
    #         review_ids.append(review_id)
    #         text.append(str(sentence))
    
    # No review IDs
    for review in df['review']:
        sentences= [i for i in nlp(review).sents]
        for sentence in sentences:
            # review_ids.append(review_id)
            text.append(str(sentence))

    reviews_df = pd.DataFrame()
    # reviews_df['review_id'] = review_ids
    reviews_df['raw_text'] = text

    # Remove symbols,punctuations...
    reviews_df['clean_text'] = reviews_df['raw_text'].str.replace('[^\w\s]','')
    reviews_df['clean_text'] = reviews_df['clean_text'].str.replace('\d+', '')
    reviews_df['clean_text'] = reviews_df['clean_text'].str.lower()
    reviews_df['clean_text'] = reviews_df['clean_text'].str.replace('^https?:\/\/.*[\r\n]*', '')

    reviews_df['clean_text'].replace('', np.nan, inplace=True)
    drop = reviews_df[pd.isnull(reviews_df['clean_text'])].index
    reviews_df.drop(drop , inplace=True)
    reviews_df = reviews_df.reset_index(drop = True) 
   

    
    def preprocess_aspect(df):
        '''
        Preprocessing text for aspect extraction and classification.
        Returns tf-idf/corpus, LDA model.
        '''
        
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

        data_words = list(sent_to_words(df['clean_text']))

        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

        # Define functions for stopwords, bigrams, trigrams and lemmatization
        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

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
        
        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)

        # Form Bigrams
        data_words_bigrams = make_bigrams(data_words_nostops)

        # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
        # python3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        return corpus, id2word


    porter = PorterStemmer()

    def stemSentence(sentence):
        token_words=word_tokenize(sentence)
        token_words
        stem_sentence=[]
        for word in token_words:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    stemmed = []
    for sentence in reviews_df['clean_text']:
        stemmed.append(stemSentence(sentence))


    reviews_df['stem_text'] = stemmed
    
    # corpus, id2word = preprocess_aspect(reviews_df)

    # Remove stop words
    # stop_words = set(stopwords.words('english'))
    # reviews_df['no_sw'] = reviews_df['clean_text'][:].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    return reviews_df #, corpus, id2word