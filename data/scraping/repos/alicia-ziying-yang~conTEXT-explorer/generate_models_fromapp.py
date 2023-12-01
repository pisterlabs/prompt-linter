#!/usr/bin/env python
# coding: utf-8

# # 0. Import all the Required Packages
import nltk
import re
import numpy as np
import pandas as pd
from pprint import pprint
from nltk import sent_tokenize
import glob, time, gc, datetime

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

# spacy for lemmatization
import spacy
import os


# # 1. Preprocess the Documents and store the Documents in a .pkl file
# - Input: All the .csv files
# - Output: Processed content .pkl files

def build_model(df_full,corpus_name,content_col):
    #corpus_name : must new a folder with the same name in the "./topic_model/"

    ID_col_name = ""
    start_time = time.time()

    try:
        df = pd.DataFrame.from_records(df_full,columns=[content_col])
        df.columns=["body"]
        df["ID"] = [x for x in range(1, len(df.values)+1)]
        
        path = os.path.join("./topic_model/", corpus_name)
        
        try:
            os.mkdir(path)
        except OSError as error:  
            return str(error)

        file_name = '/selected_content_' + corpus_name + '.pkl'
        df.to_pickle(path+file_name)
              
    except:
        print("reading failed", time.time() - start_time)
        exit(-1)


    # NLTK Stop words
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    

    def doc_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out


    total_time_for_one_doc = time.time()
    
    ID_list=df.ID.values.tolist()
    data = df.body.values.tolist()

    # Remove new line characters

    data = [sent.replace('\\n', ' ') for sent in data if type(sent) is str]
    
    data = [sent.replace('\n', ' ') for sent in data if type(sent) is str]
        
    data = [sent.replace('.', '. ') for sent in data if type(sent) is str]

    data = [sent.replace('  ', ' ') for sent in data if type(sent) is str]

        
    gc.collect()

    print("1. Converting document to words for", file_name, "...", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    data = list(doc_to_words(data))
    print("Converting doc to word time:", time.time() - start)

    gc.collect()

    # Build the bigram model
    print("2. Building the bigram model for", file_name, "...", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    bigram = gensim.models.Phrases(data, min_count=5, threshold=100) # higher threshold fewer phrases.
    print("Building Bigram:", time.time() - start)

    # Faster way to get a sentence clubbed as a trigram/bigram
    print("3. Building the bigram model for", file_name, "...", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    print("Building Bigram Model:", time.time() - start)
    
    # Remove Stop Words
    print("4. Removing stop words for", file_name, "...", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    data = remove_stopwords(data)
    print("Time spent on removing stopwords:", time.time() - start)

    # Form Bigrams
    print("5. Forming bigrams for", file_name, "...", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    data = make_bigrams(data)
    print("Time spent on forming bigrams:", time.time() - start)

    # Do lemmatization keeping only noun, adj, vb, adv
    print("6. Lemmatizing", file_name, "...", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    data = lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print("Time spent on lemmatizing:", time.time() - start)


    print("7. Writing into pickle...", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    processed_df = pd.DataFrame([[ID_list,data]], columns = ['ID','body'])
    pkl_file_name = "./topic_model/"+corpus_name+"/processed_content_" + corpus_name + '.pkl'
    processed_df.to_pickle(pkl_file_name)

    print("Total process time for one document", time.time() - total_time_for_one_doc, str(datetime.datetime.now()).split('.')[0])



    # # 2. Create the Dictionary from the Processed Content
    # - Input: Processed Content .pkl files
    # - Output: Dictionary (gensim.Dictionary.id2word file)


    print("Start Reading:", str(datetime.datetime.now()).split('.')[0])
    start = time.time()
    id2word = corpora.Dictionary(pd.read_pickle(pkl_file_name).body.values.tolist()[0])
    print(len(id2word))

    id2word.add_documents(pd.read_pickle(pkl_file_name).body.values.tolist()[0])
    gc.collect()

    print("Read time:", time.time() - start)

    id2word.save("./topic_model/"+corpus_name+"/content_dictionary_"+corpus_name)
    

    # 3. Form the Corpus with the Dictionary and Processed Content
    # - Input: Dictionary (gensim.Dictionary.id2word file) & Processed Content .pkl files
    # - Output: Corpus .pkl files

    print("Start Reading:", str(datetime.datetime.now()).split('.')[0])
    total = time.time()
    corpus = []

    start = time.time()
    data = pd.read_pickle(pkl_file_name).body.values.tolist()[0]
    corpus = [id2word.doc2bow(text) for text in data]
    print("length of data:", len(data), "; length of corpus", len(corpus))
    corpus_df = pd.DataFrame([[corpus]], columns = ['corpus'])
    print("Shape of the corpus in this iteration:", corpus_df.shape)
    save_file_name = "./topic_model/"+corpus_name+"/corpus_" + corpus_name + ".pkl"
    corpus_df.to_pickle(save_file_name)

    print("Total time:", time.time() - total)

    return True


