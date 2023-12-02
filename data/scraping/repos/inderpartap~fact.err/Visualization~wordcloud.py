#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
from wordcloud import WordCloud, STOPWORDS 




def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  



def process_words(texts,bigram_mod,trigram_mod, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out




def main():
    fake_df = pd.read_csv("News _dataset/Fake.csv")
    true_df = pd.read_csv("News _dataset/True.csv")
    # Convert to list
    fake_data = fake_df.text.values.tolist()
    fake_data_words = list(sent_to_words(fake_data))
    #print(fake_data_words[:1])
    true_data = true_df.text.values.tolist()
    true_data_words = list(sent_to_words(true_data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(fake_data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[fake_data_words], threshold=100)  
    bigram_mod_fake = gensim.models.phrases.Phraser(bigram)
    trigram_mod_fake = gensim.models.phrases.Phraser(trigram)

    bigram = gensim.models.Phrases(true_data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[true_data_words], threshold=100)  
    bigram_mod_true = gensim.models.phrases.Phraser(bigram)
    trigram_mod_true = gensim.models.phrases.Phraser(trigram)


    fake_data_ready = process_words(fake_data_words,bigram_mod_fake,trigram_mod_fake)
    true_data_ready = process_words(true_data_words,bigram_mod_true,trigram_mod_true)

    id2word_fake = corpora.Dictionary(fake_data_ready)
    id2word_true = corpora.Dictionary(true_data_ready)

    # Create Corpus: Term Document Frequency
    corpus_fake = [id2word_fake.doc2bow(text) for text in fake_data_ready]
    corpus_true = [id2word_true.doc2bow(text) for text in true_data_ready]


    wordCount_fake = {}
    for doc in fake_data_ready:
        for word in doc:
            if word not in wordCount_fake:
                wordCount_fake[word] =0
            else:
                wordCount_fake[word] +=1
        

    wordCount_true = {}
    for doc in true_data_ready:
        for word in doc:
            if word not in wordCount_true:
                wordCount_true[word] =0
            else:
                wordCount_true[word] +=1


    wc = WordCloud(background_color="white",width=6000,height=6000,normalize_plurals=False,min_font_size=12).generate_from_frequencies(wordCount_fake)
    plt.imshow(wc)
    plt.savefig("checking.jpg")



    wc = WordCloud(background_color="white",width=4000,height=4000,relative_scaling=0.5,normalize_plurals=False).generate_from_frequencies(wordCount_true)
    plt.imshow(wc)
    plt.savefig("true_words.jpg")


    import json


    #saving wordcloud in json format for easy reading
    result=[]
    for k, v in wordCount_true.items():
        if v > 0:
            result.append({'name':k, 'weight':v})
        

    with open('wc_true.json', 'w') as fp:
        json.dump(result, fp)


    result=[]
    for k, v in wordCount_fake.items():
        if v > 0:
            result.append({'name':k, 'weight':v})
        

    with open('wc_fake.json', 'w') as fp:
        json.dump(result, fp)




if __name__ == "__main__":
    main()