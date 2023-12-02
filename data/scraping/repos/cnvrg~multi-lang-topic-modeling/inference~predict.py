# Copyright (c) 2023 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
import spacy
import gensim.corpora as corpora
import translators as ts
import joblib
from random import randrange
import os
import argparse
import tqdm
import numpy as np
from gensim.models import CoherenceModel
import pathlib
import requests

FILES = ['model_results.csv','lda_model.sav','corp_dict.sav']

BASE_FOLDER_URL = "https://libhub-readme.s3.us-west-2.amazonaws.com/multi_lang_modeling/"

def download_model_files():
    """
    Downloads the model files if they are not already present or pulled as artifacts from a previous train task
    """
    current_dir = str(pathlib.Path(__file__).parent.resolve())
    for f in FILES:
        if not os.path.exists(current_dir + f'/{f}') and not os.path.exists('/input/train/' + f):
            print(f'Downloading file: {f}')
            response = requests.get(BASE_FOLDER_URL + f)
            f1 = os.path.join(current_dir, f)
            with open(f1, "wb") as fb:
                fb.write(response.content)

download_model_files()

if os.path.exists("/input/train/lda_model.sav"):
    topic_cnt = int(pd.read_csv('/input/train/model_results.csv')['Topics_Count'][0])
    topic_cnt = 3
    model_file = '/input/train/lda_model.sav'
    topic_word_cnt = int(pd.read_csv('/input/batch_predict/topic_word_cnt.csv')['Topic_Word_Cnt'][0])
    dictionary_path = "/input/train/corp_dict.sav"
    
else:
    print('Running Stand Alone Endpoint')
    script_dir = pathlib.Path(__file__).parent.resolve()
    topic_cnt = 3
    model_file = os.path.join(script_dir,'lda_model.sav')
    topic_word_cnt = 10
    dictionary_path = os.path.join(script_dir,"corp_dict.sav")

def data_cleaning(df):
    df['text'] = df['text'].apply(str)
    # Remove punctuation
    df['text_processed'] = df['text'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    df['text_processed'] = df['text_processed'].map(lambda x: x.lower())
    # Convert sentences into individual words
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    processed_values = df.text_processed.values.tolist()
    data_words = list(sent_to_words(processed_values))
    # Phrase Modeling: Bigram and Trigram Models
    # Build the bigram and trigram models;higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # Stop Words definition
    stop_words = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now',
'd','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't",'re','edu','use']
    
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    data_words_bigrams = make_bigrams(data_words)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    # Lemmatization of words
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_lemmatized

def predict(data):
    cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
    original_text = data["txt"][4:]
    data = data["txt"]
    lda_model = joblib.load(model_file)
    corp_dict = joblib.load(dictionary_path)
    df = pd.DataFrame([[data[4:],data[:2]]],columns=['text','title'])
    texts = data_cleaning(df)#data_lemmatized
    corpus = [corp_dict.doc2bow(text) for text in texts]
    translated_list = []
    score_list = []
    for index, score in sorted(lda_model[corpus[0]], key=lambda tup: -1*tup[1])[0:topic_cnt]:
        lang_cde = data[:2]
        topics_list = []
        for topic_iteration in range(topic_cnt):
            ac = lda_model.print_topic(index, topic_word_cnt).split()
            translated_val = [ts.google(z, from_language='en', to_language=lang_cde) for z in [y.group() for y in [re.search('"(.*)"', x) for x in list(filter(('+').__ne__, ac))]]]
            joined_string = [y+'×'+z for y, z in zip([x[0:5] for x in list(filter(('+').__ne__, ac))], translated_val)]
            translated_string = " + ".join([y for y in joined_string])
        translated_list.append(translated_string)
        score_list.append(str(score))
    response = {}
    response["query doc"] = original_text[:30]
    response['topic'] = translated_list
    response['score'] = score_list
    return response
