#!/usr/bin/env python
# encoding=utf8
# Indivisa
# Copyright (C) 2019 Marcos Fernández Pichel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from aux import remove_non_ascii,preprocessForTopic,init_nltk
import string
import json
import traceback
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import numpy as np
import spacy
from catenae import Link,Electron,CircularOrderedSet
import pandas as pd
from pprint import pprint
import pyLDAvis
import pyLDAvis.gensim
import re
import matplotlib.pyplot as plt

class TopicAnalysis(Link):

    # Remote method to add input topics dynamically
    def add_topic(self,context,topic,id,image):
        if self.id == id:
            self.add_input_topic(topic)
            self.rpc_call('Consumer', 'recalculate_hash', args=[self.input_topics,self.id,image])

    # Remote method to remove input topics dynamically
    def remove_topic(self,context,topic,id,image):
        if self.id == id:
            self.remove_input_topic(topic)
            self.rpc_call('Consumer', 'recalculate_hash', args=[self.input_topics,self.id,image])

    # Remote callable method that allows us to add ntopics dynamically
    def add_ntopics(self,context,output_topic):
        try:
            tmp = output_topic.split("-")
            if self.id == tmp[0]:
                if tmp[1] not in self.ntopics: # Avoids duplicate values
                    print("Adding new number of topics...")
                    self.ntopics.append(tmp[1])
                    self.output_topics.append(output_topic)
                    print("Number of topics",self.ntopics)
                    print("Output topics",self.output_topics)
                    print("###################################")
        except Exception as e:
            print("Exception", e)

    # Remote method to initialize corpus in case it isn't
    def init_corpus(self,context,id,c_args):
        if self.id == id:
            if not hasattr(self,'corpus'):
                if len(c_args)==1:
                    self.corpus = CircularOrderedSet(int(c_args[0]))
                    print("Length ->",c_args[0])
                else:
                    self.corpus = []
                    self.window = c_args
                    print("Timestamps ->",self.window)

    def __make_bigram_mod(self,doc):
        bigram = gensim.models.Phrases(doc, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod

    def  __make_bigrams(self,texts):
        bigrams = []
        for doc in texts:
            bigram_mod = self.__make_bigram_mod(doc)
            bigrams.append(bigram_mod[doc])
        return bigrams

    def __lemmatization(self,texts,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): # Do lemmatization keeping only noun, adj, vb, adv
        """https://spacy.io/api/annotation"""
        nlp = spacy.load('en', disable=['parser', 'ner'])
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def __preprocess_corpus(self,corpus):
        texts = []
        for text in list(corpus): # we need to put it as a list to avoid run time errors related with changing size of the set
            for element in text:
                if element[0] == b'body' or element[0] == b'submission_title':
                    aux = preprocessForTopic(str(element[1]),self.languages)
                    texts.append(aux)

        data_words_bigrams = self.__make_bigrams(texts)

        data_lemmatized = self.__lemmatization(data_words_bigrams)

        # Create Dictionary to associate an id to every word in a document
        id2word = corpora.Dictionary(data_lemmatized)
        print("!!!Id2word",id2word)

        # Create Corpus
        texts = data_lemmatized
        for phrase in texts:
            for word in phrase:
                word = word.encode('ascii', errors='ignore')
        # Term Document Frequency (it associates the previous id with the number of times it appears in the document, e.g [0,1]-> word 0 occurs once)
        corpus = [id2word.doc2bow(text) for text in texts]
        print("!!!Corpus",corpus)
        return corpus,id2word

    def __render_model(self,model,corpus,dict,ntopics):
        data = pyLDAvis.gensim.prepare(model,corpus,dict)
        div_id = "pyldavis"
        html = pyLDAvis.prepared_data_to_html(data,template_type="simple",visid=div_id)
        found = '!function(LDAvis){' + re.search(r"\!function\(LDAvis\)\{(.+?)\}\(LDAvis\)\;", html,re.MULTILINE | re.DOTALL).group(1) + '}(LDAvis);'
        #print("Found->",found)
        return found

    def __make_topic_analysis(self,corpus,id2word,ntopics):
        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                   id2word=id2word,
                                                   num_topics=ntopics,
                                                   random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)
        pprint(lda_model.print_topics())
        top_words_per_topic = []
        for t in range(lda_model.num_topics):
            top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 5)]) # We pick the 5 words with the highest value for each topic

        return self.__render_model(lda_model,corpus,id2word,ntopics)

        # return pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_json() # The dataframe has this form: Topic - Word - P
                                                                                           #                              0       w1    0.231
                                                                                           #                              0       w2    0.142
                                                                                           #                              0       w3    0.789
                                                                                           #                              0       w4    0.542
                                                                                           #                              0       w5    0.639
                                                                                           #                              1       w1    0.541
                                                                                           #                              1       w2    0.142
                                                                                           #                              ...

    def setup(self):
        try:
            self.ntopics = []
            self.languages = init_nltk() # Initialize languages for natural language toolkit
            self.id = self.consumer_group
            print("Id del contenedor",self.id)
            if self.args:
                if "#" in self.args:
                    aux = self.args[:self.args.index("#")]
                    if len(aux)==1:
                        self.corpus = CircularOrderedSet(int(aux[0]))
                        print("Length->",aux[0])
                    else:
                        self.corpus = []
                        self.window = aux
                        print("Timestamps->",self.window)
                    aux2 = self.args[self.args.index('#')+1:]
                for arg in aux2:
                    self.ntopics.append(arg)
                print("Initial number of topics...",self.ntopics)
        except Exception as e:
            print("Exception",e)

    def send_electrons(self,electron):
        corpus,id2word = self.__preprocess_corpus(self.corpus)
        print("Ntopics",self.ntopics)
        for n,o in zip(self.ntopics,self.output_topics):
            electron.topic = o
            electron.value = self.__make_topic_analysis(corpus,id2word,n)
            self.send(electron)

    def transform(self, electron):
        try:
            if type(self.corpus) is list:
                timestamp = int(electron.value['timestamp'])
                if electron.value['src'] == 'twitter':
                    timestamp /= 1000 # because tweet timestamps are expressed in miliseconds
                if int(self.window[0]) <= timestamp <= int(self.window[1]): # if the text belongs to the given range
                    dict_items = remove_non_ascii(electron.value.items())
                    dict_set = frozenset(dict_items)  # value
                    self.corpus.append(dict_set)
                    self.send_electrons(electron)
            else:
                dict_items = remove_non_ascii(electron.value.items())
                dict_set = frozenset(dict_items)  # value
                self.corpus.add(dict_set)
                self.send_electrons(electron)
        except (Exception, UnicodeEncodeError) as e:
            print("Exception", e)
            traceback.print_exc()

if __name__ == "__main__":
    t = TopicAnalysis()
    t.start()
