# Credits - https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python

from ... import functions
from ... import text_processing

import numpy as np
import pandas as pd
import os.path

from gensim import models
from gensim import corpora
from gensim.models import LsiModel

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

class LSI:
    def __init__(self, args, tokenizer=None, process=None):
        params = ["n_topics","vector_mode"]
        self.args = pd.Series([vars(args)[param] for param in params], index=params)
        self.process = process
        self.tokenizer = tokenizer
        self.train_data = []

    @property
    def params(self):
        return self.args.to_dict()
        
    def initialize(self, data = None, useDataCampPrePrcoessing=True):        
        if data:
            if self.tokenizer:
                tokenized_texts = self.tokenizer(data)
            else:
                if not useDataCampPrePrcoessing:
                    processor = text_processing.Preprocessing(removeStopwords=True)
                    tokenized_texts = processor.Bulk_Tokenizer(functions.flatten(data))    
                else:
                    tokenized_texts = self.preprocess_data(functions.flatten(data))
            self.train_data = tokenized_texts

    def generate_embedding(self, data, useDataCampPrePrcoessing=True, returnarray=True):
        if self.tokenizer:
            tokenized_texts = self.tokenizer(data)
        else:
            if not useDataCampPrePrcoessing:
                processor = text_processing.Preprocessing(removeStopwords=True)
                tokenized_texts = processor.Bulk_Tokenizer(functions.flatten(data))    
            else:
                tokenized_texts = self.preprocess_data(functions.flatten(data))

        self.train_data = tokenized_texts + self.train_data

        # prepare necessary elements
        dictionary, doc_term_matrix = self.prepare_corpus(self.train_data)

        if self.args.vector_mode == "tfidf":
            tfidf = models.TfidfModel(doc_term_matrix)
            doc_term_matrix = tfidf[doc_term_matrix]

        # generate LSA model
        self.model = LsiModel(doc_term_matrix, num_topics=self.args.n_topics, id2word=dictionary)

        # Bag of Words (Term Id, Occurrence) for each document
        bow_texts = [dictionary.doc2bow(tokenized_text) for tokenized_text in tokenized_texts]

        if self.args.vector_mode == "tfidf":
            tfidf = models.TfidfModel(bow_texts)
            bow_texts = tfidf[bow_texts]

        embeddings = np.zeros((len(data), self.args.n_topics))
        for i, topic_distribution in enumerate(self.model[bow_texts]):
            for j, val in topic_distribution:
                embeddings[i,j] = val

        if not returnarray:
            embeddings = [list(each) for each in embeddings]

        return embeddings


    def preprocess_data(self, doc_set):
        """
        Input  : document list
        Purpose: preprocess text (tokenize, removing stopwords, and stemming)
        Output : preprocessed text
        """
        # initialize regex tokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        # create English stop words list
        en_stop = set(stopwords.words('english'))
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()
        # list for tokenized documents in loop
        texts = []
        # loop through document list
        for i in doc_set:
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            # add tokens to list
            texts.append(stemmed_tokens)
        return texts

    def prepare_corpus(self, doc_clean):
        """
        Input  : clean document
        Purpose: create term dictionary of our corpus and Converting list of documents (corpus) into Document Term Matrix
        Output : term dictionary and Document Term Matrix
        """
        # Creating the term dictionary of our corpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
        dictionary = corpora.Dictionary(doc_clean)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        # generate LDA model
        return dictionary, doc_term_matrix

    def compute_coherence_values(self, dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
        """
        Input   : dictionary : Gensim dictionary
                  corpus : Gensim corpus
                  texts : List of input texts
                  stop : Max num of topics
        purpose : Compute c_v coherence for various number of topics
        Output  : model_list : List of LSA topic models
                  coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, stop, step):
            # generate LSA model
            model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    def plot_graph(self, data ,start, stop, step):

        doc_clean = self.preprocess_data(functions.flatten(data))
        dictionary,doc_term_matrix=self.prepare_corpus(doc_clean)

        if self.args.vector_mode == "tfidf":
            tfidf = models.TfidfModel(doc_term_matrix)
            doc_term_matrix = tfidf[doc_term_matrix]

        model_list, coherence_values = self.compute_coherence_values(
                                                                        dictionary, \
                                                                        doc_term_matrix,\
                                                                        doc_clean,\
                                                                        stop, start, step \
                                                                    )
        # Show graph
        x = range(start, stop, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

        return x, coherence_values

    def print(self,*message):
        if self.process:           
            self.process.print(*message)
        else:
            print(*message)

    # ALIASES
    train = initialize
    infer_vector = generate_embedding