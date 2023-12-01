import warnings

from gensim import models

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import gc
import gensim
import pyLDAvis
import numpy as np
import pandas as pd
import pyLDAvis.gensim
import gensim.downloader
from termcolor import colored
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from gensim.models.coherencemodel import CoherenceModel


gc.enable()
global mallet_path
os.environ['MALLET_HOME'] = 'C:/mallet'
mallet_path = 'C:/mallet/bin/mallet'  # update this path


# ======================================================================================================================
class TopicModeling():
    def __init__(self, train_set:pd.Series, bigram:bool=False):
        self.lda_model = None
        self.train_set = np.asarray(train_set)
        self.__bigram = bigram
        self.corpus, self.dictionary, self.word_list = self.__get_corpus()

    def __make_bigrams(self, words:list, bi_min:int=10) -> gensim.models.phrases.Phraser:
        bigram = gensim.models.Phrases(words, min_count=bi_min, threshold=3)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod

    def __get_corpus(self, **kwargs) -> tuple:

        try:    document = kwargs['document']
        except: document = self.train_set

        if not isinstance(document, np.ndarray):  train_array = np.asarray(document["preprocessed"])
        else:                                     train_array = document
        
        bag_of_words = [line.split() for line in train_array if line is not np.nan]

        if not self.__bigram: word_list = bag_of_words
        else:
            bigram_mod = self.__make_bigrams(bag_of_words)
            word_list = [bigram_mod[review] for review in bag_of_words]
            
        id2word = gensim.corpora.Dictionary(word_list)
        id2word.compactify()
        id2word.filter_extremes(no_below=9, no_above=0.13, keep_n=100000)
        corpus = [id2word.doc2bow(text) for text in word_list]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf, id2word, word_list

    def __topic_extraction(self) -> (list, list):
        __topic = list()
        __percentage = list()
        for idx, topic in self.lda_model.print_topics(-1, num_words=10):
            print('\nMALLET Topic: {} \nWords: {}'.format(idx, topic))
            splitted = topic.split('+')
            __topic.append([])
            __percentage.append([])
            for word in splitted:
                __topic[-1].append(word.split('*')[1].split('"')[1])
                __percentage[-1].append(word.split('*')[0])

        __model_array = list()
        for i in range(len(__topic)):
            __model_array.append(__topic[i])
            __model_array.append(__percentage[i])
        return __model_array, __topic

    def __coherence(self, **kwargs) -> float:

        try:    lda_model = kwargs['model']
        except: lda_model = self.lda_model

        try:    word_list = kwargs['texts']
        except: word_list = self.word_list

        try:    dictionary = kwargs['dictionary']
        except: dictionary = self.dictionary

        coherence_model_lda = CoherenceModel(model=lda_model, texts=word_list, dictionary=dictionary,
                                             coherence='c_v')
        coherence_value = coherence_model_lda.get_coherence()
        return coherence_value

    def __visualization(self, num_topics: int) -> int:
        lda_model = self.lda_model
        try:
            visualisation = pyLDAvis.gensim.prepare(lda_model, self.corpus, self.dictionary)
            modelStr = 'gensim'
        except:
            model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model)
            visualisation = pyLDAvis.gensim.prepare(model, self.corpus, self.dictionary)
            modelStr = 'mallet'
        pyLDAvis.save_html(visualisation, "results/" + modelStr + 'LDA_Visualization_' + str(num_topics) + 'topic.html')
        return 1

    def cross_validation(self, limit:int, start:int=2, step:int=3) -> (list, float):
        X = np.array(self.train_set)
        kf = KFold(5, shuffle=True, random_state=42)
        coherence_values = []
        for num_topics in range(start, limit, step):
            num_coherence = []
            for _, train_ind in kf.split(X):
                X_cv = X[train_ind]
                corpus, dictionary, word_list = self.__get_corpus(document=X_cv)

                model = gensim.models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
                num_coherence.append(self.__coherence(model=model, texts=word_list, dictionary=dictionary))
                
                del model
                
            coherence_values.append(num_coherence)
            gc.collect()
        return coherence_values

    def topic_modeling(self, **kwargs):
        """https://radimrehurek.com/gensim_3.8.3/models/wrappers/ldamallet.html"""
        try:
            library = kwargs['library']
            num_topics = kwargs['num_topics']
            iterations = kwargs['iterations']
        except:
            library = 'hdp'
            num_topics = 20
            iterations = 1000
            print(colored("model goes with it's default number of topics: #" + str(num_topics), 'green'))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if library == 'mallet':
                lda_model = gensim.models.wrappers.LdaMallet(
                    mallet_path,
                    corpus=self.corpus,
                    num_topics=num_topics,
                    id2word=self.dictionary,
                    workers=3,
                    #iterations=iterations,
                    random_seed=42)

            elif library == 'gensim':
                lda_model = gensim.models.LdaModel(
                    corpus=self.corpus,
                    num_topics=num_topics,
                    id2word=self.dictionary,
                    passes=2)

            elif library == 'hdp':
                lda_model = gensim.models.hdpmodel.HdpModel(
                    corpus=self.corpus,
                    id2word=self.dictionary,
                    max_time=1000,
                    max_chunks=None,
                    chunksize=256,
                    kappa=1.0,
                    tau=64.0,
                    K=15, T=150,
                    alpha=0.5, gamma=0.8,
                    eta=0.01, scale=1.0,
                    var_converge=0.0000001,
                    random_state=None)

            else:
                raise ValueError("Wrong library name. select 'gensim' or 'mallet'")

        self.lda_model = lda_model
        model_array, model_topic = self.__topic_extraction()
        np.savetxt("results/" + library + "_" + str(num_topics) + "topics.csv", model_array, delimiter=",", fmt='%s',
                   encoding="utf-8")

        coherence_value = self.__coherence()
        print(colored('Coherence value is:\t' + str(coherence_value), 'cyan'))

        # self.__visualization(num_topics)
        return model_array, model_topic

 
# ======================================================================================================================
line = lambda array: [((array[-1]-array[0])/(len(array)-1)) * x_ind + array[0] for x_ind in range(len(array))]
optimal_elbow = lambda array: np.argmax([array[ind] - line(array)[ind] for ind in range(len(array))])


def elbow_method(coherence_values, name:str, start=5, limit=32, step=3) -> int:
    # Show graph
    mean = [np.array(coherence_values[i]).mean() for i in range(0, len(coherence_values))]
    std_plus = [mean[i] + np.array(coherence_values[i]).std() for i in range(0, len(coherence_values))]
    std_neg = [mean[i] + -np.array(coherence_values[i]).std() for i in range(0, len(coherence_values))]
    plot_coherence(mean, std_neg, std_plus, name, start, limit, step)
    optimal_topic = min(optimal_elbow(mean), optimal_elbow(std_plus), optimal_elbow(std_neg))
    
    return start + step*optimal_topic


def plot_coherence(mean:list, std_neg:list, std_plus:list, name:str, start:int, limit:int, step:int):
    x = range(start, limit, step)
    print(mean)
    print(std_plus)
    print(std_neg)
    plt.plot(x, mean, '-or', label='mean')
    plt.plot(x, std_plus, '-', color='gray', label='std')
    plt.plot(x, std_neg, '-', color='gray')
    plt.fill_between(x, std_neg, std_plus, color='gray', alpha=0.2)
    plt.xlim(start - 0.025, limit - 1 + 0.025)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(loc='best')
    string = "picture/coherence-topics-" + name + ".png"
    plt.savefig(string)
    plt.clf()


