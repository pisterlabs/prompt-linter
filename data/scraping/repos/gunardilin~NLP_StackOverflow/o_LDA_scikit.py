from sklearn.pipeline import Pipeline
from h_readsqlite import SqliteCorpusReader
from i_vectorizer import TextNormalizer, GensimVectorizer, GensimVectorizer_Topic_Discovery
from i_vectorizer import NgramVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF

from gensim.models import LsiModel, LdaModel#, EnsembleLda
from o_ensemble_lda_foundation import EnsembleLda
from o_ldamodel import LdaTransformer
from o_lsimodel import LsiTransformer
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

import time
from l_pipeline import timer

import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim_models

import os
import warnings
import sys
import pickle
import itertools

import webbrowser
import platform

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Logging for debug purpose:
gensim_logfile_path = 'other/temp/gensim_logs.log'
# if os.path.exists(gensim_logfile_path):
#     os.remove(gensim_logfile_path)
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
#                    level=logging.DEBUG,
#                    filename=gensim_logfile_path)

class SklearnEstimator(object):
    def __init__(self, n_topics=50, estimator="LDA"):
        self.n_topics = n_topics
        self.documents = None

        if estimator == 'LSA':
            self.estimator = TruncatedSVD(n_components=self.n_topics)
        elif estimator == 'NMF':
            self.estimator = NMF(n_components=self.n_topics)
        else:
            self.estimator = LatentDirichletAllocation(n_components=self.n_topics)
    
    def fit(self, documents, labels=None):
        return self
    
    def transform(self, documents):
        self.documents = list(documents)
        self.estimator.fit_transform(self.documents)
        
class SklearnTopicModels(object):
    def __init__(self, n_topics=20, estimator="LDA", \
        db_path:str="DB/StackOverflow.sqlite", \
        gensim_lexicon:str="other/lexicon.pkl"):
        self.n_topics = n_topics
        self.corpus_reader = SqliteCorpusReader(path=db_path)
        self.model = Pipeline([
            ("norm", TextNormalizer()),
            ("vect", GensimVectorizer(gensim_lexicon, False, True)),
            ("model", SklearnEstimator(self.n_topics, estimator)),
        ])
        self.estimator_model = None
    
    def fit_transform(self, year:int):
        docs = self.corpus_reader.docs(year)
        self.model.fit_transform(docs)
        return self.model
    
    def get_topics(self, n_words=25):
        vectorizer = self.model.named_steps['vect']
        # Get object Estimator
        self.estimator_model = self.model.steps[-1][1]
        # Get a dict with id as index and token as value: 
        names = vectorizer.id2word.id2token
        topics = dict()
        
        for idx, topic in enumerate(self.estimator_model.estimator.components_):
            # topic is an array of token weights
            # In next line we get the indices that would sort an array.
            # The indices would be starting from token with most weight only
            # as much as n_words.
            features = topic.argsort()[:-(n_words-1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
        return topics
    
    def visualize_topic(self):
        # lda_tf -> self.estimator_model
        # dtm_tf -> self.model.steps[-1][1].documents
        # tf_vectorizer -> self.model.steps[-2][1]
        pyLDAvis.sklearn.prepare(self.estimator_model.estimator,\
            self.model.steps[-1][1].documents, \
            self.model.named_steps['vect'])
        # pyLDAvis.gensim_models.prepare()
        return

class GensimTopicModels(object):
    def __init__(self, n_topics=20, estimator="LDA", \
        db_path:str="DB/StackOverflow.sqlite", \
        eval_every=None,    # Change to 1 when finding optimal passes and itterations parameters
        random_state=None,
        memory_friendly=True, # memory_friendly false necessary for find_optimal_lda_num_topics
        doc_matrix=None,    # Necessary for find_optimal_lda_num_topics
        vectorizer=None):   # Necessary for find_optimal_lda_num_topics
        self.lexicon_path = None
        self.estimator_str = estimator
        self.corpus_reader = SqliteCorpusReader(path=db_path)     
        self.n_topics = n_topics
        self.start_year = None
        self.end_year = None
        self.docs = []
        self.doc_matrix = doc_matrix
        self.vectorizer = vectorizer
        self.id2word = None
        self.model = None
        self.model_folderpath = None#"other/model"
        self.model_filepath = None#"{}/{}_model".format(self.model_folderpath, estimator)
        self.tempFolderPath = "other/temp"
        self.doc_matrix_pickle_path = None
        self.memory_friendly = memory_friendly

        if estimator == 'LSA':
            self.estimator = LsiTransformer(num_topics=self.n_topics)
        elif estimator == 'LDA':
            self.estimator = LdaTransformer(num_topics=self.n_topics, \
                eval_every=eval_every, passes=20, iterations=400, \
                random_state=random_state)
        elif estimator == 'ensembleLDA':
            self.estimator = "ensembleLDA"
        # self.load_model()
        
    
    def save_model(self):
        if not os.path.exists(self.model_folderpath):
            os.makedirs(self.model_folderpath)
        if self.estimator_str == 'ensembleLDA':
            self.estimator.save(self.model_filepath)
        elif self.estimator_str == 'LDA' or self.estimator_str == 'LSA':
            self.estimator.gensim_model.save(self.model_filepath)
        return
    
    def load_model(self):
        if os.path.exists(self.model_filepath):
            if self.estimator == "LDA":
                self.estimator.gensim_model = LdaModel.load(self.model_filepath)
            elif self.estimator == "LSI":
                self.estimator.gensim_model = LsiModel.load(self.model_filepath)
        return
    
    def fit(self, year:int):
        docs = self.corpus_reader.docs(year)
        self.docs = list(docs)
        self.doc_matrix = self.model.fit_transform(self.docs)
        self.docs = None                # Free up memory
        vectorizer = self.model.named_steps['vect']
        vectorizer.documents = None     # Free up memory
        self.estimator.id2word = vectorizer.id2word.id2token
        self.estimator.partial_fit(self.doc_matrix)
        self.save_doc_matrix_to_pickle(self.doc_matrix)
        self.doc_matrix = None
        self.save_model()
        return self.model
    
    def get_Doc_matrix_Vectorizer(self, start_year:int, end_year:int, limit=None):
        # Extract from corpus doc_matrix and vectorizer
        for year in range(start_year, end_year+1):
            print("Reading corpus from Y{}".format(year))
            docs = self.corpus_reader.docs(year, limit)
            self.docs += list(docs)      
        print("Variable size: {}".format(sys.getsizeof(self.docs)))
        del docs                        # Free up memory
        if self.model == None:
            self.model_folderpath = "other/model_{}-{}".format(start_year, end_year)
            self.lexicon_path = "{}/lexicon.pkl".format(self.model_folderpath)
            if not os.path.exists(self.model_folderpath):
                os.makedirs(self.model_folderpath)
            self.model = Pipeline([
                ("norm", TextNormalizer()),
                ("ngram", NgramVectorizer()),
                ("vect", GensimVectorizer_Topic_Discovery(self.lexicon_path, False, True)),
            ])
        
        doc_matrix = self.model.fit_transform(self.docs)
        vectorizer = self.model.named_steps['vect']
        if self.memory_friendly:
            self.docs = None                # Free up memory
            # vectorizer.documents = None     # Free up memory
        return doc_matrix, vectorizer
    
    def fit_multi_years(self, start_year:int, end_year:int, limit=None, 
                        vectorizer=None, doc_matrix=None):
        self.start_year = start_year
        self.end_year = end_year
        
        self.model_folderpath = "other/model_{}-{}_collections".format(start_year, end_year)
        n = 1
        self.model_filepath = "{}/{}_model_{}_topics_{}".format(self.model_folderpath,\
            self.estimator_str, self.n_topics, n)
        while True:
            if os.path.exists(self.model_filepath):
                n += 1
                self.model_filepath = "{}/{}_model_{}_topics_{}".format(self.model_folderpath,\
                    self.estimator_str, self.n_topics, n)
            else:
                break
        
        self.doc_matrix_pickle_path = "{}/doc_matrix.pickle".format(self.model_folderpath)
        
        self.lexicon_path = "{}/lexicon.pkl".format(self.model_folderpath)
        self.model = Pipeline([
            ("norm", TextNormalizer()),
            ("ngram", NgramVectorizer()),
            ("vect", GensimVectorizer_Topic_Discovery(self.lexicon_path, False, True)),
        ])
        
        if vectorizer==None or doc_matrix==None:
            print("Extracting from Corpus the doc_matrix and vectorizer.")
            
            # Delete existing lexicon  only during first run, when vectorizer 
            # and doc_matrix are not supplied.
            if os.path.exists(self.lexicon_path):
                self.remove_temp([self.lexicon_path])
            
            self.doc_matrix, self.vectorizer = self.get_Doc_matrix_Vectorizer(\
                start_year, end_year, limit)
        else:
            # This else mode is necessary when executing find_optimal_lda_num_topics
            # This way the whole corpus doesn'need to be reread multiple times.
            self.doc_matrix, self.vectorizer = doc_matrix, vectorizer
            print("Doc_matrix and vectorizer are supplied.")

        if self.estimator != "ensembleLDA":
            self.estimator.id2word = self.vectorizer.id2word.id2token
            self.estimator.partial_fit(self.doc_matrix)
        elif self.estimator == "ensembleLDA":
            
            # # Increase max recursion limit. Necessary for EnsembleLda.
            # sys.setrecursionlimit(2500)
            
            corpus = self.doc_matrix
            dictionary = vectorizer.id2word.id2token
            topic_model_class = LdaModel
            ensemble_workers = 4
            num_models = 40
            distance_workers = None
            num_topics = self.n_topics
            passes=50
            iterations=5000
            epsilon = 1
            eval_every=None
            
            self.estimator = EnsembleLda(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=passes,
                iterations=iterations,
                num_models=num_models,
                topic_model_class=topic_model_class,
                ensemble_workers=ensemble_workers,
                distance_workers=distance_workers,
                epsilon=epsilon,
                eval_every=eval_every
            )
        
        self.save_doc_matrix_to_pickle(self.doc_matrix)
        self.doc_matrix = None
        self.save_model()
        return self.model
    
    def save_doc_matrix_to_pickle(self, data):
        if not os.path.exists(self.tempFolderPath):
            os.makedirs(self.tempFolderPath)
        with open(self.doc_matrix_pickle_path,"ab") as f:
            pickle.dump(data, f)
        print("Data is saved into pickle.")
        return
    
    def load_from_pickle(self, path):
        def loader(fp):
            # a load iter
            while True:
                try:
                    yield pickle.load(fp)
                except EOFError:
                    print("Generator is exhausted")
                    break
        with open(path, "rb") as f:
            for i in loader(f):
                yield(i)
        return
    
    def remove_temp(self, path_list:list):
        for filePath in path_list:
            if os.path.exists(filePath):
                os.remove(filePath)
        return
    
    def get_topics(self, n_words=25):
        print('*** Getting topics...')
        names = self.estimator.id2word
        topics = dict()
        if self.estimator_str == "ensembleLDA":
            get_topics = self.estimator.get_topics()
        elif self.estimator_str != "ensembleLDA":
            get_topics = self.estimator.gensim_model.get_topics()
        for idx, topic in enumerate (get_topics):
            features = topic.argsort()[:-(n_words-1): -1]
            tokens = [names[i] for i in features]
            topics[idx] = tokens
        return topics
    
    def visualize_topics(self, open_in_browser=True):
        # html_path = '{}/lda_{}-{}_{}_topics.html'.format(self.model_folderpath, \
        #     self.start_year, self.end_year, self.n_topics)
        html_path = "{}_Vis.html".format(self.model_filepath)
        if self.estimator_str != "LDA":
            print("** The pyLDAvis is only compatible for LDA not the currently used model.")
            return
        lda_model = self.estimator.gensim_model
        temp_list = self.load_from_pickle(self.doc_matrix_pickle_path)
        corpus = []
        for i in temp_list:
            corpus += i
        #lexicon = self.model.named_steps['vect'].id2word
        #lexicon = lda_model.id2word
        lexicon = Dictionary.load(self.lexicon_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                data = pyLDAvis.gensim_models.prepare(lda_model, corpus, lexicon)
            except:
                print("Something's wrong with Pyldavis source code.")
                return
            pyLDAvis.save_html(data, html_path)
        
        if platform.system().lower() == "darwin":  # check if on Mac OSX
            webbrowser.get('macosx')
            file_prefix = "file:///"
        else:   # all other non Mac
            file_prefix = ""
        
        if open_in_browser: 
            webbrowser.open(file_prefix+os.path.abspath(html_path),new=2)
        return
    
    def optimize_ensembleLda(self, new_epsilon="max"):
        if self.estimator_str != "ensembleLDA":
            print ("This Optimization is only for ensemble LDA.")
            return
        print('*** Optimize ensemble LDA model.')
        import numpy as np
        shape = self.estimator.asymmetric_distance_matrix.shape
        without_diagonal = self.estimator.asymmetric_distance_matrix[~np.eye(shape[0], dtype=bool)].reshape(shape[0], -1)
        
        print("Min, mean & max value of asymetric distance matrix:")
        print(without_diagonal.min(), without_diagonal.mean(), without_diagonal.max())
        
        if new_epsilon=="max":
            new_epsilon = without_diagonal.max()
        elif type(new_epsilon) == int:
            new_epsilon = new_epsilon
        self.estimator.recluster(eps=new_epsilon, min_samples=2, min_cores=2)
        return 
    
    def parse_logfile(self):
        import re
        import matplotlib.pyplot as plt
        p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
        matches = [p.findall(l) for l in open(gensim_logfile_path)]
        matches = [m for m in matches if len(m) > 0]
        tuples = [t[0] for t in matches]
        perplexity = [float(t[1]) for t in tuples]
        liklihood = [float(t[0]) for t in tuples]
        iter = list(range(0,len(tuples)*10,10))
        plt.plot(iter,perplexity,c="black")
        plt.ylabel("log liklihood")
        plt.xlabel("iteration")
        plt.title("Topic Model Convergence")
        plt.grid()
        plt.savefig("other/convergence_likelihood.pdf")
        plt.close()
        return
    
def find_optimal_lda_num_topics(min=5, max=15, step=1, start_year=2022, end_year=2022, \
    limit=None, random_state=100):
    # refer to: 
    # https://rare-technologies.com/what-is-topic-coherence/
    # https://nbviewer.org/github/devashishd12/gensim/blob/280375fe14adea67ce6384ba7eabf362b05e6029/docs/notebooks/topic_coherence_tutorial.ipynb#topic=1&lambda=1&term=
    
    model = GensimTopicModels()
    doc_matrix, vectorizer = model.get_Doc_matrix_Vectorizer(start_year, \
        end_year, limit)
    
    def read_lexicon(path):
        lexicon = Dictionary.load(path)
        for key, value in lexicon.token2id.items():
            lexicon.id2token[value] = key
        return lexicon
    
    lexicon = None
        
    coherence_values = []
    for num_topics in range(min, max+1, step):
        print("*** Evaluating num_topics: {}".format(num_topics))

        model = GensimTopicModels(n_topics=num_topics, estimator="LDA", \
            random_state=random_state, memory_friendly=False)
        model.fit_multi_years(start_year=start_year, end_year=end_year, \
            limit=limit, vectorizer=vectorizer, doc_matrix=doc_matrix)
        
        # vectorizer = model.model.named_steps['vect']

        if lexicon == None:
            folder_path = "other/model_{}-{}".format(start_year, end_year)
            lexicon_path = "{}/lexicon.pkl".format(folder_path)
            lexicon = read_lexicon(lexicon_path)
        
        coherencemodel = CoherenceModel(model=model.estimator.gensim_model, texts=vectorizer.documents, dictionary=lexicon, coherence='c_v')
        coherence_score = coherencemodel.get_coherence()
        print("->-> Coherence_score for {} topics: {}".format(num_topics, coherence_score))
        coherence_values.append(coherence_score)
        model.visualize_topics(False)
    
    # show graph
    import matplotlib.pyplot as plt
    x = range(min, max+1, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend('', frameon=False)
    # plt.legend(("coherence_values"), loc='best')
    plt.savefig("{}/convergencegraph_{}-{}.pdf".format(folder_path, start_year, end_year))
    plt.show()
    
    return

def find_optimal_lda_num_topics_slow(min=5, max=15, step=1, start_year=2022, end_year=2022, \
    lexicon_path="other/model/lexicon.pkl", limit=None, random_state=100):
    # For some reason the function "find_optimal_lda_num_topics" stops without 
    # any error message. If limit argument is given, the function works well.
    # My guess: it is due to not enough memory capacity (8GB) on MacBook Air M2.
       
    def read_lexicon(path):
        print("Reading lexicon.")
        lexicon = Dictionary.load(path)
        for key, value in lexicon.token2id.items():
            lexicon.id2token[value] = key
        return lexicon
    
    lexicon = None    
    coherence_values = []
    for num_topics in range(min, max+1, step):
        print("*** Evaluating num_topics: {}".format(num_topics))

        model = GensimTopicModels(n_topics=num_topics, estimator="LDA", \
            random_state=random_state, memory_friendly=True)
        model.fit_multi_years(start_year=start_year, end_year=end_year, \
            limit=limit)
        
        vectorizer = model.model.named_steps['vect']
        
        # Get lexicon:
        if lexicon == None:
            lexicon = read_lexicon(lexicon_path)
        
        coherencemodel = CoherenceModel(model=model.estimator.gensim_model, texts=vectorizer.documents, dictionary=lexicon, coherence='c_v')
        coherence_score = coherencemodel.get_coherence()
        print("->-> Coherence_score for {} topics: {}".format(num_topics, coherence_score))
        coherence_values.append(coherence_score)
        model.visualize_topics(False)
    
    # show graph
    import matplotlib.pyplot as plt
    x = range(min, max+1, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig("other/convergencegraph_{}-{}.pdf".format(start_year, end_year))
    plt.show()
    
    return
        
        

if __name__ == "__main__":
    
    ## With Sklearn
    # start_time = time.time()
    # model = SklearnTopicModels(n_topics=50, estimator="LDA")
    # model.fit_transform(2022)
    # topics = model.get_topics()
    # for topic, term in topics.items():
    #     print("Topic #{}:".format(topic+1))
    #     print(term)  
    # # model.visualize_topic()
    # timer(start_time, time.time())
    
    
    ## With Gensim for single year
    # start_time = time.time()
    # model = GensimTopicModels(n_topics=50, estimator="LDA")
    # model.fit(2022)
    # # print(model.estimator.gensim_model.print_topics(10))
    # topics = model.get_topics()
    # n = 0
    # for topic in topics.values():
    #     n += 1
    #     print("Topic #{}:".format(n))
    #     print(topic)
    # model.visualize_topics()
    # timer(start_time, time.time())
    

    ## With Gensim for multi years
    # start_time = time.time()
    # model = GensimTopicModels(n_topics=7, estimator="LDA")
    # model.fit_multi_years(start_year=2019, end_year=2021, limit=100)
    # print(model.estimator.gensim_model.print_topics(10))
    # # # model.optimize_ensembleLda()
    # topics = model.get_topics()
    # n = 0
    # for topic in topics.values():
    #     n += 1
    #     print("Topic #{}:".format(n))
    #     print(topic)
    # model.visualize_topics()
    # # model.parse_logfile()
    # timer(start_time, time.time())
    
    
    ## Check optimal num topics
    # start_time = time.time()
    # find_optimal_lda_num_topics(5, 8, 1, 2021, 2021, limit=100)
    # timer(start_time, time.time())

    ## Generate collections of LDA models:
    start_time = time.time()
    for i in range(10):
        model = GensimTopicModels(n_topics=6, estimator="LDA")
        model.fit_multi_years(start_year=2012, end_year=2021)
        # print(model.estimator.gensim_model.print_topics(10))
        # # model.optimize_ensembleLda()
        # topics = model.get_topics()
        # n = 0
        # for topic in topics.values():
        #     n += 1
        #     print("Topic #{}:".format(n))
        #     print(topic)
        model.visualize_topics(open_in_browser=False)
        # model.parse_logfile()
        timer(start_time, time.time())