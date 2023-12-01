#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 18:21:15 2018

@author: Fabs
"""
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.util import ngrams
from wordcloud import WordCloud
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import learning_curve
from gensim import corpora
from gensim.models import LsiModel, TfidfModel, phrases 
from gensim.models.coherencemodel import CoherenceModel
from textblob import TextBlob, Blobber
from concurrent.futures import ProcessPoolExecutor, as_completed
from IPython.display import display_html
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve
from collections import Counter
import string as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import spacy

wnl = WordNetLemmatizer()
nlp = spacy.load('en')

def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, 
                   figure_size=(24.0,16.0), title = None, title_size=40, 
                   image_color=False, stop_words = None, scale = 1, 
                   collocations = True):
    ''' credits to: https://www.kaggle.com/aashita/word-clouds-of-various-shapes
    '''
    wordcloud = WordCloud(background_color='white',
                    stopwords = stop_words,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask,
                    scale = scale, 
                    collocations=collocations)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  

def clean_text(text, lowering = True, remove_punct = True, 
               lemmatization = True, special_char = True, stop_words = None):
    
    # Word tokenization
    words = text.split()
        
    # Lowercasing
    if (lowering):
        words = [word.lower() for word in words]
        
    # Special Character removal
    if (special_char):
        words = [re.sub(r'[^\x00-\x7F]+','', word) for word in words]

    # Punctuation marks removal
    if (remove_punct):
        translator = str.maketrans('', '', st.punctuation)
        # words = filter(lambda x: x != "'s", words) # special case
        words = [word.translate(translator) for word in words]
        words = filter(None, words)
    
    # Stopwords removal
    if (stop_words is not None):
        words = [word for word in words if word not in stop_words] 
        
    # Lemmatization
    if (lemmatization):    
        words = [wnl.lemmatize(word, pos='v') for word in words]  
    
    # Words to sentence
    sentence = " ".join(words)
    return sentence

def get_clean_stopwords(stop_words):
    words = [word.lower() for word in stop_words]
    translator = str.maketrans('', '', st.punctuation)
    words = [word.translate(translator) for word in words]
    return(words)

def word_frequency(text):
    freq_dict = defaultdict(int)
    for sentences in tqdm(text):
        for word in sentences.split():
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    return(fd_sorted)

def ngram_tokens(sentence, ngram = 2):
    words = sentence.split()
    return(list(ngrams(words, ngram)))

def ngram_frequency(text, ngram = 2):
    freq_dict = defaultdict(int)
    for sentences in tqdm(text):
        ng_tokens = ngram_tokens(sentences, ngram)
        for ng in ng_tokens:
            index = '_'.join(ng)
            freq_dict[index] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["ngram", "count"]
    return(fd_sorted)

def apply_collocations(text, set_collocation):
    for b1,b2 in set_collocation:
        res = text.replace("%s %s" % (b1 ,b2), "%s_%s" % (b1 ,b2))
    return res

def plot_learning_curve(estimator, X, y, title, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,
        train_sizes=train_sizes, scoring = 'f1')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def comparison_plot(df_1,df_2,col_1,col_2, space, figsize = (10,8), color = 'salmon'):
    ''' Credits to https://www.kaggle.com/arunsankar/key-insights-from-quora-insincere-questions
    '''
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color=color)
    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color=color)

    ax[0].set_xlabel('Word count', size=14, color="black")
    ax[0].set_ylabel('Words', size=14, color="black")
    ax[0].set_title('Top words in sincere questions', size=18, color="black")

    ax[1].set_xlabel('Word count', size=14, color="black")
    ax[1].set_ylabel('Words', size=14, color="black")
    ax[1].set_title('Top words in insincere questions', size=18, color="black")

    fig.subplots_adjust(wspace=space)
    plt.show()

# Getting Concept words by collocations
def get_collocations(text, verbose = True, bigram_freq = True):
    if (verbose):
        print('Word Tokenization...')
    tokens = [t.split() for t in text]
    
    if (verbose):
        print('Making Bigramer Model...')
        
    bigramer = phrases.Phrases(tokens)  # train model with default settings
    
    
    if (bigram_freq):   
        if (verbose):
            print('Making Bigramer list...')
        
        bigram_counter = list()
        bigram_list = list(bigramer.vocab.items())
        for key, value in bigram_list:
            str_key = key.decode()
            if len(str_key.split("_")) > 1:
                bigram_counter.append(tuple([str_key, value]))
        bigram_df = pd.DataFrame(bigram_counter, columns=['bigrams', 'count'])

    
    if (bigram_freq):
        res_dict = {'bigramer': bigramer, 'bigram_freq': bigram_df}
    else:
        res_dict = {'bigramer': bigramer, 'bigram_freq': None}
    
    return(res_dict)

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))

def prepare_corpus(doc_clean):
    """
    Input  : clean document
    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    Output : term dictionary and Document Term Matrix
    """
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # Applying TFIDF to corpus
    tfidf = TfidfModel(doc_term_matrix)
    corpus_tfidf = tfidf[doc_term_matrix]
    return dictionary,corpus_tfidf

def create_gensim_lsa_model(doc_clean,number_of_topics,words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    print('Preparing Corpus with TFIDF...')
    dictionary,doc_term_matrix = prepare_corpus(doc_clean)

    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel

def compute_coherence_values(doc_clean, start=2, stop = 10, step=2, min_doc = 1):
    """
    Input   : dictionary : Gensim dictionary
              corpus : Gensim corpus
              texts : List of input texts
              stop : Max num of topics
    purpose : Compute c_v coherence for various number of topics
    Output  : model_list : List of LSA topic models
              coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    
    print('Preparing Corpus with TFIDF...')
    dictionary,doc_term_matrix = prepare_corpus(doc_clean)
    
    coherence_values = []
    model_list = []
    for num_topics in tqdm(range(start, stop, step)):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def get_topics(model, dtm):
    topics = model[dtm]
    topics = sorted(topics, reverse=True, key=lambda x: x[1])
    topic = 'T' + str(topics[0][0])
    return(topic)
    
def NB_sentimenter(text, classifier, positivity = True):
    sent = classifier(text)
    
    if (positivity):
        get_score = sent.sentiment[1] # positivity
    else:
        get_score = sent.sentiment[2] # negativty

    return(get_score)

def parallel_process(array, function, n_jobs=6, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out


#def threshold_search(y_true, y_proba):
#    'https://www.kaggle.com/ryanzhang/tfidf-naivebayes-logreg-baseline'
#    best_threshold = 0
#    best_score = 0
#    for threshold in [i * 0.01 for i in range(100)]:
#        score = metrics.f1_score(y_true=y_true, y_pred=y_proba > threshold)
#        if score > best_score:
#            best_threshold = threshold
#            best_score = score
#    search_result = {'threshold': best_threshold, 'f1': best_score}
#    return search_result

def threshold_search(y_true, y_proba, plot=False):
    'https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/75735'
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.001) 
    F = 2 / (1/precision + 1/recall)
    best_score = np.max(F)
    best_th = thresholds[np.argmax(F)]
    if plot:
        plt.plot(thresholds, F, '-b')
        plt.plot([best_th], [best_score], '*r')
        plt.show()
    search_result = {'threshold': best_th , 'f1': best_score}
    return search_result 
    
    
def count_tag(text):
    counts = Counter(token[1] for token in text.tags)
    return counts
    
# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(tags, flag):
    pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
    }
    
    get_tags = [tag for tag in tags if tag in pos_family[flag]]
    get_sums = sum([tags[tag] for tag in get_tags])
    return get_sums
    

