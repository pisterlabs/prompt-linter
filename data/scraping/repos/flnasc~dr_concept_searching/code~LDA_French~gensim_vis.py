"""
   Author: Dylan Hayton-Ruffner
   Description: Runs lda on the given corpus, prints out resulting topics and queries every concept from the concept file.

            Query: If the concept-word exsists in the top 4 words of a topic, all the paragraphs associated with that topic and have
            the concept word are returned

            After each successful query, the results are formated into an excel file and written to the results folder.

   Status: Finished
   ToDo: N/A

   NOTES: Concept path and results path are hard-coded


"""
TOPIC_PRESSENCE_THRESHOLD = 0.3
REGEX_PATTERN = u'(?u)\\b\\w\\w\\w\\w+\\b'
MIN_WORD_COUNT = 10
NUM_TOPICS = 7
TOP_N_SEGS = 10
TOP_N_WORDS = 0
MIN_DF = 0.00
MAX_DF = 1.00
HELP_MESSAGE = "USAGE: <num_topics> <num_iterations> <visualization_file_path (.html)> <corpus_csv_file (.csv)>"
FILETYPE = 'xml'
CONCEPTS_PATH = "../../data/concepts.txt"
import pyLDAvis
import pyLDAvis.gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import numpy as np
import sys
import random
from operator import itemgetter
from nltk import word_tokenize
# from elbow_criteria import threshold
# from elbow_criteria import limit_by_threshold
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import gensim


import matplotlib.pyplot as plt
import csv


############################# MAIN #############################

def main():
    print("\n-----LDA CONCEPT DETECTION-----")

    # check command line
    if len(sys.argv) != 4:
        print(HELP_MESSAGE)
        quit(1)

    if not sys.argv[0].isdigit():
        print(HELP_MESSAGE)
        print("<num_topics> must be numeric")

    if not sys.argv[1].isdigit():
        print(HELP_MESSAGE)
        print("<num_iterations> must be numeric")

    if not sys.argv[2].endswith(".html"):
        print(HELP_MESSAGE)
        print("<visualization_file_path> must end with '.html'")

    if not sys.argv[3].endswith(".csv"):
        print(HELP_MESSAGE)
        print("<corpus_csv_file> must end with '.csv'")

    num_topics = sys.argv[0]
    num_iter = sys.argv[1]
    vis_file_path = sys.argv[2]
    corpus_csv_file = sys.argv[3]

    # load corpus
    corpus = load_from_csv(corpus_csv_file)

    # create CountVectorizer to get help remove short segments
    stop_words = load_stop_words("../../data/stopwords-fr.txt")
    vectorizer = CountVectorizer(lowercase=True, max_df=MAX_DF, min_df=MIN_DF, token_pattern=r"(?u)\b\w\w\w+\b")

    # remove short segments from the corpus
    proc_corpus, proc_corpus_text_only = remove_short_segs(corpus, vectorizer)
    proc_corpus_text_only = [seg.split() for seg in proc_corpus_text_only]

    # remove stop words from the corpus
    proc_stop_words = []
    for i in range(len(proc_corpus_text_only)):
        proc_stop_words.append([])
        for j in range(len(proc_corpus_text_only[i])):
            if proc_corpus_text_only[i][j] not in stop_words and len(proc_corpus_text_only[i][j]) >= 3:
                proc_stop_words[i].append(proc_corpus_text_only[i][j])

    # vectorize text with gensim's Dictionary
    id2word = Dictionary(proc_stop_words)
    corp = [id2word.doc2bow(text) for text in proc_stop_words]

    # run mallet lda model
    path_to_mallet_binary = "Mallet/bin/mallet"
    mallet_model = LdaMallet(path_to_mallet_binary, corpus=corp, num_topics=13, id2word=id2word, optimize_interval=20,
                             random_seed=4, iterations=1000)

    # convert to gensim model to build visualization
    gensim_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet_model)
    vis = pyLDAvis.gensim.prepare(gensim_model, corp, id2word)

    # save visualization
    pyLDAvis.save_html(vis, "sa_visualization.html")

    return 0


############################# LOAD DATA #############################
def load_from_csv(path):
    """
    Loads all the segments from a csvfile.
    :param path: string, path to csvfile
    :return: list, a list of all the segments
    """
    segs = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        for row in reader:
            segs.append(row)
    return segs


def remove_short_segs(corpus, vectorizer):
    """
    Remove the short segments from the corpus i.e. less than min word count.
    :param corpus: list, a list of all text segments
    :param vectorizer: CountVectorizer object, built for french
    :return: proc_corpus, a list of all text segments with # of words > min word count
    """

    proc_corpus = []
    proc_corpus_text_only = []
    for seg in corpus:
        id = seg[0]
        text = seg[1]
        vec = vectorizer.fit_transform([text])
        if vec.shape[1] > MIN_WORD_COUNT:
            proc_corpus.append([id, text])
            proc_corpus_text_only.append(text)

    return proc_corpus, proc_corpus_text_only


def load_stop_words(path):
    """
    Loads the stop words from txt file
    :param path: string, path to text file
    :return: list, list of stop words
    """
    stop_words = []
    with open(path) as txtfile:
        for line in txtfile:
            stop_words.append(line.strip().lower())
    return stop_words











if __name__ == "__main__":
    main()
