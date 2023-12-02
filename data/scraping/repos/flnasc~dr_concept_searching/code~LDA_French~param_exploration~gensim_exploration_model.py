
# VARY WORDS TOPICS
REGEX_PATTERN = u'(?u)\\b\\w\\w\\w\\w+\\b'
MIN_DF = 0.00
MAX_DF = 1.00
CORPUS_PATH = "../data/lemmatized_segments/symb-du-mal-full-lemma.csv"
import csv
import sys
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

############################# MAIN #############################


def topic_model(W, K, N):
    """

    :param w: min_number of words per segment
    :param k: number of topics
    :param n: number of iterations
    :return:
    """
    print("\n-----LDA CONCEPT DETECITON-----")
    print('MODEL:', hash((W,K,N)), W, K, N)
    corpus = load_from_csv(CORPUS_PATH)

    # Create CountVectorizer to get Document-Term matrix

    stop_words = load_stop_words("../data/stopwords-fr.txt")
    vectorizer = CountVectorizer(lowercase=True, max_df=MAX_DF, min_df=MIN_DF, token_pattern=r"(?u)\b\w\w\w+\b")

    proc_corpus, proc_corpus_text_only = remove_short_segs(corpus, vectorizer, W)
    proc_corpus_text_only = [seg.split() for seg in proc_corpus_text_only]
    proc_stop_words = []

    for i in range(len(proc_corpus_text_only)):
        proc_stop_words.append([])
        for j in range(len(proc_corpus_text_only[i])):
            if proc_corpus_text_only[i][j] not in stop_words and len(proc_corpus_text_only[i][j]) >= 3:
                proc_stop_words[i].append(proc_corpus_text_only[i][j])

    # train vectorizer on corpus
    print('Corpus Size:', len(proc_stop_words))
    id2word = Dictionary(proc_stop_words)
    corp = [id2word.doc2bow(text) for text in proc_stop_words]

    # print("Number of Features: " + str(len(feature_names)))


    # redirect stdout for capturing LL/token

    # initialize model
    path_to_mallet_binary = "../mallet_git/bin/mallet"

    mallet_model = LdaMallet(path_to_mallet_binary, corpus=corp, num_topics=K, id2word=id2word, optimize_interval=20,
                             random_seed=9, iterations=N)

    u_mass = CoherenceModel(model=mallet_model, texts=proc_stop_words, corpus=corp, coherence='u_mass')
    c_v = CoherenceModel(model=mallet_model, texts=proc_stop_words, corpus=corp, coherence='c_v')
    c_uci = CoherenceModel(model=mallet_model, texts=proc_stop_words, corpus=corp, coherence='c_uci')
    c_npmi = CoherenceModel(model=mallet_model, texts=proc_stop_words,  corpus=corp, coherence='c_npmi')

    u_mass_val = u_mass.get_coherence()
    c_v_val = c_v.get_coherence()
    c_uci_val = c_uci.get_coherence()
    c_npmi_val = c_npmi.get_coherence()

    print('U_MASS_VAL:', u_mass_val)
    print('C_V_VAL:', c_v_val)
    print('C_UCI_VAL:', c_uci_val)
    print('C_NPMI_VAL:', c_npmi_val)

    return 0

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


def remove_short_segs(corpus, vectorizer, w):
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
        if vec.shape[1] > w:
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
    if len(sys.argv) < 4:
        print('USAGE W [min words per seg] K [num topics] N [num iterations]')
        exit(1)
    W = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    topic_model(W, K, N)

