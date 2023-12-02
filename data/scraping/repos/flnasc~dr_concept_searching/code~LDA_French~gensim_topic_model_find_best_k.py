"""
   Author: Dylan Hayton-Ruffner
   Description: Runs lda on the given corpus, prints out resulting topics and queries every concept from the concept file.
            Usage: change the CORPUS path variable to specify a corpus. Run from commandline with: python3 gensim_topic_model.py

            After each successful query, the results are formated into an excel file and written to the results folder.

   Status: Finished
   ToDo: N/A

   NOTES: Concept path and results path are hard-coded


"""
TOPIC_PRESSENCE_THRESHOLD = 0.3
REGEX_PATTERN = u'(?u)\\b\\w\\w\\w\\w+\\b'
MIN_WORD_COUNT = 100
NUM_TOPICS = 7
TOP_N_SEGS = 10
TOP_N_WORDS = 0
MIN_DF = 0.00
MAX_DF = 1.00
CORPUS_PATH = "data/lemmatized_segments/symb-du-mal-full-lemma.csv"
import csv
import sys
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from elbow_criteria import threshold
from elbow_criteria import limit_by_threshold
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from statistics import mean, median, stdev



############################# MAIN #############################

def main():
    print("\n-----LDA CONCEPT DETECITON-----")
    corpus = load_from_csv(CORPUS_PATH)

    # Create CountVectorizer to get Document-Term matrix

    stop_words = load_stop_words("data/stopwords-fr.txt")
    vectorizer = CountVectorizer(lowercase=True, max_df=MAX_DF, min_df=MIN_DF, token_pattern=r"(?u)\b\w\w\w+\b")

    proc_corpus, proc_corpus_text_only = remove_short_segs(corpus, vectorizer)
    proc_corpus_text_only = [seg.split() for seg in proc_corpus_text_only]
    proc_stop_words = []

    for i in range(len(proc_corpus_text_only)):
        proc_stop_words.append([])
        for j in range(len(proc_corpus_text_only[i])):
            if proc_corpus_text_only[i][j] not in stop_words and len(proc_corpus_text_only[i][j]) >= 3:
                proc_stop_words[i].append(proc_corpus_text_only[i][j])

    # train vectorizer on corpus

    id2word = Dictionary(proc_stop_words)
    corp = [id2word.doc2bow(text) for text in proc_stop_words]

    # print("Number of Features: " + str(len(feature_names)))

    # initialize model
    path_to_mallet_binary = "/Users/fnascime/Dev/mallet/mallet-2.0.8/bin/mallet"

    coherence_values = []

    for my_k in range(14,15):

        mallet_model = LdaMallet(path_to_mallet_binary, corpus=corp, num_topics=my_k, id2word=id2word, optimize_interval=20,
                                 random_seed=9, iterations=5000)

        gensim_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(mallet_model)
        coherencemodel = CoherenceModel(model=gensim_model, texts=proc_stop_words, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())


    max = 0
    best_k= 999
    for index, coherence in enumerate(coherence_values) :

        print ("K: ", 11 + index, " -> Coherence: ", coherence)
        if coherence > max:
            max = coherence
            best_k = index

    print (" *** Summary ***")
    print (" Best K     : ", best_k)
    print ("Best coherence : ", max)
    print ("Median         : ", median(coherence_values))
    print ("Mean           : ", mean(coherence_values))
    #print ("Stdev          : ", stdev(coherence_values))


    #doc_topics = list(mallet_model.read_doctopics(mallet_model.fdoctopics(), renorm=False))
    #topic_word = TopicWord(mallet_model)
    #topic_word.get_topic_word()
    #topic_word.write_to_csv("output/topic_" +str(mallet_model.random_seed) + "_" + str(mallet_model.iterations) + "_" + str(mallet_model.num_topics) + ".csv")

    #topic_doc = TopicDoc(mallet_model)
    #topic_doc.get_topic_doc()
    #topic_doc.write_to_csv("output/topic_doc"+str(mallet_model.random_seed)+ "_" + str(mallet_model.iterations)+ "_" + str(mallet_model.num_topics) + ".csv", num_docs=50)

    return 0


class TopicWord:

    def __init__(self, mallet_model):
        self.model = mallet_model
        self.topic_word = None

    def get_topic_word(self):
        topics = self.model.show_topics(num_topics=self.model.num_topics, formatted=False)
        self.topic_word = topics

    def write_to_csv(self, path):
        with open(path, "w+") as csv_file:
            writer = csv.writer(csv_file)
            for topic in self.topic_word:
                row = [topic[0]]
                row.extend([word[0] for word in topic[1]])
                writer.writerow(row)


class TopicDoc:

    def __init__(self, mallet_model):
        self.model = mallet_model
        self.topic_doc = None

    def get_topic_doc(self):
        doc_topic = list(self.model.read_doctopics(self.model.fdoctopics(), renorm=False))
        topic_doc = [[] for i in range(self.model.num_topics)]
        for i in range(len(doc_topic)):
            doc = doc_topic[i]
            for topic in doc:
                topic_doc[topic[0]].append((i, topic[1]))

        for topic in topic_doc:
            topic = topic.sort(key=itemgetter(1), reverse=True)

        self.topic_doc = topic_doc

    def write_to_csv(self, path, num_docs=10):
        #IDS RETURNED ARE LOCAL TO MALLET NO GLOBAL CORPUS NUMBERING
        with open(path, "w+") as csv_file:
            writer = csv.writer(csv_file)
            count = 0
            for topic in self.topic_doc:
                row = [count]
                row.extend([doc[0] for doc in topic[:num_docs]])
                writer.writerow(row)
                count += 1







def write_concepts_csv(concepts):
    with open("../../data/concepts_sm1.csv", "w+") as csv_file:
        writer = csv.writer(csv_file)
        for concept in concepts:
            row = [concept[0]]
            row.extend([seg[0] for seg in concept[1]])
            writer.writerow(row)


def get_topic_word(topics):
    threshold_topic_word = []
    topics = [[topic[0], topic[1]] for topic in topics]
    # sort and reverse
    for i in range(len(topics)):
        freq = []
        words = topics[i][1]
        for word in words:
            freq.append(word[1])
        thresh = threshold(freq)
        for j in range(len(words)):
            if words[j][1] < thresh:
                topics[i][1] = topics[i][1][:j]
                break
    return topics


def get_topic_doc(doc_topics, num_topics, proc_corpus):
    assert len(doc_topics) == len(proc_corpus), "output from mallet is of different length than input corpus"

    topic_doc = {}

    for i in range(num_topics):
        topic_doc[i] = []

    # each document
    for i in range(len(doc_topics)):

        # each topic in each document
        for topic in doc_topics[i]:

            if topic[0] in topic_doc:
                topic_doc[topic[0]].append(
                    (proc_corpus[i][0], topic[1]))  # append (the ID of the segment in the corpus, and the topic
                # relevance to that segment)

    for topic in topic_doc:

        topic_doc[topic] = sorted(topic_doc[topic], key=(itemgetter(1)), reverse=True)
        pseudocount_list = [seg[1] for seg in topic_doc[topic]]
        thresh = threshold(pseudocount_list)
        new_doc = []
        for i in range(len(topic_doc[topic])):
            if topic_doc[topic][i][1] > thresh:
                new_doc.append(topic_doc[topic][i])

        topic_doc[topic] = new_doc

    return topic_doc


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


def parse_concepts(concepts_raw):
    lines = concepts_raw.split('\n')
    concepts = []
    for line in lines:
        if len(line.split()) > 0:
            concepts.append(line.split())
    return concepts


def load_corpus_txt(c_size):
    # works with a raw text file
    filepath = input("Filepath to corpus: ")
    print("LOADING FILE: " + filepath)
    doc_string = load_document(filepath)
    vectorizer = CountVectorizer(stop_words='english', lowercase=True)
    preproc = vectorizer.build_analyzer()
    proc_text = preproc(doc_string)

    count = -1
    corpus_i = 0
    corpus = []
    n = c_size
    final = [proc_text[i * n:(i + 1) * n] for i in range((len(proc_text) + n - 1) // n)]

    return final


############################# Test Elbow Algorithm #############################


def run_elbow(model, feature_names):
    """Prints the topic information. Takes the sklearn.decomposition.LatentDiricheltAllocation lda model,
    the names of all the features, the number of words to be printined per topic, a list holding the freq
    of each topic in the corpus"""
    print("Elbow Limited Topics:")
    message_list = []

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % (topic_idx)

        # get the names of the features in sorted order -> argsort() return sorted indicies
        list_feat = [feature_names[i]
                     for i in topic.argsort()[::-1]]  # [::-1] reverses list

        # get the frequencis of the top words (limited by the threshold function)
        feat_freq = sorted(topic, reverse=True)
        cutoff = threshold(sorted(topic, reverse=True))
        limited_freq = limit_by_threshold(feat_freq, cutoff)

        for j in range(len(limited_freq)):
            message += "%s: %s, " % (str(list_feat[j]), str(limited_freq[j]))

        message_list.append(message)
        print(message)
    print()

    return message_list


if __name__ == "__main__":
    main()
