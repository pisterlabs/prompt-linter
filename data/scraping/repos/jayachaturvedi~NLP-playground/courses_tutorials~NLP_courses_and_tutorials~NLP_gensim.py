## https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python

from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import pandas as pd

path = r'C:\Users\K1774755\PycharmProjects\aurelie_NLP\NLP'
data = "All work and no play makes jack a dull boy. all work and no play"

phrases = sent_tokenize(data)
words = word_tokenize(data)

print(phrases)
print(words)

### LATENT SEMANTIC INDEXING
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def open_file(path,file_name):
    try: file = open(os.path.join(path, file_name), encoding="utf8")
    except: file = open(os.path.join(path, file_name), errors='ignore')
    return file

def load_data(path, file_name):
    """
    Input  : path and file_name
    Purpose: loading text file
    Output : list of paragraphs/documents and
             title(initial 100 words considred as title of document)
    """
    documents_list = []
    titles = []
    with open_file(path,file_name) as fin:
        for line in fin.readlines():
            text = line.strip()
            documents_list.append(text)
    print("Total Number of Documents:", len(documents_list))
    titles.append(text[0:min(len(text), 100)])
    return documents_list, titles


def preprocess_data(doc_set):
    """
    Input  : documents list
    Purpose: preprocess text (tokenize, removing stopwords, and stemming)
    Output : preprocessed text
    """
    tokenizer = RegexpTokenizer(r'\w+')  # initialize regex tokenizer
    en_stop = set(stopwords.words('english'))  # create English stop words list
    p_stemmer = PorterStemmer()  # Create p_stemmer of class PorterStemmer
    texts = []  # list for tokenized documents in loop

    for i in doc_set:  # loop through document list
        tokens = tokenizer.tokenize(i.lower())  # clean and tokenize document string
        stopped_tokens = [i for i in tokens if not i in en_stop]  # remove stop words from tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]  # stem tokens
        texts.append(stemmed_tokens)  # add tokens to list
    return texts


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
    # generate LDA model
    return dictionary, doc_term_matrix


def create_gensim_lsa_model(doc_clean, number_of_topics, words):
    """
    Input  : clean document, number of topics and number of words associated with each topic
    Purpose: create LSA model using gensim
    Output : return LSA model
    """
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel


##DETERMINE NUMBER OF TOPICS
def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
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
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print("coherence value =",coherencemodel.get_coherence(),"for #topics=", num_topics)
    return model_list, coherence_values


def plot_graph(doc_clean, start, stop, step):
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix, doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

def run_example():
    number_of_topics, words, start, stop, step = [2, 5, 2, 12, 1]
    [documents_list, titles] = load_data(path, 'articles.txt')
    doc_clean = preprocess_data(documents_list)
    [dictionary, doc_term_matrix] = prepare_corpus(doc_clean)
    [model_list, coherence_values] = compute_coherence_values(dictionary, doc_term_matrix, doc_clean,
                                                              stop=5, start=2,step=1)

    lsamodel = create_gensim_lsa_model(doc_clean=doc_clean, number_of_topics=2, words=3)
    plot_graph(doc_clean, start=2, stop=12, step=3)