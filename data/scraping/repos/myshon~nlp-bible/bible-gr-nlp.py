
import re
from unicodedata import name
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import glob
import string
import gensim
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases, LdaModel
from gensim.models.ldamulticore import LdaMulticore
import pandas as pd
from num2words import num2words
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sbn 
import matplotlib.pyplot as plt
from scipy.stats import hmean
from scipy.stats import norm
from greek_accentuation.characters import *

import unidecode


import os 
output = "books"
if not os.path.exists(output):
    os.makedirs(output)

from pathlib import Path

data = [] 
book_names = []
ifiles = glob.glob("books/SBLGNTtxt/*.txt")
for ifile in ifiles: 
    book = open(ifile, "r", encoding='utf-8').read().strip() 
    name = Path(ifile).with_suffix('').name
    book_names.append(name)
    data.append(book)

stop_words = stopwords.words('greek')
#stop_words = list(map(unidecode.unidecode, stop_words))
stop_words = list(map(strip_accents, stop_words))
stop_words = list(map(lambda s:s.translate(str.maketrans('', '', string.punctuation)), stop_words))
stop_words = list(map(lambda s:s.translate(str.maketrans('', '', string.punctuation)), stop_words))
#stop_words.extend([
#"autou", "autois", "touto", "auten", "touton", "tauta", "tou","mou", "moi","umou", "emas", "umon", "emoi", 
#"emou", "sou", "soi", "pas", "panta", "pantas", "pante", "pantes", "pasai", "pollous", "en", "umeis"])
stop_words = list(dict.fromkeys(stop_words))

for i, book in enumerate(data, 0):
    # remove NUMBER:NUMBER. pattern at the beginning
    text_index = data[i].find("\n")
    data[i] = data[i][text_index:]
    data[i] = re.sub(r"\w{1,} \d{1,}\:\d{1,}", "", data[i])
    # remove new lines 
    data[i] = re.sub('\s+', " ", data[i]) 
    # remove new line
    data[i] = re.sub("\n", " ", data[i])
    #lower case 
    data[i] = data[i].lower() 
    ## remove accents
    #data[i] = unidecode.unidecode(data[i])
    data[i] = strip_accents(data[i])
    # remove punctuation 
    data[i] = data[i].translate(str.maketrans('', '', string.punctuation))
    # remove stopwords 
    tokens = data[i].split()
    without_stopwords = ' '.join([word for word in tokens if word not in stop_words])
    data[i] = without_stopwords

all_books = ''
i = 1
for pbook in data:
    if i > 4: # only gospels first 4 books
        break
    all_books += pbook + " "
    i += 1
    
# WORDCLOUD
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(all_books)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Zipf’s law plot
# preprocessing on data
# data is a list of all the Bible's books 

# # call the CountVectorizer
# cvec = CountVectorizer()
# # fit transform as we're working directly on all the corpus
# cvec.fit_transform(data)
# # np matrix sparse
# all_df = cvec.transform(data)
# # create a dataframe: sum on all the term occurrences
# tf = np.sum(all_df,axis=0)
# # remove an axis from the tf
# tf2 = np.squeeze(np.asarray(tf))
# # thus we can transform it as a Dataframe
# term_freq_df = pd.DataFrame([tf2],columns=cvec.get_feature_names_out()).transpose()
# # create the plot
# # 0 is the counts
# counts = term_freq_df[0]
# # index the words
# tokens = term_freq_df.index
# # ranks is the position of the word 
# ranks = np.arange(1, len(counts)+1)
# indices = np.argsort(-counts)
# # grab the frequencies
# frequencies = counts[indices]
# # plot figure
# plt.figure(figsize=(15,15))
# # set limits
# plt.ylim(1,10**4.1)
# plt.xlim(1,10**4.1)
# # log log plot
# plt.loglog(ranks, frequencies, marker=".")
# # draw a line to highligh zipf's expected behaviour
# plt.plot([1,frequencies[0]],[frequencies[0],1],color='r')
# plt.xlabel("Frequency rank of token", fontsize=20)
# plt.ylabel("Absolute frequency of token", fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.grid(True)
# # add the text
# for n in list(np.logspace(-0.5, np.log10(len(counts)-2), 25).astype(int)):
#     dummy = plt.text(ranks[n], frequencies[n],
#                      " " + tokens[indices[n]],
#                      verticalalignment="bottom",
#                      horizontalalignment="left",
#                      fontsize=20)


# plt.savefig("ziplaw.png")
# print("completed")


# check for top-level environment


def format_topics_sentences(ldamodel, corpus):
    r"""This function associate to each review the dominant topic
    Parameters
    ----------
    lda_model:      gensim lda_model
                    The current lda model calculated
    corpus:         gensim corpus
                    this is the corpus from the reviews
    texts:          list
                    list of words of each review
    real_text:      list 
                    list of real comments 
                    
    Return
    ------
    """
    
    topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]): #from the corpus rebuild the reviews
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num) #topic + weights
                topic_keywords = ", ".join([word for word, prop in wp]) #topic keyword only
                #prop_topic is the percentage of similarity of that topic
                topics_df = topics_df.append(pd.Series([int(topic_num),\
                                            round(prop_topic,2), topic_keywords]), ignore_index=True)
                #round approximate the prop_topic to 2 decimals
            else:
                break

    topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    return(topics_df)


def run_lda(cleaned_comments, num_topics, chunksize):
    r"""This is the main function which computes the LDA
    Parameters
    ----------
    cleaned_comments: "further_cleaning" in dataframe 
    comments: "comments" in dataframe 
    save_path: option whhere to save the output 
    num_topic: number of topics 
    chunksize: the chunk size of each comment
    Return
    ------
    lda_model:          Gensim
                        LDA model
    """

    #tokenize
    data_words = []
    for sentence in cleaned_comments:
        data_words.append(simple_preprocess(str(sentence),deacc=True))#deacc remove punctuation

    # Create Dictionary
    id2word = Dictionary(data_words)  #this create an index for each word
    #e.g. id2word[0] = "allowed"
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]#bag of words
    #corpus gives the frequency of a word in a document (a document == a single review)
    # Build LDA model

    #creation of lda with X topics and representation
    print("Computing LDA with  {} topics, {} chunksize...".format(num_topics, chunksize))
    # gensim.models.ldamodel.LdaModel
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                       id2word=id2word,
                       num_topics=num_topics,
                       random_state=42,
                       eval_every=100,
                       chunksize=chunksize,
                       passes=5,
                       iterations=400,
                       per_word_topics=True)

    print("Writing classification onto csv file...")
    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus)
    print("Topic Keywords")
    print(df_topic_sents_keywords["Topic_Keywords"].unique())
    print(f"Perplexity {lda_model.log_perplexity(corpus)}")
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Coherence {coherence_lda}")
    return lda_model, df_topic_sents_keywords, corpus

num_topics = [2, 3,  4, 5]
chunksizes = [20, 50, 100]


# if __name__ == '__main__':
#     for num_topic in num_topics:
#         for chunksize in chunksizes:
#             print(f"!!!!!! Num Topic {num_topic} and chunksize {chunksize}")
#             lda_model, df_lda, corpus = run_lda(data, num_topic, chunksize)


from sklearn.manifold import TSNE
from collections import Counter
from six.moves import cPickle
import gensim.models.word2vec as w2v
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import os
import sys
import io
import re
import json
import nltk
nltk.download('punkt')


# use the documents' list as a column in a dataframe 
df = pd.DataFrame(data, columns=["text"]) 


def get_word2vec(text): 
    r"""
    Parameters
    -----------
    text: str, text from dataframe, df['text'].tolist()"""
    num_workers = multiprocessing.cpu_count()
    num_features = 200# tune this bit
    epoch_count = 10
    tokens = [nltk.word_tokenize(sentence) for sentence in text]
    sentence_count = len(text) 
    word2vec = None
    word2vec = w2v.Word2Vec(sg=1,
                            seed=1,
                            workers=num_workers,
                        #    size=num_features,
                            min_count=min_frequency_val,
                            window=5,
                            sample=0)

    print("Building vocab...")
    word2vec.build_vocab(tokens)
    print("Word2Vec vocabulary length:", len(word2vec.wv))
    print("Training...")
    word2vec.train(tokens, total_examples=sentence_count, epochs=epoch_count)
    print("Saving model...")
    word2vec.save(w2v_file)
    return word2vec
  
# save results in an analysis folder 
save_dir = "analysis"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# set up a minimal frequency value
min_frequency_val=6 
# save the word2vec in a file 
w2v_file = os.path.join(save_dir, "word_vectors.w2v")  
word2vec = get_word2vec(df['text'].tolist())


def get_word_frequencies(text):
    r""" This function return a Counter with the most common words
    in a given text
    
    Parameters
    ----------
    text: df['text'].tolist()
    
    Return 
    ------
    freq: Counter, most common words with their freqs
    """
    frequencies = Counter()
    tokens = [nltk.word_tokenize(sentence) for sentence in text]
    for token in tokens:
        for word in token:
            frequencies[word] += 1
    freq = frequencies.most_common()
    return freq

def most_similar(input_word, num_similar):
    r""" This function uses word2vec to find the most 
    num_similar similar words wrt a given 
    input_word
    
    Parameters
    -----------
    input_word: str, input word 
    num_similar: int, how many similar words we want to get
    
    Return 
    ------
    output: list, input word and found words
    """
    
    sim = word2vec.wv.most_similar(input_word, topn=num_similar)
    output = []
    found = []
    for item in sim:
        w, n = item
        found.append(w)
    output = [input_word, found]
    return output


def calculate_t_sne(word2vec):
    r""" Main function to copmute the t-sne representation 
    of the computed word2vec
    """
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    dim0 = word2vec.wv[vocab].shape[1]
    arr = np.empty((0, dim0), dtype='f')
    labels = []
    vectors_file = os.path.join(save_dir, "vocab_vectors.npy")
    labels_file = os.path.join(save_dir, "labels.json")

    print("Creating an array of vectors for each word in the vocab")
    for count, word in enumerate(vocab):
        if count % 50 == 0:
            print_progress(count, vocab_len)
        w_vec = word2vec[word]
        labels.append(word)
        arr = np.append(arr, np.array([w_vec]), axis=0)
    save_bin(arr, vectors_file)
    save_json(labels, labels_file)

    x_coords = None
    y_coords = None
    x_c_filename = os.path.join(save_dir, "x_coords.npy")
    y_c_filename = os.path.join(save_dir, "y_coords.npy")
    print("Computing T-SNE for array of length: " + str(len(arr)))
    tsne = TSNE(n_components=2, random_state=1, verbose=1)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    print("Saving coords.")
    save_bin(x_coords, x_c_filename)
    save_bin(y_coords, y_c_filename)
    return x_coords, y_coords, labels, arr

  
def t_sne_scatterplot(word):
    r""" Function to plot the t-sne result for a given word 
    Parameters
    ----------
    word: list, given word we want to plot the w2v-tsne plot + its neighbours
    """
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    vocab_elems = [key for key in vocab]
    dim0 = word2vec.wv[vocab_elems[0]].shape[0]
    arr = np.empty((0, dim0), dtype='f')
    w_labels = [word]
    # check all the similar words around 
    nearby = word2vec.wv.similar_by_word(word, topn=num_similar)
    arr = np.append(arr, np.array([word2vec[word]]), axis=0)
    for n in nearby:
        w_vec = word2vec[n[0]]
        w_labels.append(n[0])
        arr = np.append(arr, np.array([w_vec]), axis=0)

    tsne = TSNE(n_components=2, random_state=1)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    plt.rc("font", size=16)
    plt.figure(figsize=(16, 12), dpi=80)
    plt.scatter(x_coords[0], y_coords[0], s=800, marker="o", color="blue")
    plt.scatter(x_coords[1:], y_coords[1:], s=200, marker="o", color="red")

    for label, x, y in zip(w_labels, x_coords, y_coords):
        plt.annotate(label.upper(), xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()-50, x_coords.max()+50)
    plt.ylim(y_coords.min()-50, y_coords.max()+50)
    filename = os.path.join(plot_dir, word + "_tsne.png")
    plt.savefig(filename)
    plt.close()

    
def test_word2vec(test_words):
    r""" Function to check if a test word exists within our vocabulary
    and return the word along with its most similar, thorugh word 
    embeddings
    
    Parameters
    ----------
    test_words: str, given word to check
    
    Return 
    ------
    output: list, input word and associated words 
    """
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    output = []
    associations = {}
    test_items = test_words
    for count, word in enumerate(test_items):
        if word in vocab:
            print("[" + str(count+1) + "] Testing: " + word)
            if word not in associations:
                associations[word] = []
        similar = most_similar(word, num_similar)
        t_sne_scatterplot(word)
        output.append(similar)
        for s in similar[1]:
            if s not in associations[word]:
                associations[word].append(s)
            else:
                print("Word " + word + " not in vocab")
    filename = os.path.join(save_dir, "word2vec_test.json")
    save_json(output, filename)
    filename = os.path.join(save_dir, "associations.json")
    save_json(associations, filename)
    filename = os.path.join(save_dir, "associations.csv")
    handle = io.open(filename, "w", encoding="utf-8")
    handle.write(u"Source,Target\n")
    for w, sim in associations.items():
        for s in sim:
            handle.write(w + u"," + s + u"\n")
    return output

  
def show_cluster_locations(results, labels, x_coords, y_coords):
    r""" function to retrieve the cluster location from t-sne 
    Parameters
    ----------
    results: list, word and its neighbours 
    labels: words
    x_coords, y_coords: float, x-y coordinates 2D plane 
    """
    for item in results:
        name = item[0]
        print("Plotting graph for " + name)
        similar = item[1]
        in_set_x = []
        in_set_y = []
        out_set_x = []
        out_set_y = []
        name_x = 0
        name_y = 0
        for count, word in enumerate(labels):
            xc = x_coords[count]
            yc = y_coords[count]
            if word == name:
                name_x = xc
                name_y = yc
            elif word in similar:
                in_set_x.append(xc)
                in_set_y.append(yc)
            else:
                out_set_x.append(xc)
                out_set_y.append(yc)
        plt.figure(figsize=(16, 12), dpi=80)
        plt.scatter(name_x, name_y, s=400, marker="o", c="blue")
        plt.scatter(in_set_x, in_set_y, s=80, marker="o", c="red")
        plt.scatter(out_set_x, out_set_y, s=8, marker=".", c="black")
        filename = os.path.join(big_plot_dir, name + "_tsne.png")
        plt.savefig(filename)
        plt.close()


# if __name__ == '__main__':
#     x_coords, y_coords, labels, arr = calculate_t_sne(word2vec)
#     # and let's save the t-sne plots with the words clusters 
#     frequencies = get_word_frequencies(df['text'].tolist())
#     # check the first 50 most frequent words and see if they're in the w2v
#     for item in frequencies[:50]:
#         test_words.append(item[0])
#     results = test_word2vec(test_words)
#     # and once we have all the word + neighbors let's see how the t-sne has grouped them 
#     show_cluster_locations(results, labels, x_coords, y_coords)