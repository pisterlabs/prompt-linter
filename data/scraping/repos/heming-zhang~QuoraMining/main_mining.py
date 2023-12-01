from pythondb import DataBase
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from pyLDAvis import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from biterm.btm import oBTM
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from scipy import optimize 
import matplotlib.pyplot as plt
import pyLDAvis
import spacy
import datetime
import gensim
import nltk
import collections
import re
import string
import numpy as np


def f_1(x, A, B):  
    return A*x + B  
def f_2(x, A, B, C):
    return A*x*x + B*x + C
##############################
# Doc Classify
def ori_classify_doc():
    alltextlist = []
    films_answertext = DataBase(0, '0', '0', 0, '0', '0', 0).select_answertext()
    for text in films_answertext: 
        answercount = int(text[1])
        date = str(text[2])
        question = str(text[4])
        view_weight = (int(text[5]) // 2000) + 1 # weight to improve topic mining
        answer = str(text[6])
        datenorm = datetime.datetime.strptime(date, '%Y/%m/%d')
        datenum = int(datenorm.strftime('%Y%m%d'))
        # change datenum limitation to control weeks
        if datenum >= 20190408:
            alltextlist.append(question) # use append or just use '+' to aggregate string
            if answercount > 0 : 
                # for i in range(view_weight): # use view_weight to add weight
                    alltextlist.append(answer)
            # print(date, question, view_weight, answer)
    fw = open('./textfiles/Ori All Text from Apr2, 2019.txt', 'w')
    for text in alltextlist:
        text = text + "\n"
        fw.write(text)
    # print(alltextlist)
    return alltextlist

def classify_doc():
    alltextlist = []
    films_answertext = DataBase(0, '0', '0', 0, '0', '0', 0).select_answertext()
    for text in films_answertext: 
        answercount = int(text[1])
        date = str(text[2])
        question = str(text[4])
        view_weight = (int(text[5]) // 50) + 1 # weight to improve topic mining
        answer = str(text[6])
        datenorm = datetime.datetime.strptime(date, '%Y/%m/%d')
        datenum = int(datenorm.strftime('%Y%m%d'))
        # change datenum limitation to control weeks
        if datenum >= 20190408:
            alltextlist.append(question) # use append or just use '+' to aggregate string
            if answercount > 0 : 
                # for i in range(view_weight): # use view_weight to add weight
                    alltextlist.append(answer)
            # print(date, question, view_weight, answer)
    fw = open('./textfiles/All Text with Weight from Apr2, 2019.txt', 'w')
    for text in alltextlist:
        text = text + "\n"
        fw.write(text)
    # print(alltextlist)
    return alltextlist

def loop_classify_doc(weight):
    weight = weight
    alltextlist = []
    films_answertext = DataBase(0, '0', '0', 0, '0', '0', 0).select_answertext()
    for text in films_answertext: 
        answercount = int(text[1])
        date = str(text[2])
        question = str(text[4])
        view_weight = (int(text[5]) // weight) + 1 # weight to improve topic mining
        answer = str(text[6])
        datenorm = datetime.datetime.strptime(date, '%Y/%m/%d')
        datenum = int(datenorm.strftime('%Y%m%d'))
        # change datenum limitation to control weeks
        if datenum >= 20190513 and datenum < 20190602:
            alltextlist.append(question) # use append or just use '+' to aggregate string
            if answercount > 0 : 
                for i in range(view_weight): # use view_weight to add weight
                    alltextlist.append(answer)
            # print(date, question, view_weight, answer)
    fw = open('./textfiles/Dec2, 2019.txt', 'w')
    for text in alltextlist:
        text = text + "\n"
        fw.write(text)
    # print(alltextlist)
    return alltextlist
##############################



def text_clean_set():
    stop = stopwords.words('english')
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    return stop, exclude, lemma

def text_clean(text):
    stop, exclude, lemma= text_clean_set()
    # free numbers
    num_free = re.sub(r'\d+', '', text)
    # free English abbreviation
    abr_free0 = re.sub("’s", "", num_free)
    abr_free1 = re.sub("’re", "", abr_free0)
    abr_free2 = re.sub("’ve", "", abr_free1)
    abr_free3 = re.sub("’m", "", abr_free2)
    abr_free4 = re.sub("n’t", "", abr_free3)
    # free Chinese puncuation
    punc_free0 = re.sub("–", "", abr_free4)
    punc_free1 = re.sub("—", "", punc_free0)
    punc_free2 = re.sub("\…", "", punc_free1)
    punc_free3 = re.sub("\‘", "", punc_free2)
    punc_free4 = re.sub("\’", "", punc_free3)
    punc_free5 = re.sub("\“", "", punc_free4)
    punc_free6 = re.sub("\”", "", punc_free5)
    # English free
    stop_free = " ".join([i for i in punc_free6.lower().split() if i not in stop])
    punc_free = "".join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized



##############################
# Text clean run
def ori_text_clean_run():
    temp_text = ""
    text_cleaned0 = [] # text_cleaned0 has only one list
    text_cleaned1 = []
    alltextlist = ori_classify_doc()
    text_cleaned2 = [text_clean(text).split() for text in alltextlist] 
    fw = open('./textfiles/Ori Cleaned Text from Apr2, 2019.txt', 'w')
    for list1 in text_cleaned2:
        for list2 in list1:
            text_cleaned0.append(list2)
            if list2 == "": continue
            else:
                list2 = list2 + "\t"
                fw.write(list2)
        temp_text = " ".join(list1)
        text_cleaned1.append(temp_text)
    # print(text_cleaned1) # text_cleaned1 has many single word in lists
    # fr = open('./textfiles/Apr2, 2019.txt', 'r')
    # print(fr.read())
    # print(text_cleaned0) # text_cleaned2 has many lists in list
    return text_cleaned2, text_cleaned1, text_cleaned0

def text_clean_run():
    temp_text = ""
    text_cleaned0 = [] # text_cleaned0 has only one list
    text_cleaned1 = []
    alltextlist = classify_doc()
    text_cleaned2 = [text_clean(text).split() for text in alltextlist] 
    fw = open('./textfiles/Cleaned Text with Weight from Apr2, 2019.txt', 'w')
    for list1 in text_cleaned2:
        for list2 in list1:
            text_cleaned0.append(list2)
            if list2 == "": continue
            else:
                list2 = list2 + "\t"
                fw.write(list2)
        temp_text = " ".join(list1)
        text_cleaned1.append(temp_text)
    # print(text_cleaned1) # text_cleaned1 has many single word in lists
    # fr = open('./textfiles/Apr2, 2019.txt', 'r')
    # print(fr.read())
    # print(text_cleaned0) # text_cleaned2 has many lists in list
    return text_cleaned2, text_cleaned1, text_cleaned0

def loop_text_clean_run(weight):
    weight = weight
    temp_text = ""
    text_cleaned0 = [] # text_cleaned0 has only one list
    text_cleaned1 = []
    alltextlist = loop_classify_doc(weight)
    text_cleaned2 = [text_clean(text).split() for text in alltextlist] 
    fw = open('./textfiles/Apr2, 2019.txt', 'w')
    for list1 in text_cleaned2:
        for list2 in list1:
            text_cleaned0.append(list2)
            if list2 == "": continue
            else:
                list2 = list2 + "\t"
                fw.write(list2)
        temp_text = " ".join(list1)
        text_cleaned1.append(temp_text)
    # print(text_cleaned1) # text_cleaned1 has many single word in lists
    # fr = open('./textfiles/Apr2, 2019.txt', 'r')
    # print(fr.read())
    # print(text_cleaned0) # text_cleaned2 has many lists in list
    return text_cleaned2, text_cleaned1, text_cleaned0
##############################




def get_wordfrequency():
    text_cleaned2, text_cleaned1, text_cleaned0  = text_clean_run()
    print(collections.Counter(text_cleaned0))


##############################
# LDA Models
def lda_model(num_topics):
    num_topics = num_topics
    text_cleaned2, text_cleaned1, text_cleaned0 = text_clean_run()
    dictionary = corpora.Dictionary(text_cleaned2)
    # dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_cleaned2]
    # using Bag of Words
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = dictionary, iterations=50)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    # Compute Perplexity
    perplexity_lda = ldamodel.log_perplexity(doc_term_matrix)
    print('\nPerplexity: ', perplexity_lda)  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=ldamodel, texts=text_cleaned2, dictionary=dictionary, coherence='c_v') # u_mass
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    return perplexity_lda, coherence_lda

def tfidf_model(num_topics):
    num_topics = num_topics
    text_cleaned2, text_cleaned1, text_cleaned0  = text_clean_run()
    dictionary = corpora.Dictionary(text_cleaned2)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_cleaned2]
    # using TF-IDF
    Lda = gensim.models.ldamodel.LdaModel
    tfidf = models.TfidfModel(doc_term_matrix)
    corpus_tfidf = tfidf[doc_term_matrix]
    ldamodel_tfidf = Lda(corpus_tfidf, num_topics=num_topics, id2word = dictionary, iterations=50)
    print(ldamodel_tfidf.print_topics(num_topics=num_topics, num_words=10))

    # Compute Perplexity
    perplexity_tfidf = ldamodel_tfidf.log_perplexity(doc_term_matrix)
    print('\nPerplexity: ', perplexity_tfidf)  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_tfidf = CoherenceModel(model=ldamodel_tfidf, texts=text_cleaned2, dictionary=dictionary, coherence='c_v')
    coherence_tfidf = coherence_model_tfidf.get_coherence()
    print('\nCoherence Score: ', coherence_tfidf)
    # visualize the LDA results
    # vis = pyLDAvis.gensim.prepare(ldamodel_tfidf, corpus_tfidf, dictionary)
    # pyLDAvis.save_html(vis, './pictures/tf-idf lda.html')
    return perplexity_tfidf, coherence_tfidf
##############################



##############################
# BTM Models
def ori_btm_model(num_topics):
    num_topics = num_topics
    # texts = open('./textfiles/Ori-Apr2, 2019.txt').read().splitlines()
    text_cleaned2, texts, text_cleaned0 = ori_text_clean_run()
    # vectorize texts
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()
    # get vocabulary
    vocab = np.array(vec.get_feature_names())
    # get biterms
    biterms = vec_to_biterms(X)
    # create btm
    btm = oBTM(num_topics = num_topics, V = vocab)
    print("\n\n Train Online BTM ..")
    for i in range(0, 1): # prozess chunk of 200 texts
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=10)
    
    print("\n\n Topic coherence ..")
    res, C_z_sum = topic_summuary(btm.phi_wz.T, X, vocab, 20)
    Coherence_mean = C_z_sum/num_topics
    print(Coherence_mean)

    # topics = btm.transform(biterms)
    # print("\n\n Visualize Topics ..")
    # vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    # pyLDAvis.save_html(vis, './textfiles/online_btm.html')
    
    # print("\n\n Texts & Topics ..")
    # for i in range(len(texts)):
        # print(topics[i].argmax())
        # print("{} (topic: {})".format(texts[i], topics[i].argmax()))
    return Coherence_mean

def btm_model(num_topics):
    num_topics = num_topics
    # texts = open('./textfiles/Ori-Apr2, 2019.txt').read().splitlines()
    text_cleaned2, texts, text_cleaned0 = text_clean_run()
    # vectorize texts
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()
    # get vocabulary
    vocab = np.array(vec.get_feature_names())
    # get biterms
    biterms = vec_to_biterms(X)
    # create btm
    btm = oBTM(num_topics = num_topics, V = vocab)
    print("\n\n Train Online BTM ..")
    for i in range(0, 1): # prozess chunk of 200 texts
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=10)
    
    print("\n\n Topic coherence ..")
    res, C_z_sum = topic_summuary(btm.phi_wz.T, X, vocab, 20)
    Coherence_mean = C_z_sum/num_topics
    print(Coherence_mean)

    topics = btm.transform(biterms)
    # print("\n\n Visualize Topics ..")
    # vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    # pyLDAvis.save_html(vis, './textfiles/online_btm.html')
    
    num_list = []
    result = {}
    print("\n\n Texts & Topics ..")
    for i in range(len(texts)):
        print(topics[i].argmax())
        num_list.append(topics[i].argmax())
        print("{} (topic: {})".format(texts[i], topics[i].argmax()))
    num_array = np.array(num_list)
    for i in set(num_list):
        result[i] = num_list.count(i)
    print(result)

    return Coherence_mean

def loop_btm_model(num_topics, weight):
    num_topics = num_topics
    weight = weight
    # texts = open('./textfiles/Ori-Apr2, 2019.txt').read().splitlines()
    text_cleaned2, texts, text_cleaned0 = loop_text_clean_run(weight)
    # vectorize texts
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()
    # get vocabulary
    vocab = np.array(vec.get_feature_names())
    # get biterms
    biterms = vec_to_biterms(X)
    # create btm
    btm = oBTM(num_topics = num_topics, V = vocab)
    print("\n\n Train Online BTM ..")
    for i in range(0, 1): # prozess chunk of 200 texts
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=10)
    
    print("\n\n Topic coherence ..")
    res, C_z_sum = topic_summuary(btm.phi_wz.T, X, vocab, 20)
    Coherence_mean = C_z_sum/num_topics
    print(Coherence_mean)

    # topics = btm.transform(biterms)
    # print("\n\n Visualize Topics ..")
    # vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    # pyLDAvis.save_html(vis, './textfiles/online_btm.html')
    
    # print("\n\n Texts & Topics ..")
    # for i in range(len(texts)):
        # print(topics[i].argmax())
        # print("{} (topic: {})".format(texts[i], topics[i].argmax()))
    return Coherence_mean
##############################



def kmeans_model():
    num_clusters = 5
    km = KMeans(n_clusters = num_clusters)
    text_cleaned2, text_cleaned0 = text_clean_run()
    dictionary = corpora.Dictionary(text_cleaned2)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in text_cleaned2]
    # using TF-IDF and Clusterclear
    tfidf = models.TfidfModel(doc_term_matrix)
    corpus_tfidf = tfidf[doc_term_matrix]
    km.fit(text_cleaned0)
    clusters = km.labels_.tolist()
    print(clusters)



##############################
# Plots
def lda_plot(epoch_time):
    epoch_time = epoch_time
    perplexity_ldas = []
    coherence_ldas = []
    for num_topics in range(1, epoch_time + 1):
        perplexity_lda, coherence_lda = lda_model(num_topics)
        perplexity_ldas.append(perplexity_lda)
        coherence_ldas.append(coherence_lda)

    plt.subplot(211)
    X = range(1, epoch_time + 1)
    plt.plot(X, coherence_ldas, label = "LDA-Coherence")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence")
    plt.title("LDA-Coherence-Graph")
    plt.legend()
    plt.tight_layout()

    plt.subplot(212)
    X = range(1, epoch_time + 1)
    plt.plot(X, perplexity_ldas, label = "LDA-Perplexity")
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    plt.title("LDA-Perplexity-Graph")
    plt.legend()
    plt.tight_layout()

    plt.show()
    plt.savefig("./pictures/LDA Coherence.png")

def tfidf_lda_plot(epoch_time):
    epoch_time = epoch_time
    perplexity_tldas = []
    coherence_tldas = []
    for num_topics in range(1, epoch_time + 1):
        perplexity_tlda, coherence_tlda = tfidf_model(num_topics)
        perplexity_tldas.append(perplexity_tlda)
        coherence_tldas.append(coherence_tlda)

    plt.subplot(211)
    X = range(1, epoch_time + 1)
    plt.plot(X, coherence_tldas, label = "Tf Idf-LDA-Coherence")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence")
    plt.title("Tf Idf-LDA-Coherence-Graph")
    plt.legend()
    plt.tight_layout()

    plt.subplot(212)
    X = range(1, epoch_time + 1)
    plt.plot(X, perplexity_tldas, label = "Tf Idf-LDA-Perplexity")
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    plt.title("Tf Idf-LDA-Perplexity-Graph")
    plt.legend()
    plt.tight_layout()

    plt.show()
    plt.savefig("./pictures/Tf-Idf LDA Coherence.png")

def comparison_plot(epoch_time):
    epoch_time = epoch_time
    coherence_ldas = []
    coherence_tfidfs = []
    perplexity_ldas = []
    perplexity_tfidfs = []
    for num_topics in range(1, epoch_time + 1):

        perplexity_lda, coherence_lda = lda_model(num_topics)
        perplexity_ldas.append(perplexity_lda)
        coherence_ldas.append(coherence_lda)

        perplexity_tfidf, coherence_tfidf = tfidf_model(num_topics)
        perplexity_tfidfs.append(perplexity_tfidf)
        coherence_tfidfs.append(coherence_tfidf)

    X = range(1, epoch_time + 1)
    plt.plot(X, perplexity_ldas, label = "LDA-Perplexity")
    plt.plot(X, perplexity_tfidfs, '^', label = "Tf-Idf_LDA-Perplexity")
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    plt.title("LDA-Perplexity-Graph")
    plt.legend()
    plt.show()
    plt.savefig("./pictures/LDA-Perplexity.png")

def btm_plot(epoch_time):
    epoch_time = epoch_time
    fifty_coherence_btms = []
    hundred_coherence_btms = []
    thousand_coherence_btms = []
    twothousand_coherence_btms = []
    fivethousand_coherence_btms = []
    ori_coherence_btms = []
    for num_topics in range(1, epoch_time + 1):

        fifty_coherence_btm = loop_btm_model(num_topics, 50)
        fifty_coherence_btms.append(fifty_coherence_btm)

        hundred_coherence_btm = loop_btm_model(num_topics, 100)
        hundred_coherence_btms.append(hundred_coherence_btm)

        thousand_coherence_btm = loop_btm_model(num_topics, 1000)
        thousand_coherence_btms.append(thousand_coherence_btm)

        twothousand_coherence_btm = loop_btm_model(num_topics, 2000)
        twothousand_coherence_btms.append(twothousand_coherence_btm)

        fivethousand_coherence_btm = loop_btm_model(num_topics, 5000)
        fivethousand_coherence_btms.append(fivethousand_coherence_btm)

        ori_coherence_btm = ori_btm_model(num_topics)
        ori_coherence_btms.append(ori_coherence_btm)

    X = range(1, epoch_time + 1)
    plt.plot(X, fifty_coherence_btms, color = 'red', label = "BTM-Coherence-Weight 50")
    plt.plot(X, hundred_coherence_btms, color = 'orange', label = "BTM-Coherence-Weight 100")
    plt.plot(X, thousand_coherence_btms, color = 'green', label = "BTM-Coherence-Weight 1000")
    plt.plot(X, twothousand_coherence_btms, color = 'blue', label = "BTM-Coherence-Weight 2000")
    plt.plot(X, fivethousand_coherence_btms, color = 'yellow', label = "BTM-Coherence-Weight 5000")
    plt.plot(X, ori_coherence_btms,'-^', color = 'blueviolet', label = "Ori_BTM-Coherence")
    plt.xticks(np.arange(0, 21, 2))
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence")
    plt.title("BTM-Coherence-Graph")
    plt.legend()
    plt.show()
    plt.savefig("./pictures/BTM-Coherence.png")

def loop_weight_btm_plot(num_topics, epoch_time, step):
    num_topics = num_topics
    epoch_time = epoch_time
    step = step
    coherence_loop_btms = [] #
    ori_coherence_btms = []
    for weight in range(1, epoch_time + 1, step):

        coherence_loop_btm = loop_btm_model(num_topics, weight)
        coherence_loop_btms.append(coherence_loop_btm)

        ori_coherence_btm = ori_btm_model(num_topics) #
        ori_coherence_btms.append(ori_coherence_btm) #

    X = range(1, epoch_time + 1, step)
    Y = coherence_loop_btms
    # line fitting
    # A2, B2, C2 = optimize.curve_fit(f_2, X, Y)[0]
    # X1 = np.arange(0, epoch_time, 0.01) 
    # Y1 = A2*X1*X1 + B2*X1 + C2
    A1, B1 = optimize.curve_fit(f_1, X, Y)[0]
    X1 = np.arange(0, epoch_time, 0.01)
    Y1 = A1*X1 + B1
    plt.plot(X1, Y1, color = 'orange', label = "BTM_Fitting-Coherence")
    plt.plot(X, coherence_loop_btms, '--', label = "BTM-Coherence")
    plt.plot(X, ori_coherence_btms, '-^', color = 'blueviolet' , label = "Ori_BTM-Coherence") #
    plt.xlabel("Weight/Ori_BTM-Epoch")
    plt.ylabel("Coherence")
    plt.title("BTM-Coherence-Graph")
    plt.legend()
    plt.show()
    plt.savefig("./pictures/BTM-Coherence.png")
##############################



if __name__ == "__main__":
    num_topics = 4
    # lda_model(num_topics)
    # tfidf_model(num_topics)
    btm_model(num_topics)
    # kmeans_model()
    # get_wordfrequency()
    
    # epoch_time = 20
    # lda_plot(epoch_time)
    # tfidf_lda_plot(epoch_time)
    # btm_plot(epoch_time)
    # comparison_plot(epoch_time)
    # loop_weight_btm_plot(num_topics, epoch_time, step = 50)

