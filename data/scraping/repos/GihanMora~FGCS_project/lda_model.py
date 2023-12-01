import json
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
# spacy for lemmatization
import spacy
import pandas as pd
# Enable logging for gensim
import logging

from nltk import ngrams

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
def get_ngrams(tokens, n ):
    n_grams = list(ngrams(tokens, n))
    print(n_grams)
    return [ ' '.join(grams) for grams in n_grams]
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def sent_to_words(sentences):#split sentences to words and remove punctuations

    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



def remove_stopwords(texts):#remove stopwords to do more effective extraction
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):#lemmatize words to get core word

    nlp = spacy.load('en', disable=['parser', 'ner'])
    nlp.max_length = 150000000
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def run_lda_model(posts,number_of_topics):#this will extract paragraph and header text from given json file and extract the topics from that
    print("lda model started")

    # print(posts)

    data_words = list(sent_to_words(posts))
    # print(data_words)
    # print("words_list",len(data_words))
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    # print(data_words)

    print('remove_punctuations...')
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # print(data_words_nostops)
    print("words_list_no stop", len(data_words_nostops))
    data_words_bigrams = make_bigrams(data_words_nostops,bigram_mod)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print('data_lemmatized...')
    tri_g = []
    for each in data_lemmatized:
        tr = get_ngrams(each, 3)
        tri_g.append(tr)
    # print("words_list_lemma", len(data_lemmatized))
    # Create Dictionary
    # id2word = corpora.Dictionary(data_lemmatized)
    id2word = corpora.Dictionary(tri_g)
    # print('id2word',id2word)
    # Create Corpus
    texts = data_lemmatized
    print('lemms',texts)

    # Term Document Frequency
    # corpus = [id2word.doc2bow(text) for text in texts]
    corpus = [id2word.doc2bow(text) for text in tri_g]
    # print('corpus',corpus)
    # View
    print('corpus is created')#(word,frequency of occuring)
    topics = []
    try:
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=number_of_topics,
                                                passes=5,
                                                alpha='auto')
        print('topics are extracting')
        topics = lda_model.print_topics()

    except ValueError:#handling exceptions if corpus is empty
        print("corpus is empty or not valid")

    # print(topics)
    import os
    os.environ.update({'MALLET_HOME': r'C:/new_mallet/mallet-2.0.8/'})
    mallet_path = 'C:/new_mallet/mallet-2.0.8/bin/mallet'  # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)
    print('mallet',ldamallet.show_topics(formatted=False))
    words_list = {'Topic_' + str(i): [word for word, prob in lda_model.show_topic(i, topn=10)] for i in range(0, lda_model.num_topics)}
    mallet_list ={'Topic_' + str(i): [word for word, prob in ldamallet.show_topic(i, topn=10)] for i in range(0, ldamallet.num_topics)}
    print("lda",words_list)
    print("mallet",mallet_list)
    return words_list

df = pd.read_csv("../Data/ANXIETY_all_posts.csv", encoding='utf8')
# print(df['post'][0])
anxiety_post_set = []
for each_p in df['post']:
    anxiety_post_set.append(each_p)

run_lda_model(anxiety_post_set,10)
#To run this scrpit individually use following line and run the script
# topics = run_lda_model(path to the json object,number_of_topics)
# print(topics)
# run_lda_model("F://Armitage_project/crawl_n_depth/extracted_json_files/www.axcelerate.com.au_0_data.json",3)