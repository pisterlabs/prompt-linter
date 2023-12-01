from kneed import KneeLocator
from sklearn.cluster import KMeans
from pprint import pprint
from gensim.models import Phrases #gensim version 3.6.0
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import pyLDAvis.gensim  #2.1.2
import nltk
from nltk.corpus import stopwords
import re
import string
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from utilities import utils, processing_utils
import pickle
from sklearn.decomposition import PCA
from gensim.models.coherencemodel import CoherenceModel
from sklearn.manifold import TSNE

DATASETS_DIR = '../datasets/'
DATASETS_MED_DIR = '../datasets/Meditation.csv'
DATASETS_LPT_DIR = '../datasets/LifeProTips.csv'
DATASETS_FSP_DIR = '../datasets/Friendship.csv'
DATASETS_DEPRESSION_DIR = '../datasets/Depression.csv'
DATASETS_TRAINING_DATASET_DIR = '../datasets/training_dataset.p'
DATASETS_COVID_DATASET_DIR = '../datasets/the-reddit-covid-dataset-comments.csv.zip'

DATASETS_X_TRAIN_DIR = '../datasets/x_train.p'
DATASETS_Y_TRAIN_DIR = '../datasets/y_train.p'
DATASETS_X_VAL_DIR = '../datasets/x_val.p'
DATASETS_Y_VAL_DIR = '../datasets/y_val.p'
DATASETS_X_TEST_DIR = '../datasets/x_test.p'
DATASETS_Y_TEST_DIR = '../datasets/y_test.p'
DATASETS_COVID_19_DATASET_DIR = '../datasets/covid_19_dataset.p'

DATASETS_X_TEST_CSV_DIR = '../datasets/x_test.csv'
DATASETS_Y_TEST_CSV_DIR = '../datasets/y_test.csv'
DATASETS_MANUAL_LABELED = '../datasets/data_manual_labeled.csv'
DATASETS_COVID_19_DATASET_CSV_DIR = '../datasets/covid_19_dataset.csv'

# directories to experiment results.
EXPERIMENTS_DIR = '../experiments'
EXPERIMENTS_BERT_RESULTS_TEST_100_PREDICTIONS_DIR = '../experiments/fine_tuning_bert_results/test_100_predictions.p'
EXPERIMENTS_BERT_RESULTS_COVID_19_PREDICTIONS_DIR = '../experiments/fine_tuning_bert_results/covid_19_predictions.p'
EXPERIMENTS_COVID_19_DATASET_PREDICTED_CSV_DIR = '../experiments/fine_tuning_bert_results/covid_19_dataset_predicted.csv'
EXPERIMENTS_SIMPLE_CLASSIFIER_DIR = '../experiments/simple_classifier_results'
EXPERIMENTS_LDA_ANALYSIS_DIR = '../experiments/LDA_analysis_results'
EXPERIMENTS_LDA_ANALYSIS_PYLDAVIS_DIR = '../experiments/LDA_analysis_results/pyLDAvis.html'

# path to pre_trained_model
PRETRAINED_MODEL_DIR = '../pre_trained_model'
ZIP_DIR = '../pre_trained_model/enwiki_dbow-20220306T033226Z-001.zip'
PRETRAINED_MODEL_TRAIN_DIR = '../pre_trained_model/train_doc2vec.p'
PRETRAINED_MODEL_TEST_DIR = '../pre_trained_model/test_doc2vec.p'
PRETRAINED_MODEL_COVID_19_DIR = '../pre_trained_model/covid_19_doc2vec.p'
PRETRAINED_MODEL_COVID_19_TFIDF_DIR = '../pre_trained_model/covid_19_tfidf.p'
PRETRAINED_MODEL_X_TFIDF_DIR = '../pre_trained_model/x_tfidf.p'

#Text Cleaning
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
def clean_text(text, flg_stemm=False, flg_lemm=True):
    global stopwords
    text = text.values.tolist()
    stopwords = stopwords.words('english') + stopwords.words('french') + stopwords.words('spanish')
    newStopWords = ['people','you','covid','like','the','http','get']
    stopwords.extend(newStopWords)
    ps = nltk.stem.porter.PorterStemmer()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    text_cleaned = []
    text_cleaned1 = []
    for word in text:
        word = word.lower()
        word = re.sub('\[.*?\]', '', word)
        word = re.sub('https?://\S+|www\.\S+', '', word)
        word = re.sub('<.*?>+', '', word)
        word = re.sub('[%s]' % re.escape(string.punctuation), '', word)
        word = re.sub('\n', '', word)
        word = re.sub('\w*\d\w*', '', word)
        word = re.sub('http','',word)
        word = re.sub('[^A-Za-z0-9.]+', ' ', word)
        tokenized_word = nltk.word_tokenize(word)
        # remove stopwords, numbers, word length less than 1
        lst_text = [w for w in tokenized_word if
                    (len(w) > 2 and w not in stopwords)]

        # Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            lst_text = [ps.stem(word) for word in lst_text]

        # Lemmatisation (convert the word into root word)
        if flg_lemm == True:
            lst_text = [lem.lemmatize(word) for word in lst_text]

        lst_str = ' '.join(lst_text)
        #print(lst_str)
        text_cleaned.append(lst_text)
        text_cleaned1.append(lst_str)
        #print(text_cleaned)

    return text_cleaned, text_cleaned1


def num_topic(X_cv):
    #find the optimal number of clusters
    number_of_clusters = 20
    wcss = []

    for i in range (1, number_of_clusters):
        model = KMeans(n_clusters=i,
        init='k-means++',
        max_iter=20)
        model.fit(X_cv)
        wcss.append(model.inertia_)

    kl = KneeLocator(range(1, 20), wcss, curve="convex", direction="decreasing")
    print(kl.elbow)
    plt.plot(range(1, number_of_clusters), wcss)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    #plt.show()
    path = EXPERIMENTS_LDA_ANALYSIS_DIR + '/optimal topic number'
    plt.savefig(path)
    return kl.elbow

def lda_model(X_LDA, n_topic):

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(X_LDA, min_count=20)
    for idx in range(len(X_LDA)):
        for token in bigram[X_LDA[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                X_LDA[idx].append(token)
    #print(X_LDA)

    dictionary = Dictionary(X_LDA)
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in X_LDA]

    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    #print(dictionary)

    num_topics = n_topic
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    pprint(model.print_topics())
    return corpus, dictionary, model

def get_coherence(text_lda, n_topic):
    results = []
    for n in n_topic:
        corpus, dictionary, model = lda_model(text_lda, n)
        texts = [[dictionary[word_id] for word_id, freq in doc] for doc in corpus]
        cm = CoherenceModel(model=model, corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        results.append(coherence)
    print(results)
    plt.plot(n_topic, results)
    plt.xlabel('Number of clusters')
    plt.ylabel('coherence')
    # plt.show()
    path = EXPERIMENTS_LDA_ANALYSIS_DIR + '/coherence results'
    plt.savefig(path)
    return results


def plot_topic(model, num):

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=nltk.corpus.stopwords.words("english"),
                    background_color='white',
                    width=2500,
                    height=1800,
                    max_words=10,
                    colormap='tab10',
                    color_func=lambda *args, **kwargs: cols[i],
                    prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, int(num/2), figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    #plt.show()
    path = EXPERIMENTS_LDA_ANALYSIS_DIR + '/top words in each topic'
    plt.savefig(path)

def main():

    covid_19_predicted = pd.read_csv(EXPERIMENTS_COVID_19_DATASET_PREDICTED_CSV_DIR)
    #print(covid_19_predicted.info)
    covid_19_depression = covid_19_predicted.loc[covid_19_predicted["predictions"] == 1]
    #print(covid_19_depression.info)

    text = covid_19_depression['body']
    #print(X)
    #clean the dataset
    text_lda, text_tf =clean_text(text)

    #transform to tfidf
    if not os.path.exists(PRETRAINED_MODEL_COVID_19_TFIDF_DIR):
        vectorizer = TfidfVectorizer()
        covid_19_depression_tfidf = vectorizer.fit_transform(text_tf)
        covid_19_depression_tfidf = covid_19_depression_tfidf.toarray()
        pca = PCA(n_components=300)
        covid_19_depression_pca = pca.fit_transform(covid_19_depression_tfidf)
        pickle.dump(covid_19_depression_pca, open(PRETRAINED_MODEL_COVID_19_TFIDF_DIR, 'wb'))

    covid_19_depression_pca = pickle.load(open(PRETRAINED_MODEL_COVID_19_TFIDF_DIR, 'rb'))
    print(covid_19_depression_pca.shape)

    #find the optimal number topic
    n_topic = num_topic(covid_19_depression_pca)
    # n_topic = 8
    topics = [7, 8, 9, 10, 11, 12, 13, 14, 15]

    #LDA model 
    corpus, dictionary, model = lda_model(text_lda, n_topic)
    coherence = get_coherence(text_lda, topics)

    plot_topic(model, n_topic)

    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, EXPERIMENTS_LDA_ANALYSIS_PYLDAVIS_DIR)
    pyLDAvis.show(vis)




if __name__ == '__main__':
    main()