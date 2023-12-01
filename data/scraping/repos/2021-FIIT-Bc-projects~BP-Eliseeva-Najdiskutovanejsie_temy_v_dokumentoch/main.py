# -*- coding: utf-8 -*-
# pip install pyLDAvis

import gensim
import matplotlib
import nltk
import pandas as pd
# import numpy as np
import os
import warnings
import pyLDAvis
import pyLDAvis.gensim_models

from gensim import corpora
from sklearn.datasets import fetch_20newsgroups
from wordcloud import WordCloud
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
from pprint import pprint
import pyforms
from pyforms import BaseWidget
from pyforms.controls import ControlText, ControlButton, ControlFile, ControlNumber, ControlCombo, ControlLabel
import matplotlib.pyplot as plt
import spacy

# import pkg_resources

# print(pkg_resources.get_distribution('gensim').version)
# print(pkg_resources.get_distribution('nltk').version)
# print(pkg_resources.get_distribution('pandas').version)
# print(pkg_resources.get_distribution('spacy').version)
# print(pkg_resources.get_distribution('pyLDAvis').version)
# print(pkg_resources.get_distribution('wordcloud').version)
# print(pkg_resources.get_distribution('matplotlib').version)

global corpus_test
global corpus_train
global stop_words
global nlp
global data_all_lemmatized
global id2word
global settings

def measure_model(model):
    # perplexity
    print(model.log_perplexity(corpus_test))
    # coherence
    coherence_model_lda = CoherenceModel(model=model, texts=data_all_lemmatized, dictionary=id2word, coherence='c_v')
    print(coherence_model_lda.get_coherence())


def ShowWordCloud(data):
    string = ','.join(str(text) for text in data)
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, regexp=r"\w[\w']+\w").generate(string)

    plt.figure(figsize=(30, 30))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)


def remove_stopwords_spacy(texts):
    filtered_texts = []
    for text in texts:
        filtered_sentence = []
        for word in text:
            lexeme = nlp.vocab[word]
            if not lexeme.is_stop:
                filtered_sentence.append(word)
        filtered_texts.append(filtered_sentence)
    return filtered_texts


def remove_stopwords_nltk(texts):
    filtered_texts = []
    for doc in texts:
        filtered_sentence = []
        for word in gensim.utils.simple_preprocess(str(doc)):
            if word not in stop_words:
                filtered_sentence.append(word)
        filtered_texts.append(filtered_sentence)
    return filtered_texts


def twenty_newsgroup_to_list():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    data = {'texts': newsgroups_train.data, 'topic_names': newsgroups_train['target_names'],
            'topics_assigned': newsgroups_train['target']}
    return data


def countWordsTotal(texts):
    total = 0
    for text in texts:
        total += len(text)
    return total


def remove_twosymbols(texts):
    for text in texts:
        for word in text:
            if len(word) <= 2:
                text.remove(word)
    return texts


def prepare_data(data):
    data_words = list(sent_to_words(data))
    # ShowWordCloud(data_words)

    print("Povodny text po tokenizacii, pocet slov:")
    print(countWordsTotal(data_words))
    print(data_words[0][:50])

    data_words_nostops = remove_stopwords_spacy(data_words)
    print("Pocet slov po vymazani stop-slov:")
    print(countWordsTotal(data_words_nostops))
    print(len(data_words_nostops[0]))
    print(data_words_nostops[0][:50])

    data_lemmatized = []
    for text in data_words_nostops:
        doc = nlp(" ".join(text))
        data_lemmatized.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])

    print("Pocet slov po lemmatizacii:")
    print(countWordsTotal(data_lemmatized))
    print(len(data_lemmatized[0]))
    print(data_lemmatized[0][:50])

    data_words_nostops = remove_stopwords_spacy(data_lemmatized)
    print("Pocet slov po opakovanom vymazani stop-slov:")
    print(countWordsTotal(data_words_nostops))
    print(len(data_words_nostops[0]))
    print(data_words_nostops[0][:50])

    bigram = gensim.models.Phrases(data_words_nostops, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    data_bigrams = [bigram_mod[text] for text in data_words_nostops]

    result = remove_twosymbols(data_bigrams)
    print("Pocet slov po vymazani dvojpismenkovych a menej slov:")
    print(countWordsTotal(result))
    print(len(result[0]))
    print(result[0][:50])

    return result


def clean_data(papers):
    papers = papers.drop(columns=['id', 'event_type', 'pdf_name', 'year', 'abstract'], axis=1)
    papers['paper_text_processed'] = papers['paper_text'].str.replace('[ \t\n\r\f\v]', " ")
    papers['paper_text_processed'] = papers['paper_text_processed'].str.replace(r"\S*@\S*\s?", "")
    papers['paper_text_processed'] = papers['paper_text_processed'].str.lower()
    return papers


"""### Custom hyperparameter tuning"""


def hypertuning(corpus, texts, limit=15, start=2, step=2):
    global settings
    model_results = {'num_topics': [],
                     'coherence': []
                     }

    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                chunksize=settings["chunk"],
                                                passes=settings["passes"],
                                                eta=settings["eta"],
                                                alpha=settings["alpha"])
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coh = coherence_model.get_coherence()
        # perpl = model.log_perplexity(corpus_test)
        model_results['num_topics'].append(num_topics)
        model_results['coherence'].append(coh)
        print(coh)
    return model_results


def model_creation():
    print("model creation started")
    global corpus_test
    global corpus_train
    global data_all_lemmatized
    global id2word
    global stop_words
    global nlp
    global settings

    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    papers_all = pd.read_csv(settings["path"])
    papers_all = papers_all[:500]
    papers_all = clean_data(papers_all)
    data_all_lemmatized = prepare_data(papers_all['paper_text_processed'].values)
    ShowWordCloud(data_all_lemmatized)

    id2word = corpora.Dictionary(data_all_lemmatized)

    print(len(id2word))
    id2word.filter_extremes(no_above=0.5, no_below=5)
    print(len(id2word))

    corpus_all = [id2word.doc2bow(text) for text in data_all_lemmatized]
    corpus_train = corpus_all[:int(2 * len(papers_all) / 3)]
    corpus_test = corpus_all[int(2 * len(papers_all) / 3):]

    """## LDA Gensim"""
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_train,
                                                id2word=id2word,
                                                num_topics=settings["topics"],
                                                random_state=100,
                                                chunksize=settings["chunk"],
                                                passes=settings["passes"],
                                                alpha=settings["alpha"],
                                                eta=settings["eta"])
    pprint(lda_model.print_topics())

    measure_model(lda_model)

    # vis = pyLDAvis.gensim_models.prepare(lda_model, corpus_all, id2word)
    # pyLDAvis.display(vis)


def start_hypertuning(start, limit, step):
    print("tuning started")
    global corpus_test
    global corpus_train
    global data_all_lemmatized
    global id2word
    global stop_words
    global nlp
    global settings
    model_results = hypertuning(corpus_train, data_all_lemmatized, limit, start, step)

    best_res = max(model_results['coherence'])
    i = model_results['coherence'].index(best_res)
    best_topic = model_results['num_topics'][i]
    for x in model_results:
        print(x)
        print(model_results[x][i])

    plt.figure()
    plt.plot(model_results['num_topics'], model_results['coherence'])
    plt.xlabel("num_topics")
    plt.ylabel("coherence")
    plt.show()

    eta = [0.1, 0.3, 0.5, 0.7, 0.9, 'symmetric', 'auto']
    eta_model_results = []

    for e in eta:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus_train,
                                                id2word=id2word,
                                                num_topics=best_topic,
                                                random_state=100,
                                                chunksize=100,
                                                passes=2,
                                                eta=e,
                                                alpha='auto')
        coherence_model = CoherenceModel(model=model, texts=data_all_lemmatized, dictionary=id2word, coherence='c_v')
        coh = coherence_model.get_coherence()
        eta_model_results.append(coh)
        print(coh)

    plt.figure()
    plt.plot(eta, eta_model_results)
    plt.xlabel("eta")
    plt.ylabel("coherence")
    plt.show()

    eta_best_res = max(eta_model_results)
    i_eta = eta_model_results.index(eta_best_res)
    best_eta = eta[i_eta]

    best_model = gensim.models.ldamodel.LdaModel(corpus=corpus_train,
                                                 id2word=id2word,
                                                 num_topics=best_topic,
                                                 random_state=100,
                                                 chunksize=100,
                                                 passes=2,
                                                 alpha='auto',
                                                 eta=best_eta)

    measure_model(best_model)


class startMenu(BaseWidget):

    def __init__(self):
        super(startMenu, self).__init__('Start menu')

        # Definition of the forms fields
        self._topicnumber = ControlNumber(label="Topic number", default=2, minimum=2, maximum=15)
        self._chunksize = ControlText(label="Chunk size", default="2000")
        self._passes = ControlNumber(label="Number of passes", default=2, minimum=1, maximum=10)
        self._alpha = ControlCombo("Alpha")
        self._alpha.add_item("symmetric")
        self._alpha.add_item("asymmetric")
        self._alpha.add_item("auto")
        self._eta = ControlCombo(label="Eta")
        eta = [0.1, 0.3, 0.5, 0.7, 0.9, 'symmetric', 'auto']
        for e in eta:
            self._eta.add_item(str(e))
        self._file = ControlFile('Get data')
        self._button = ControlButton('Start LDA algorithm')
        # Define the button action
        self._button.value = self.__buttonAction

        self._topicstart = ControlNumber(label="Topic start", default=2, minimum=1, maximum=15)
        self._topiclimit = ControlNumber(label="Topic limit", default=10, minimum=1, maximum=15)
        self._topicstep = ControlNumber(label="Topic step", default=2, minimum=1, maximum=15)
        self._button2 = ControlButton('Start hyperparameter tuning')
        self._button2.value = self.__button2Action
        self._topicstep.hide()
        self._topicstart.hide()
        self._topiclimit.hide()
        self._button2.hide()


    def __buttonAction(self):
        global settings
        settings = {
            "path": self._file.value,
            "topics": int(self._topicnumber.value),
            "passes": int(self._passes.value),
            "alpha": self._alpha.value
        }
        try:
            float(self._eta.value)
            settings["eta"] = float(self._eta.value)
        except ValueError:
            settings["eta"] = self._eta.value

        if self._chunksize.value.isnumeric():
            settings["chunk"] = int(self._chunksize.value)
        else:
            settings["chunk"] = 2000

        if os.path.isfile(settings["path"]):
            print(settings)
            model_creation()
            self.changeWindow()
        else:
            return

    def changeWindow(self):
        self._topicnumber.hide()
        self._chunksize.hide()
        self._passes.hide()
        self._alpha.hide()
        self._eta.hide()
        self._file.hide()
        self._eta.hide()
        self._button.hide()

        self._topicstep.show()
        self._topicstart.show()
        self._topiclimit.show()
        self._button2.show()

    def __button2Action(self):
        start_hypertuning(int(self._topicstart.value), int(self._topiclimit.value), int(self._topicstep.value))



# Execute the application
if __name__ == "__main__":
    # model_creation("papers.csv")
    pyforms.start_app(startMenu, geometry=(200, 200, 400, 400))
