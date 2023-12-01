import pandas as pd
from catboost import CatBoostRegressor

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import saveData, lemmatize


class TopicModel:
    def __init__(self,
                 prepareDF,
                 descriptionRegexPattern: str,
                 vocab: list,
                 oneHotSkills: pd.DataFrame):
        self.resDF = prepareDF.copy()
        self.id2word = None
        self.model = None
        self.encodeCorpus = None
        self.descRP = descriptionRegexPattern
        self.vocab = vocab
        self.oneHotSkills = oneHotSkills

    def fit(self, modelConfig: dict, savePath: str) -> None:
        pass

    def predict(self) -> None:
        pass

    def prepare_input(self) -> None:
        pass

    def inference(self, resume: str) -> (int, list[str]):
        pass

    def model_eval(self, topicTermData: str = './data/descriptionTopics.csv') -> None:
        pass


class LDAmodel(TopicModel):
    def prepare_input(self) -> None:
        text = [text.split() for text in self.resDF.Description.values]

        # higher threshold fewer phrases.
        bigram = gensim.models.Phrases(text, min_count=5, threshold=5)
        trigram = gensim.models.Phrases(bigram[text], threshold=5)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        textPrepData = trigram_mod[bigram_mod[text]]

        self.id2word = corpora.Dictionary(textPrepData)
        self.encodeCorpus = [self.id2word.doc2bow(text) for text in textPrepData]

        self.resDF['VacancyCorpus'] = [word for word in textPrepData]

    def fit(self, modelConfig, savePath ='./models/LDAmodel.pkl') -> None:
        print('Создание модели LDA. Обучение модели...')
        self.model = gensim.models.LdaModel(self.encodeCorpus,
                                            id2word=self.id2word,
                                            **modelConfig)
        saveData(self.model, savePath)

    def predict(self):
        resTopics = self.model.get_document_topics(self.encodeCorpus)
        self.resDF['TopicLabel'] = np.array([i[0][0] if len(i) > 0 else -1 for i in resTopics], dtype=np.int)
        self.resDF['TopicProb'] = [i[0][1] if len(i) > 0 else None for i in resTopics]

    def inference(self, resume):
        important_words = lemmatize(resume, delSymbPattern=self.descRP, tokens=self.vocab)
        assert len(important_words) > 0, \
            'Опишите свои навыки более конкретно (после обработки резюме не было найдено ни одного навыка)'

        ques_vec = []
        ques_vec = self.id2word.doc2bow(important_words.split())

        topic_vec = []
        topic_vec = self.model[ques_vec]
        word_count_array = np.empty((len(topic_vec), 2), dtype=np.object)
        for i in range(len(topic_vec)):
            word_count_array[i, 0] = topic_vec[i][0]
            word_count_array[i, 1] = topic_vec[i][1]

        idx = np.argsort(word_count_array[:, 1])
        idx = idx[::-1]
        word_count_array = word_count_array[idx]

        return word_count_array[0][0], important_words.split()

    def model_eval(self, topicTermData='./data/descriptionTopics.csv'):
        descrTopics = {}
        # Compute Perplexity # a measure of how good the model is. lower the better.
        print('\nPerplexity: ', self.model.log_perplexity(self.encodeCorpus))

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=self.model, texts=self.resDF.VacancyCorpus,
                                             dictionary=self.id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)

        for topic in self.model.show_topics(num_topics=72):
            sks = re.findall(re.compile(r'"\w+"'), topic[1])
            descrTopics[topic[0]] = ' '.join([x[1:-1] for x in sks])

        saveData(pd.Series(descrTopics), topicTermData)


class NMFmodel(TopicModel):
    def prepare_input(self) -> None:
        text = self.resDF.Description.values
        self.vectorizer = TfidfVectorizer(max_features=1000,
                                          max_df=0.997,
                                          min_df=0.003)
        self.encodeCorpus = self.vectorizer.fit_transform(text)

    def fit(self, modelConfig, savePath ='./models/NMFmodel10500.pkl') -> None:
        print('Создание модели NMF. Обучение модели...')
        self.model = NMF(**modelConfig)
        self.model.fit(self.encodeCorpus)
        saveData(self.model, savePath)

    def predict(self):
        # model = loadData(modelPath)
        self.resDF['TopicLabel'] = self.model.transform(self.encodeCorpus).argmax(axis=1).astype(np.int)

    def inference(self, resume):
        important_words = lemmatize(resume, delSymbPattern=self.descRP, tokens=self.vocab)
        assert len(important_words) > 0, \
            'Опишите свои навыки более конкретно (после обработки резюме не было найдено ни одного навыка)'

        preps = important_words.split()
        return self.model.transform(self.vectorizer.transform(preps)).sum(axis=0).argmax(), preps

    def model_eval(self, topicTermData):
        descrTopics = {}
        feature_names = self.vectorizer.get_feature_names_out()
        sns.heatmap(cosine_similarity(self.model.components_)).set(xticklabels=[], yticklabels=[])
        plt.title('Косинусная близость выделенных тематик')
        plt.show()

        for topic_idx, topic_words in enumerate(self.model.components_):
            top_words_idx = topic_words.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            descrTopics[topic_idx + 1] = ' '.join(top_words)

        saveData(pd.Series(descrTopics), topicTermData)


class CatBoostModel:
    def __init__(self, config):
        self.config = config
        self.model = CatBoostRegressor(**config)

    def train(self, X, y, savePath, printLoss=True):

        if not os.path.exists(savePath):
            print('\nОбучение модели предсказания заработной платы...')
            self.model.fit(X, y)
            self.model.save_model(savePath)
            print(f'Модель сохранена в "{savePath}"')
        else:
            print('\nНайдена обученная ранее модель. Загрузка модели...')
            self.model.load_model(savePath)

        self.infer_table = X.groupby(['Schedule', 'Experience'], as_index=False).size()
        if printLoss and not os.path.exists(savePath):
            # print relative RMSE
            print(f"Best model relative RMSEloss: {self.model.best_score_['learn']['RMSE'] / (y.max() - y.min())}")

    def inference(self, resume):
        del self.infer_table['size']
        self.infer_table['Description'] = [resume]*self.infer_table.shape[0]
        self.infer_table['Salary'] = self.model.predict(self.infer_table)
        del self.infer_table['Description']

        return self.infer_table