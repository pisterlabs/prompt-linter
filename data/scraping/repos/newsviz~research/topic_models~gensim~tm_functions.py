# coding: utf-8
import re
from gensim.models.ldamodel import LdaModel

from gensim.models import CoherenceModel, LdaMulticore
import pandas as pd
import warnings
import time

warnings.filterwarnings("ignore")


def text_to_words(text, return_tokenized, lemmatizer, stop_words):
    """
    Обработка текста.

    В текущей версии делает следующее:
    - оставляет только кириллицу;
    - переводит слова в нижний регистр;
    - отбрасывает стоп-слова;
    - лемматизирует;

    Параметры:
    ---------
    text: string
        Текст
    return_tokenized : bool
        Возвращать список токенов или объединенный стринг
    lemmatizer: lemmatizer
        Лемматизатор
    stop_words : list
        Список стоп слов

    Returns:
        Список токенов или объединенный стринг
    """
    letters = re.sub("[^а-яА-Я]", ' ', text)
    words = letters.lower().split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    if not return_tokenized:
        return ' '.join(lemmatized)
    else:
        return lemmatized


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, use_multicore=False):
    """
    Расчёт c_v coherence для разного количества топиков.

    Считает coherence, чтобы по полученному значению оптимизировать количество топиков.

    Параметры:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : []
        Список текстов
    limit : int
        Максимальное количество топиков
    start: int
        Стартовок количество топиков
    step : int
        Шаг увеличения количества топиков
    use_multicore : bool
        Использовать LdaMulticore или нет


    Returns:
    -------
    model_list : []
        Список LDA topic models
    coherence_values : []
        Coherence для натренированных моделей
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(f'Num topics: {num_topics}. {time.ctime()}')
        if use_multicore:
            model = LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=10, per_word_topics=True,
                                 workers=5)
        else:
            model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, per_word_topics=True)

        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def format_topics_sentences(ldamodel, corpus, texts):
    """
    Создаёт DdataFrame с информацией о самом распространённом топике для каждого текста.

    Параметры:
    ldamodel : Натренированная LDA модель
    corpus : Gensim corpus
    texts : []
        Список текстов

    Returns:
    --------
    sent_topics_df : pd.DataFrame

    """
    sent_topics_df = pd.DataFrame()
    # Словарь с топиками и топ-словами для них.
    words_per_topic = {j: [i[0] for i in ldamodel.show_topic(j)] for j in range(ldamodel.num_topics)}

    for i, row in enumerate(ldamodel[corpus]):
        if ldamodel.per_word_topics == False:
            # Самый распространённый топик в каждом тексте
            row_topics = sorted(row, key=lambda x: x[1], reverse=True)

        else:
            row_topics = sorted(row[0], key=lambda x: x[1], reverse=True)

        # Номер топика и его доля
        topic_num, prop_topic = row_topics[0]
        topic_keywords = ', '.join([word for word in words_per_topic[topic_num] if word in texts[i]])
        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                                               ignore_index=True)

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df
