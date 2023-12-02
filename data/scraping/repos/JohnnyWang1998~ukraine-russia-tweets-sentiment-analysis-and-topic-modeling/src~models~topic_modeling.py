import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel, ldamodel


def generate_daily_topic(text_dict: dict, topic_num: int, learning_decay: float):
    """
    Generate daily topic model
    text_dict: {'date': [text]}
    topic_num: number of topics
    learning_decay: learning rate decay
    return: {'date': [topic_prob]}
    """

    # initilize array to store topics
    topic_table = np.empty([len(text_dict), topic_num], dtype=object)

    # loop every day data
    row = 0
    for date, text_list in tqdm(text_dict.items()):

        vectorizer = CountVectorizer(
            analyzer='word',
            min_df=3,  # minimum required occurences of a word
            lowercase=True,  # convert all words to lowercase
            token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
            max_features=5000,  # max number of unique words
        )
        data_matrix = vectorizer.fit_transform(text_list)

        lda_model = LatentDirichletAllocation(
            n_components=topic_num,
            learning_decay=learning_decay,
            learning_method='online',
            random_state=20,
            n_jobs=-1  # Use all available CPUs
        )
        lda_model.fit_transform(data_matrix)

        for i, topic in enumerate(lda_model.components_):
            topics = str([vectorizer.get_feature_names()[i]
                         for i in topic.argsort()[-5:]])
            topic_table[row, i] = topics
        row += 1

    # return it as a dataframe
    topic_df = pd.DataFrame(topic_table)
    topic_df.index = list(text_dict.keys())
    topic_df.columns = ['topic_'+str(i) for i in range(topic_num)]
    return topic_df


def compute_coherence_values(dictionary, corpus, id2word, texts, limit, start=3, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, random_state=1)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
