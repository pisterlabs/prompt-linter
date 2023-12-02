import imp
import re

import numpy as np
import pandas as pd
import tqdm
from nltk.corpus import stopwords

import gensim
import gensim.corpora as corpora
import spacy
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess


def get_model_and_opt_num_of_topics(yearly_dataset, limit, step=1, start=2):
    
    words = list(tools.sentences_to_words(yearly_dataset['abstract'].values))
    lemmatized_words = tools.clean_words(words, stop_words)
    
    id2word = corpora.Dictionary(lemmatized_words)
    texts = lemmatized_words
    corpus = [id2word.doc2bow(text) for text in texts]
    
    model_list, coherence_values = tools.compute_coherence_values(limit=limit,
                                                                mallet_path=mallet_path,
                                                                dictionary=id2word,
                                                                corpus=corpus,
                                                                texts=lemmatized_words,
                                                                step=step, start=start)
    
    max_coherence_value = max(coherence_values)
    max_index = coherence_values.index(max_coherence_value)
    num_of_topics = list(range(start, limit, step))

    return model_list[max_index], num_of_topics[max_index], corpus


def get_dataframe(ldamodel,
                   corpus):
    data_with_topics = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0: 
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                data_with_topics = data_with_topics.append(pd.Series([int(topic_num),
                                                                  round(prop_topic,4),
                                                                  topic_keywords]),
                                                         ignore_index=True)
            else:
                break
    data_with_topics.columns = ['Dominant_Topic',
                                'Perc_Contribution',
                                'Topic_Keywords']

    return data_with_topics


tools = imp.load_source('tools', 'src/lda_tools.py')


df = pd.read_csv('src/data/prisoners_dilemma_articles_meta_data_clean.csv')

data = df[['abstract', 'unique_key', 'title', 'date']]
data = data.drop_duplicates()
data = data.reset_index(drop=True)

stop_words = stopwords.words('english')
mallet_path = '/Users/storm/rsc/mallet-2.0.8/bin/mallet'


years = sorted(data.date.unique())
periods = np.linspace(min(years), max(years), 10)
periods = periods[2:]

results = []
for year in tqdm.tqdm(periods):
    if year == int(2018):
        yearly_data = data
    else:
        yearly_data = data[data['date'] <= int(year)]

    model, index, corpus = get_model_and_opt_num_of_topics(yearly_data, limit=15, start=3)
    
    df =  get_dataframe(model, corpus)
    
    df['class'] = year

    filename = "src/data/topics_up_to_{}.csv".format(int(year))
    
    df.to_csv(filename)
