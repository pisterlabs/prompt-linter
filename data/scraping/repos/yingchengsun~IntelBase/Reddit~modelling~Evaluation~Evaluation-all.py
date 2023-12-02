'''
Created on Aug 11, 2018

@author: yingc
'''
from gensim.models import ldamodel
from gensim.corpora import Dictionary
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

import numpy as np
from pprint import pprint
df_fake = pd.read_csv('fake - Copy.csv')
df_fake[['title', 'text', 'language']].head()
df_fake = df_fake.loc[(pd.notnull(df_fake.text)) & (df_fake.language == 'english')]

# remove stopwords and punctuations
def preprocess(row):
    return strip_punctuation(remove_stopwords(row.lower()))
    
df_fake['text'] = df_fake['text'].apply(preprocess)

# Convert data to required input format by LDA
texts = []
for line in df_fake.text:
    lowered = line.lower()
    words = re.findall(r'\w+', lowered, flags = re.UNICODE | re.LOCALE)
    texts.append(words)

dictionary = Dictionary(texts)

training_texts = texts[:50]
holdout_texts = texts[50:75]
test_texts = texts[75:100]

'''
training_corpus = [dictionary.doc2bow(text) for text in training_texts]
holdout_corpus = [dictionary.doc2bow(text) for text in holdout_texts]
test_corpus = [dictionary.doc2bow(text) for text in test_texts]

from gensim.models.callbacks import CoherenceMetric, DiffMetric, PerplexityMetric, ConvergenceMetric

# define perplexity callback for hold_out and test corpus
pl_holdout = PerplexityMetric(corpus=holdout_corpus, logger="visdom", title="Perplexity (hold_out)")
pl_test = PerplexityMetric(corpus=test_corpus, logger="visdom", title="Perplexity (test)")

# define other remaining metrics available
ch_umass = CoherenceMetric(corpus=training_corpus, coherence="u_mass", logger="visdom", title="Coherence (u_mass)")
ch_cv = CoherenceMetric(corpus=training_corpus, texts=training_texts, coherence="c_v", logger="visdom", title="Coherence (c_v)")
diff_kl = DiffMetric(distance="kullback_leibler", logger="visdom", title="Diff (kullback_leibler)")
convergence_kl = ConvergenceMetric(distance="jaccard", logger="visdom", title="Convergence (jaccard)")

callbacks = [pl_holdout, pl_test, ch_umass, ch_cv, diff_kl, convergence_kl]

# training LDA model
model = ldamodel.LdaModel(corpus=training_corpus, id2word=dictionary, num_topics=35, passes=50, chunksize=150, iterations=200, alpha='auto', callbacks=callbacks)

# to get a metric value on a trained model
print(CoherenceMetric(corpus=training_corpus, coherence="u_mass").get_value(model=model))
'''