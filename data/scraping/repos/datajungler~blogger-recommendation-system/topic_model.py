import gensim
import numpy as np
from gensim.models import LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re

import pandas as pd


import nltk
nltk.download('punkt')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from gensim.parsing.preprocessing import remove_stopwords


df = pd.read_csv("us_hiking_review_list.csv")

comments = df['review_content'].str.replace('\\n|\\r', ' ')
comments = comments.str[:500].values


tagged_data = [TaggedDocument(words=word_tokenize(remove_stopwords(_d.lower())), tags=[str(i)]) for i, _d in enumerate(comments)]
model = Doc2Vec(tagged_data, vector_size=5, window=2, min_count=1, workers=4)


W = model[model.wv.vocab]
print(W.shape)

import gensim
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        newStopWords = ['it','they','use','the','this']
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in newStopWords and len(token) > 3:
            nltk.bigrams(token)
            result.append(token)
    return result
	
	
from gensim import corpora
from gensim.utils import simple_preprocess

k = 7
alpha = 0.31
beta = 0.31
mydict = corpora.Dictionary([preprocess(line) for line in comments])
corpus = [mydict.doc2bow(preprocess(line)) for line in comments]
lda = LdaModel(corpus=corpus,id2word=mydict, num_topics=k, alpha=alpha, eta=beta, random_state=111)



num_words = 20
for topic in lda.print_topics(num_words=num_words):
  print(topic)
  
topic_list = list()
from operator import itemgetter
for t in lda.get_document_topics(bow=corpus):
  topic_list.append(max(t, key = itemgetter(1))[0])

print(topic_list)

df['topics'] = topic_list
df.to_csv("topics.csv")

# model coherence
from gensim.models import CoherenceModel
# Tune k
for i in range(1,k+1):
  lda = LdaModel(corpus=corpus,id2word=mydict, num_topics=i)
  coherence_model_lda = CoherenceModel(model=lda, corpus=corpus, dictionary=mydict, coherence='u_mass')
  coherence_lda = coherence_model_lda.get_coherence()
  print('\n# topics: {0}, Coherence Score: {1:.2f}'.format(i, coherence_lda))


# Tune alpha, beta
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')

beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')

for a in alpha:
  for b in beta:
    lda = LdaModel(corpus=corpus,id2word=mydict, num_topics=k, alpha=a, eta=b)
    coherence_model_lda = CoherenceModel(model=lda, corpus=corpus, dictionary=mydict, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\n# Alpha: {0}, Beta: {1}, Coherence Score: {2:.2f}'.format(a, b, coherence_lda))


# PCA: visualize embedding
from sklearn.decomposition import PCA
from matplotlib import pyplot
pca = PCA(n_components=2)
result = pca.fit_transform(W)

pyplot.scatter(result[:, 0], result[:, 1])






