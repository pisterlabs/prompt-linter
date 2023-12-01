def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn #Just to ignore some annoying deprecation warnings
import os
import random
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pandas as pd
from gensim import corpora
import numpy as np
import matplotlib.pyplot as plt
import gensim
import spacy
import warnings
from pprint import pprint
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet

nlp_model = spacy.load('en_core_web_sm')
nlp_model.max_length = 1000000000
SEED = 1962 #For reproducing the results
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

"""Now, let us load the dataframe, and let us drop empty rows"""

df = pd.read_csv('./df_review_complete.csv')
df = df.dropna(subset=["tokens"], inplace=True).reset_index()

data_words = np.array(df['tokens'])
del df #Clear some memory
bigram = gensim.models.Phrases(data_words, min_count=10)
trigram = gensim.models.Phrases(bigram[data_words])
for idx in range(len(data_words)):
    for token in bigram[data_words[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            data_words[idx].append(token)
    for token in trigram[data_words[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            data_words[idx].append(token)
data_words = [d.split() for d in data_words]
id2word = corpora.Dictionary(data_words)
id2word.filter_extremes(no_below=10, no_above=0.2) #Filtering frequent words
corpus = [id2word.doc2bow(text) for text in data_words]


from gensim.models.wrappers import LdaMallet
os.environ.update({'MALLET_HOME':r'./mallet'})
mallet_path = './mallet/bin/mallet' # update this path


lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=14, id2word=id2word, random_seed = SEED)

lda_model_vis = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model) #Forced to convert the model to display the graph below
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model_vis, corpus, id2word)
doc_lda = lda_model_vis[corpus]

pprint(lda_model.print_topics(num_words=50))
vis

"""We can also calculate the coherence score:"""

coherencemodel = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v') #How the model perfoms to create human-interpretable topics
print(coherencemodel.get_coherence())
