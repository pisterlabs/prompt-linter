import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim 
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import cufflinks as cf
import gensim
np.random.seed(2018)
import nltk
nltk.download('wordnet')

from gensim.models import CoherenceModel


"""
Compute c_v coherence for various number of topics

Returns:
-------
model_list : List of LDA topic models
coherence_values : Coherence values corresponding to the LDA model with respective number of topics
"""
def compute_coherence_values(limit, start, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit+1, step):
        model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, random_state=101,passes=2, workers=4)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_values.append(coherence_score)
    return model_list, coherence_values


# Using LDA_TF_IDF model to classify all reviews
def LDA_TF_IDF_apply(i):
    bestmodel = model_list[coherence_values.index(max(coherence_values))]
    result = sorted(bestmodel[bow_corpus[i]], key=lambda tup: -1*tup[1])
    if len(result)>0:
        index = result[0][0]
    else:
        index = 'other'
    return index

# Plot LDA model
def LDA_Result(bestmodel, corpus_tfidf, dictionary):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(bestmodel, corpus_tfidf, dictionary)
    pyLDAvis.save_html(vis,fileobj='Italian.html')
    return vis





