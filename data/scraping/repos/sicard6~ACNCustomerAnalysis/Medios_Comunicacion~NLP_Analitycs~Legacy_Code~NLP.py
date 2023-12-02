#In[]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import spacy
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()# Visualise inside a notebook
import es_dep_news_trf
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

semanaRaw=pd.read_csv(r'C:\Users\nicolas.gomez.garzon\OneDrive - Accenture\Desktop\NLP\ACNCustomerAnalysis\data\raw\semana.csv')
print(semanaRaw.info())
semanaRaw=semanaRaw[semanaRaw['Empresa']=='Ecopetrol']
print(semanaRaw)
#In[]:

# Our spaCy model:
nlp = es_dep_news_trf.load()

removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
tokens = []
for summary in nlp.pipe(semanaRaw['Resumen']):
    proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
    tokens.append(proj_tok)

semanaRaw['tokens'] = tokens
print(semanaRaw['tokens'])

#In[]:
dictionary = Dictionary(semanaRaw['tokens'])
dictionary.filter_extremes(no_below=1, no_above=0.7, keep_n=1000)
print(dictionary.values())
#In[]:
corpus = [dictionary.doc2bow(doc) for doc in semanaRaw['tokens']]
#In[]:
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=50, num_topics=3, workers = 4, passes=10)
#In[]:
topics = []
score = []
for i in range(1,20,1):
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers = 4, passes=10, random_state=100)
    cm = CoherenceModel(model=lda_model, texts = semanaRaw['tokens'], corpus=corpus, dictionary=dictionary, coherence='c_v')
    topics.append(i)
    score.append(cm.get_coherence())
_=plt.plot(topics, score)
_=plt.xlabel('Number of Topics')
_=plt.ylabel('Coherence Score')
plt.show()

#In[]:
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=100, num_topics=5, workers = 4, passes=100)

#In[]:
lda_model.print_topics(-1)
#In[]:
lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(lda_display)
# %%
