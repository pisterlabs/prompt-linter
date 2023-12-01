from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import pickle
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import *
from nltk.stem.porter import *
from sklearn import model_selection
import numpy as np
np.random.seed(400)

# 1. load the data
df_train_jokes = pd.read_csv("shortjokes.csv")
print(df_train_jokes.head())
print(df_train_jokes.shape)

# 2. data preprocessing functions
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result

processed_docs = []

for doc in df_train_jokes['Joke']:
    processed_docs.append(preprocess(doc))

# 3. create bag of words
dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print("Dictionary and corpus created...")



m1 = LdaModel(bow_corpus, 5, dictionary, iterations=50)

print("LDA models built...")
cm = CoherenceModel(m1, dictionary=dictionary, corpus=bow_corpus, coherence='u_mass')
print("Coherence model built...")
coherence = cm.get_coherence()  # get coherence value
print(coherence)


coherenceList_umass = []
coherenceList_cv = []
num_topics_list = np.arange(3,12)
for num_topics in num_topics_list:
    lda= LdaModel(bow_corpus, num_topics=num_topics,id2word = dictionary, iterations=50)
    print("LDA model finished...")
    cm = CoherenceModel(model=lda, corpus=bow_corpus,
                        dictionary=dictionary, coherence='u_mass')
    coherenceList_umass.append(cm.get_coherence())
    # cm_cv = CoherenceModel(model=lda, corpus=bow_corpus,
    #                     dictionary=dictionary, coherence='c_v')
    # coherenceList_cv.append(cm_cv.get_coherence())
    print("coherence finished")

plotData = pd.DataFrame({'Number of topics':num_topics_list,
                         'CoherenceScore':coherenceList_umass})
import matplotlib.pyplot as plt
import seaborn as sns
f,ax = plt.subplots(figsize=(10,6))
sns.set_style("darkgrid")
sns.pointplot(x='Number of topics',y= 'CoherenceScore',data=plotData)
plt.axhline(y=-3.9)
plt.title('Topic coherence')
plt.show()
# plt.savefig('Topic coherence plot.png')