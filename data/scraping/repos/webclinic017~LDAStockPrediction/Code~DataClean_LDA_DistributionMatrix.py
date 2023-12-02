#!/usr/bin/env python
# coding: utf-8

# # Ejemplo de prueba 
# 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# 
# 
# X contiene las noticias, pero quiero ver en qué formato
# 

# In[1]:


# Parallelizing using Pool.apply()

import multiprocessing as mp

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
X, _ = make_multilabel_classification(random_state=0)

print(X)
print(X.shape)

lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X)
lda.transform(X[-2:])


# Vale, parece que X es un array de numpy, que contiene 100 filas con 20 columnas cada uno: un vector fila por cada noticia, que contiene el conteo de veces que aparece cada palabra (de un total de 20 palabras). Esto es lo que se llama "Bag Of Words"
# 
# En nuestro caso tenemos que coger todas las noticias, eliminar las palabras te tipo "stopwords" (palabras como "de", "la", "un", ...) y hacernos una lista con todas las palabras únicas que aparecen en el conjunto de todas las noticias (que serán más de 20 seguro, pero bueno).

# ## Leer y limpiar csv

# In[1]:


import pandas as pd

df = pd.DataFrame()
for chunk in pd.read_csv('Noticias_Dataset_IDANAE_latin1.csv', encoding='latin1',sep=';', chunksize=1000, nrows=69000):
    df = pd.concat([df, chunk], ignore_index=True)
    
#df = pd.read_csv("noticias_todo_url.csv", error_bad_lines = False, nrows=35000, chunksize=1000)
# df = pd.read_csv("Noticias_Dataset_IDANAE_latin1.csv", encoding='latin1', sep=';', dtype=object; error_bad_lines = False)
#df['CUERPO SCRAPEADO'] = df['Text']
print(df.shape)
df.head()


# In[2]:


import datetime
# len(df['FECHA SCRAPING'].unique())
len(df['FECHA SCRAPING'].unique())

#Eliminar fechas 0 o nulas
df = df[df['FECHA SCRAPING'] != '0']
df = df[df['FECHA SCRAPING'].notna()]
df = df[df['CUERPO SCRAPEADO'].notna()]

df['Date']= pd.to_datetime(df['FECHA SCRAPING'],format="%d/%m/%Y")
df=df[(df['Date']<datetime.datetime(2020,1,1))]
df=df[(df['Date']>datetime.datetime(2015,1,1))]
#df.sort_values(by=['Date'])
df.shape


# In[3]:


df.iloc[2]['CUERPO SCRAPEADO']


# In[3]:


# eliminar caracteres extraños
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\x93','')
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\x94','')
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\n',' ')
# Remove Emails
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\S*@\S*\s?', '')

# Remove new line characters
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace('\s+', ' ')

# Remove distracting single quotes
df['CUERPO SCRAPEADO'] = df['CUERPO SCRAPEADO'].str.replace("\'", "")

noticia1 = df.iloc[2]['CUERPO SCRAPEADO']
noticia1


# ## Procesar noticias
# 
# 
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# 
# https://machinelearningmastery.com/clean-text-machine-learning-python/
# 

# In[4]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
import gensim

#nltk.download('stopwords')  # Esto solo hay que hacerlo una vez cuando instalas nltk
#nltk.download('punkt')  # lo mismo

STOP_WORDS_SPANISH = stopwords.words('spanish')
#stemmer = SnowballStemmer('spanish')

#Tokenizar
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(df['CUERPO SCRAPEADO']))

print(data_words[:1])

#def tokenizar_noticia(txt):
#    if pd.isnull(txt):
#        return []
#    tokens = word_tokenize(txt)
#    tokens = [w for w in tokens if not w.lower() in STOP_WORDS_SPANISH] 
#    words = [word for word in tokens if word.isalpha()]
#    stemmed_words = [stemmer.stem(i) for i in words] # esto tengo mis dudas de si es bueno o malo
#    return stemmed_words
#tokens1 = tokenizar_noticia(noticia1)
#print(tokens1)


# In[5]:


#Para buscar palabras que van juntas

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
#trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)

print(bigram_mod[data_words[0]])


# In[9]:


print(len(data_words))


# In[6]:


# tokenizar todas las noticias
#noticias = df['CUERPO SCRAPEADO'].tolist()
#print(len(noticias))

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

#Eliminar stopwords, hacer biagramas y lematizar
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in STOP_WORDS_SPANISH] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

#noticias_tokens = [tokenizar_noticia(noticia1) for noticia1 in noticias]
#noticias_tokens= []
#for i, noticia in enumerate(noticias):
#    if i%1000==0:
#        print(i)
#    noticias_tokens.append(tokenizar_noticia(noticia))

#print(noticias_tokens[:10])


# In[7]:


# Step 1: Init multiprocessing.Pool()
#pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
#results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]

# Step 3: Don't forget to close
#pool.close()


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)


# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'es' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# In[11]:


print(len(data_lemmatized))


# In[8]:


# crear diccionario o lista de palabras unicas: words_index
from collections import Counter

#words_index = []
#for tokens in noticias_tokens:
 #   for w in tokens:
  #      if w not in words_index:
   #         words_index.append(w)
#words_index= Counter(set([w for tokens in noticias_tokens for w in tokens])).most_common(500)

counter = Counter() 
for tokens in data_lemmatized:
    for w in tokens:    
        counter.update([w])
     

#print(words_index[:5000])
print(len(counter))
counter.most_common(1000)


# In[9]:


words_index=[t[0] for t in counter.most_common(1000)]
words_index


# In[10]:


# convertir nuestras noticias a Bag of Words
import numpy as np
from scipy.sparse import csr_matrix
def convert_to_bow(noticias, words_index):
    n_noticias = len(noticias)
    n_words = len(words_index)
    X = np.zeros((n_noticias, n_words))  # crear matriz inicialmente a cero 
    for i, noticia in enumerate(noticias):
        for word in noticia:
            if word in words_index:
                j = words_index.index(word)
                X[i,j] += 1
    return X

def convert_to_bow_sparse(noticias, words_index):
    row = []
    col = []
    data = []
    for i, noticia in enumerate(noticias):
        for word in noticia:
            if word in words_index:
                j = words_index.index(word)
                row.append(i)
                col.append(j)
                data.append(1)
    return csr_matrix((data, (row, col)), shape=(len(noticias), len(words_index)))
    
    
    
X = convert_to_bow_sparse(data_lemmatized, words_index)

print(X.shape)

print(X)

print(X[1], data_lemmatized[1])


# ## Latent Dirichlet Allocation

# In[11]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification

lda = LatentDirichletAllocation(n_jobs=10,n_components=20, random_state=0)
lda.fit(X)

matriz_topics = lda.transform(X)
samples=lda.decision_function(X)
acurracy = lda.score(X)
columna_nueva_fecha=np.array(df['FECHA SCRAPING'])
X_final = np.column_stack((columna_nueva_fecha,matriz_topics)) 

print(score)
print(samples)
print(lda.transform(X[:17]))


# In[65]:


#pd.DataFrame(X_final).to_csv('MatrizFrecuencia_LDA_index_fecha.csv',sep=',',header=None)

df_final=pd.DataFrame(matriz_topics,dtype='float')
df_final['FECHA']= columna_nueva_fecha
#df_final.insert(0, 'id', df_final.index)
#df_final.rename(columns={'Unnamed: 0':'ID', 0: 'FECHA'}, inplace=True)
df_final


# In[82]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [17, 17]

df_final['Date']= pd.to_datetime(df_final['FECHA'],format="%d/%m/%Y")
df_final = df_final.sort_values(by=['Date'])
#df_plot= df_final.drop('id',axis=1)
df_plot=df_final

df_plot.groupby(by=['FECHA','Date']).mean().sort_values(by=['Date']).rolling(window=60).mean().plot(subplots=True, layout=(10,2))
#df_plot.groupby(by=['FECHA','Date']).mean()


# In[15]:


df_final.to_csv('MatrizTopics_LDA_index_fecha.csv',sep=';', index=None)


# In[16]:


pd.read_csv('MatrizTopics_LDA_index_fecha.csv', delimiter=';')

