#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
from pprint import pprint
import nltk

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel

# spacy for lemmatization
import spacy


# In[ ]:


#!pip install -r requirements.txt


# In[5]:


import re
import numpy as np
import pandas as pd
from pprint import pprint
import nltk

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

stop_words = nltk.corpus.stopwords.words('dutch')
stop_words.extend(['kb', 'pdf','nationaal','rapporteur'])

#Install Dutch Spacy with python -m spacy download nl_core_news_sm
#!python -m spacy download nl_core_news_sm
import nl_core_news_sm

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
#python3 -m spacy download en
nlp = nl_core_news_sm.load()

# Execute Applications/python 3.7/Install Certificates.command
import nltk
nltk.download('stopwords')

df = pd.read_excel('pdf_pages.xlsx')


# In[35]:


def clean_data(data):
    # Put string in list
    if type(data) == type('string'):
        data = [data]
    #for sent in data[0:100]:
        #print(sent.dtype)
    
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove special charcters
    pattern = r'[^a-zA-z0-9\s]' if not True else r'[^a-zA-z\s]'
    data = [re.sub(pattern, '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]

    data = [re.sub("\_", "", sent) for sent in data]
    return data


def parse_data(data):

    data = clean_data(data)

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
        # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    
    print(data_words_bigrams)

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_lemmatized


# ## Prepare data for model

# In[74]:


# Convert to list
data = df.loc[0:10000].par_id.values.tolist()
data_lemmatized = parse_data(data)

# Do lemmatization keeping only noun, adj, vb, adv

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[76]:


# Do lemmatization keeping only noun, adj, vb, adv

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


# In[81]:


len(corpus)


# ## LSI

# In[ ]:


# Try methods: tf-idf, LSI and LDA
lsi = gensim.models.LsiModel(corpus, id2word=id2word, num_topics=13)

doc = "mensenhandel loverboy"
vec_bow = id2word.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
print(vec_lsi)

index = gensim.similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi]
print(list(enumerate(sims)))


# ## TF-IDF

# In[89]:


tfid = TfidfModel(corpus)
doc = data_lemmatized
total= pd.DataFrame([], columns=['f','s','t'])


# In[95]:


for i in range(9598, len(doc)):
    vec_bow = id2word.doc2bow(doc[i])
    vector = tfid[vec_bow]
    d = pd.DataFrame([], columns=['f','s','t'])
    for val in vector:
        word = id2word.get(val[0])
        d=d.append(pd.DataFrame([[i, word,  val[1]]], columns=['f','s','t']))
    total=total.append(d)


# In[96]:


i


# In[97]:


total.head()


# In[98]:


total.to_csv('all_keywords2.csv')


# In[103]:


total1=pd.read_csv('all_keywords2.csv')
total2=pd.read_csv('all_keywords.csv')
total=total1.append(total2)


# In[104]:


total.shape


# In[106]:


keywords = total.groupby('f')['t'].nlargest(100)
keywords


# In[ ]:





# In[160]:


total.head()


# ## Search

# In[115]:


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[126]:


doc1=("kinderen loverboy")

doc1.split()


# In[ ]:





# In[130]:


allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
texts_out = []
for sent in doc1.split():
    doc = nlp("".join(sent))
    print(doc)
    texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])


# In[131]:


doc=texts_out


# In[161]:


doc[1][0]


# In[144]:


syn=pd.DataFrame()


# In[146]:


for index, val in enumerate(doc):
    try:
        syn[str(index)]=(X_train[np.nonzero(X_train ==(val))[0][0],:])
    except:
        syn[str(index)]=" "


# In[166]:


return_res=keywords.f[keywords.s==doc[1][0]]


# In[168]:


return_res.shape


# In[ ]:


data_lemmatized[0]


# In[ ]:


id2word.doc2bow("kinderen rapporteur".split())


# In[ ]:


"kinderen rapporteur".split()


# In[ ]:


doc = parse_data("kinderen loverboy")
vec_bow = id2word.doc2bow(doc[0])
vec_bow


# In[ ]:


doc=pd.DataFrame(np.array([["kinderen loverboy"], ["mensenhandel loverboy"]]))


# In[ ]:


doc


# In[ ]:





# In[141]:


training_data_x = pd.read_excel("synonyms.xlsx")
X_train = training_data_x.as_matrix()


# In[ ]:


X_train


# In[ ]:


try:
    print(X_train[np.nonzero(X_train =='misbruik')[0][0],:])
except:
    print ("no synonyms")


# In[ ]:


from nltk.corpus import wordnet
syns = wordnet.synsets("dog")
print(syns)


# In[ ]:


from nltk.corpus import wordnet
synonyms = []
antonyms = []

for syn in wordnet.synsets("active"):
    for l in syn.lemmas():
    synonyms.append(l.name())
    if l.antonyms():
    antonyms.append(l.antonyms()[0].name())

    print(set(synonyms))
    print(set(antonyms))


# In[71]:


for i in range(0,1000):
    #if type(i)!=type(string)):
    try:
        cl_string = re.sub('\S*@\S*\s?', '', data[i])
        print(cl_string)
        data_cl.append(cl_string)
    except:
        print(i)

