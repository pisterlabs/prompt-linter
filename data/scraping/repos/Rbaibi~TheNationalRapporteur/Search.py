#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[9]:


def search(doc1):
    total1=pd.read_csv('all_keywords2.csv')
    total2=pd.read_csv('all_keywords.csv')
    total=total1.append(total2)
    keywords = total.groupby('f').apply(lambda x: x.nlargest(100,'t'))
    doc1.split()
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    texts_out = []
    for sent in doc1.split():
        doc = nlp("".join(sent))
        print(doc)
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    doc=texts_out
    training_data_x = pd.read_excel("synonyms.xlsx")
    X_train = training_data_x.as_matrix()
    syn=pd.DataFrame()
    for index, val in enumerate(doc):
        try:
            syn[str(index)]=(X_train[np.nonzero(X_train ==(val))[0][0],:])
        except:
            syn[str(index)]=" "
    syn_sp=syn.iloc[:,0]
    syn_sp=syn_sp[syn_sp.notnull()]
    search_results= pd.DataFrame([], columns=['f'])
    for i in range(0, len(doc)):
        syn_sp=syn.iloc[:,i]
        syn_sp=syn_sp[syn_sp.notnull()]
        for j in range(0,len(syn_sp)):
            search_results=search_results.append(pd.DataFrame(keywords.f[keywords.s==syn_sp[j]]))
    search_results=search_results.drop_duplicates() 
    return search_results


# In[3]:


doc = input('Type in the search')


# In[7]:


df.head()


# In[15]:


search_results=search(doc)

results=df.par_id[df.iloc[:,0].isin(search_results.f)]


# In[22]:


df_new=df.iloc[:,0:2]

results=df_new[df.iloc[:,0].isin(search_results.f)]


# In[23]:


results


# In[ ]:





# In[24]:


results.to_csv('results.csv')


# In[ ]:




