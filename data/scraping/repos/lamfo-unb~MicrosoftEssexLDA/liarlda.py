#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import models
import pyLDAvis.gensim
import pyLDAvis.sklearn
import os
from gensim.parsing.preprocessing import remove_stopwords
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
import re
from gensim.models.phrases import Phrases
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns



stop_words = stopwords.words('english')

stop_words.extend(["time","week","day","month","cnn","year","going","covid","19","covid-19"])

print(stop_words)




# In[64]:


df = pd.read_csv("/Users/lucasgomes/Documents/projetos/Essex/MicrosoftEssexLDA/lda/train.tsv",sep="\t")


# In[65]:


df = df[(df.false == "false") | (df.false == "true")]
# df[df["0.3"]>10]
print(df.false)


# In[66]:


print(list(df))
df["Text"]=df["Says the Annies List political group supports third-trimester abortions on demand."]
# print(df["0.3"].mean())
# print(len(df))
# print(len(df[df["0.3"]>10]))
# print(df["0.3"][df["false"]=="false"].mean())


# In[67]:


df.Text = df.Text.apply(lambda x: remove_stopwords(x))
df.Text = df.Text.apply(lambda x: re.sub(r'\W', ' ', x))
df.Text = df.Text.apply(lambda x: re.sub(r' \w ', ' ', x))
df.Text = df.Text.apply(lambda x: x.lower())
df.Text = df.Text.apply(lambda x: x.split())
lemmatizer = WordNetLemmatizer()
df.Text = df.Text.apply(lambda x: [lemmatizer.lemmatize(token) for token in x] )

df.Text = df.Text.apply(lambda x: [w for w in x if not w in stop_words])


phrase_model = Phrases(df.Text, min_count=1, threshold=1)
df.Text = df.Text.apply(lambda x: phrase_model[x] )

df.Text = df.Text.apply(lambda x: [w for w in x if len(w)>1])





common_texts = df.Text.tolist()


# In[68]:


print(df.Text)
print(len(df))


# In[69]:


# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
common_dictionary.filter_extremes(no_below=5, no_above=0.5)

common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]


# In[84]:


LDA_model = models.LdaModel(corpus=common_corpus,
                         id2word=common_dictionary,
                         num_topics=20,
                         update_every=1,
                         chunksize=len(common_corpus),
                         passes=3,
                         alpha='auto',
                         random_state=42,
                         minimum_probability = 0,
                        minimum_phi_value = 0)


# In[ ]:





# In[85]:


def cleanlda(vector):
    topic_percs_sorted = sorted(vector, key=lambda x: (x[1]), reverse=True)
    return topic_percs_sorted[0][0]

lista = [cleanlda(LDA_model[common_corpus[i]]) for i in range(len(common_corpus))]
df["class"] = lista
print(df["class"].mean())


# In[103]:


# print(len(df[(df.false == "false") & (df["class"] == 1)]))
# print(df["class"].value_counts())
df3=df["Says the Annies List political group supports third-trimester abortions on demand."][df.false == "false"]
print(df3.values.tolist())


# In[48]:


# LDA_models[ideal_topic_num_index+1].save("../../MicrosoftEssexHeroku/ldamodel")
# common_dictionary.save("../../MicrosoftEssexHeroku/ldadic")
# phrase_model.save("../../MicrosoftEssexHeroku/phaser")


# In[ ]:


p = pyLDAvis.gensim.prepare(LDA_model, common_corpus, common_dictionary, n_jobs=-1, sort_topics=False)

pyLDAvis.save_html(p, "../../MicrosoftEssexHeroku/FONTE".replace("FONTE","LDA.html"))


# In[ ]:





# In[ ]:


pyLDAvis.display(p, local = True)


# In[ ]:


# md = "# Examples for each topic \n"
# for i in range(0,len(num_topics)-1):
#     md = md  + "\n"
#     print(i)
#     md = md + "## Topic "+str(i+1) + "\n"
#     collected = 0
#     for row in df.itertuples():        
#         other_corpus = common_dictionary.doc2bow(row.Text)
#         vector = LDA_models[ideal_topic_num_index+1][other_corpus]
#         topic_percs_sorted = sorted(vector, key=lambda x: (x[1]), reverse=True)
#         if topic_percs_sorted[0][0] == i:
#             if topic_percs_sorted[0][1] > 0.9:
#                 md = md +"("+str(collected+1)+") " + row.URL +" "+str(int(topic_percs_sorted[0][1]*100)) + "% \n\n"
#                 collected += 1
#                 if collected == 10:
#                     break
#             if row.Index > 1000:
#                 if topic_percs_sorted[0][1] > 0.5:
#                     md = md +"("+str(collected+1)+") "+ row.URL +" "+str(int(topic_percs_sorted[0][1]*100))+ "% \n\n"
#                     collected += 1
#                     if collected == 10:
#                         break
#             if row.Index > 2000:
#                 if topic_percs_sorted[0][1] > 0.3:
#                     md = md  +"("+str(collected+1)+") "+row.URL +" "+ str(int(topic_percs_sorted[0][1]*100)) + "% \n\n"
#                     collected += 1
#                     if collected == 10:
                        
#                         break

# print(md)
# text_file = open("../../MicrosoftEssexHeroku/sites.txt", "w")
# n = text_file.write(md)
# text_file.close()
                        
        


# In[ ]:




