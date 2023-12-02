import nltk
nltk.download('stopwords')
import re
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import numpy 

### Load stop words from en.text to process and clean the data

stop_words = pd.read_csv('..\data\en.txt',header=None)
stop_words=stop_words[0].to_list()
df = pd.read_csv('..\data\posts-formatted-for-topicmodelling.csv', encoding='latin-1')
df.head()
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #deacc = True removes punctuation

data_words = list(sent_to_words(df.text))

def remove_stopwords(texts):
    return[[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
data_words_nostops = remove_stopwords(data_words)

## Create Corpus
id2word = corpora.Dictionary(data_words_nostops)
texts = data_words_nostops
corpus = [id2word.doc2bow(text) for text in texts]

## Build model with 25 topics
lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=25,id2word=id2word,minimum_probability =0, random_state=100)

## Generate topic labels from top 3 words of the topic
x=lda.show_topics(num_topics=25,num_words=3,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
l=[]
for topic,words in topics_words:
    l.append(".".join(words))
    
## Save Document-Topic Probability matrix to csv
doct=lda.get_document_topics(corpus,minimum_probability =0.0)
df=pd.DataFrame([[x[1] for x in y] for y in doct], index = [x for x in range(len(doct))])
df.columns = l
df.to_csv("..\data\prob.csv")