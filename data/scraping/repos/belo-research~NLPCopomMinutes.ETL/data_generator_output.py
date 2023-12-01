
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

###https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
### https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# Data related libraries
import pandas as pd # Data frames manipulation.
import numpy as np # Arrays manipulation.

from nltk.corpus import stopwords
import gensim
import nltk
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Random for select a line.

import pandas as pd


### intial cleaning


copom_data=pd.read_csv(r'C:\Users\ThinkPad\Documents\rescarh\nlp_application\copom_data' + '\copom_minutes_data.csv')

#### removing 201 
pattern = "\n|\003|\017|\023|\024|\025|\026|\027|\028|\029|\030|\031|\032|\033|\034|\035"

copom_data['text']=copom_data.text.str.replace(pattern, ' ')

copom_data['text']=copom_data.text.str.replace('_', ' ')


copom_data=copom_data[copom_data['text'].str.len()>7000]

### issue with the 47th to

copom_data=copom_data[copom_data.key!='47th']

copom_data['date']=pd.to_datetime(copom_data['DataReferencia'])

##copom_data=copom_data[copom_data['date']>'2010-01-01']


def lowercase_headline(data_frame, column):
    return data_frame[column].str.lower()

def remove_punctuation(data_frame, column):
    return data_frame[column].str.replace('[^\w\s]','')

def remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def remove_short_words(text, min_len = 3):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) >= min_len:
            result.append(token)
            
    return ' '.join(result)

copom_data["cleaned_title"] = lowercase_headline(copom_data, "text")
copom_data["cleaned_title"] = remove_punctuation(copom_data, "cleaned_title")
copom_data["cleaned_title"] = copom_data["cleaned_title"].apply(remove_numbers)
copom_data["cleaned_title"] = copom_data["cleaned_title"].apply(remove_short_words)

nltk.download('punkt')

copom_data["tokens"] = copom_data["cleaned_title"].apply(nltk.word_tokenize)



### follow_tutorial

data_words=list(copom_data['tokens'])


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Define functions for stopwords, bigrams, trigrams and lemmatization

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use','january',
                   'february','march','april','may','june','july'
                   ,'year', 'months', 'month' ,'august','september','november',
                   'october','december',"twelve"
                   ,'last','committee', "according" ,
                   'billion',"morning_session",'respectively','basis'])


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Form Bigrams

data_words_ns=remove_stopwords(data_words)

data_words_bigrams = make_bigrams(data_words_ns)


''' Just using bigrams for now


wnl = WordNetLemmatizer()
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger')

def lemmatization(texts):
    texts_out = []
    for text in texts:
        lemma=[]
        for word, tag in pos_tag(text):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            if not wntag:
                lemma.append(word)
            else:
                lemma.append(wnl.lemmatize(word, wntag))
                
        texts_out.append(lemma)
    
    return texts_out

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams)


from nltk.stem import PorterStemmer
ps = PorterStemmer()


def stemmer(texts):
    text_out=[]
    for text in texts:
        text_out.append([ps.stem(doc) for doc in text])
        
    return text_out

data_stemmed=stemmer(data_lemmatized) 

copom_data['lemmatised']=data_lemmatized  '''

## https://analisemacro.com.br/data-science/topic-modeling-sobre-o-que-o-copom-esta-discutindo-uma-aplicacao-do-modelo-lda/
#copom_data.to_excel(r'C:\Users\ThinkPad\Documents\rescarh\nlp_application\copom_data' + '\copom_data_cleaned.xlsx')
def score_uniq(lda_model,id2word):

    topics = lda_model.show_topics(num_words=len(id2word), formatted=False)
    xy=[]
    for topic in topics:
        x=pd.DataFrame(topic[1])
        x['number']=topic[0]
        xy.append(x)
    
    xyz=pd.concat(xy)
    xyz.columns=['word','score','topic']
    
    all_scores=[]

    for topic in xyz['topic'].unique():
        temp=xyz[xyz['topic']==topic]
        temp=temp[temp['score']>0.006]
        temp_e=xyz[(xyz.word.isin(temp.word.unique())) & 
                   (xyz['topic']!=topic)]
        temp=temp.merge(temp_e,on='word')
        temp['score']=np.log(temp['score_x']/temp['score_y'])
        temp=temp.groupby('word')['score'].median().reset_index()
        temp['topic_id']=topic
        all_scores.append(temp)
        
        
    return pd.concat(all_scores)



id2word = corpora.Dictionary(data_words_bigrams)

id2word.filter_extremes(no_below=20,no_above=100)
 # Term Document Frequency
 
texts=data_words_bigrams
corpus = [id2word.doc2bow(text) for text in texts]

num_topics = 4
chunksize = 2000
passes = 20
iterations = 400
eval_every = None 
 # Don't evaluate model perplexity, takes too much time.
 
lda_model = gensim.models.ldamodel.LdaModel(
   corpus=corpus,
   id2word=id2word,
   chunksize=chunksize,
   alpha='auto',
   eta='auto',
   num_topics=num_topics,
   iterations=iterations,
   passes=passes,
   eval_every=eval_every,
   per_word_topics=True
   )

from collections import Counter
topics = lda_model.show_topics(num_words=100, formatted=False)

data_count=[]
for i,text in enumerate(texts):
    data_flat = pd.DataFrame({'index':i,'word':[w for w in text]})
    data_count.append(data_flat)
    
data_count=pd.concat(data_count)
data_count['count_words']=1
data_count_n=data_count.groupby(['index'])['count_words'].sum().reset_index()
data_count=data_count.groupby(['word','index']).sum().reset_index()


data_count=data_count[['index','word','count_words']]


topics = lda_model.show_topics(num_words=30, formatted=False)

index_data=[]

for topic in topics:
    x=pd.DataFrame(topic[1])
    x['topic']=topic[0]
    x.columns=['word','weight','topic']
    xy=data_count.merge(x,on='word')
    xy['weighted_count']=xy['weight']*xy['count_words']
    xy=xy.groupby(['index'])['weighted_count'].sum().reset_index()
    xy['topic']=topic[0]
    index_data.append(xy)
    
#####
index_data=pd.concat(index_data)

index_data_n=index_data.groupby(['index'])['weighted_count'].sum().reset_index()

index_data=index_data.merge(index_data_n,on='index')

index_data['weighted_count']=index_data['weighted_count_x']/index_data['weighted_count_y']

index_data=index_data[['index','weighted_count','topic']]

####

index_data.columns=['index','Topic Importance','topic_id']

copom_join=copom_data.reset_index(drop=True).reset_index()[['index','date','Titulo','key']]

index_data=index_data.merge(copom_join,on='index',how='left').sort_values(['topic_id','date'])


####

### creating _topic_index

df_uniq=score_uniq(lda_model,id2word) 

import altair as alt

from altair import datum

'''
base = alt.Chart(df_uniq).mark_bar().encode(
    x='score:Q',
     y=alt.Y('word:N', sort='-x'),
    color='topic_id:N'
).properties(
    width=180,
    height=180
)

chart = alt.hconcat()
for x in [0,1,2,3]:
    chart |= base.transform_filter(datum.topic_id == x)
chart.show()

'''


data_bars=df_uniq.copy()

data_bars.loc[data_bars['topic_id']==0,'Topic']='Policy Intervention'
data_bars.loc[data_bars['topic_id']==1,'Topic']='Exchange Rate'
data_bars.loc[data_bars['topic_id']==2,'Topic']='Economic Growth'
data_bars.loc[data_bars['topic_id']==3,'Topic']='Inflation Growth'

data_bars.columns=['Term','Term Topic Importance', 'topic_id', 'Topic']

data_bars.to_csv(r'C:\Users\ThinkPad\Documents\rescarh\nlp_application\copom_data\topic_compistion.csv')

#######

index_data=index_data.merge(data_bars[['topic_id','Topic']].drop_duplicates(),on='topic_id',how='left')

index_data=index_data[index_data['date']>'2019-01-01']

index_data.to_csv(r'C:\Users\ThinkPad\Documents\rescarh\nlp_application\copom_data\index_data.csv')