"""
Author: Yang Xu(yxu6@nd.edu)
Purpose: Workshop for twitter data and social science, session 3.
         The code includes how to apply established model to analyze the data.
"""

#%%
import os
import openai
import yaml
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import itertools
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx

os.getcwd()
ROOT_DIR = os.path.split(os.getcwd())[0]
# or
ROOT_DIR = '/where/the/data/is/saved/'
# os.chdir()

#%%
#### Load the data ####
cleaned_data = pd.read_csv(ROOT_DIR+"/Data/cleaned_data.csv",dtype="object",sep='\t')
context_data = pd.read_csv(ROOT_DIR+"/Data/context_data.csv",dtype="object",sep='\t')
annotations_data = pd.read_csv(ROOT_DIR+"/Data/annotations_data.csv",dtype="object",sep='\t')
urls_data = pd.read_csv(ROOT_DIR+"/Data/urls_data.csv",dtype="object",sep='\t')
mention_data = pd.read_csv(ROOT_DIR+"/Data/mentions_data.csv",dtype="object",sep='\t')
hashtag_data = pd.read_csv(ROOT_DIR+"/Data/hashtags_data.csv",dtype="object",sep='\t')

#%%
#### Descriptive Analysis

##### entities associated with each tweet
tweet_entities = context_data.groupby('tweet_id')['entity_name'].unique().reset_index()
tweet_entities.head()
temp = context_data.entity_name.value_counts()
temp[0:30]
target_tweet_id = context_data.loc[context_data.entity_name=="Vladimir Putin"].tweet_id.unique()
cleaned_data.loc[cleaned_data.id.isin(target_tweet_id),'text']

tweet_annotations = annotations_data.groupby('tweet_id')['normalized_text'].unique().reset_index()
temp = annotations_data.normalized_text.value_counts()
temp[0:30]

#%%
##### Word Cloud for the top entities
# type(tweet_annotations.iloc[0,1])
text = [x.tolist() for x in tweet_annotations.normalized_text]
text2 = itertools.chain.from_iterable(text)
text2 = [str(element) for element in text2]
wordcloud = WordCloud(width=1000, height=500, max_words=80).generate_from_frequencies(Counter(text2))
plt.figure(figsize=[100,50])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#%%
##### separate annotations between KyivPost and RT_com
annotations_source = tweet_annotations.merge(cleaned_data[['id', 'source']], left_on='tweet_id', right_on='id', how='left')

KyivPost_annotations = [x.tolist() for x in annotations_source.loc[annotations_source.source=='KyivPost','normalized_text']]
KyivPost_annotations_string = itertools.chain.from_iterable(KyivPost_annotations)
KyivPost_annotations_string = [str(element) for element in KyivPost_annotations_string]

RT_com_annotations = [x.tolist() for x in annotations_source.loc[annotations_source.source=='RT_com','normalized_text']]
RT_com_annotations_string = itertools.chain.from_iterable(RT_com_annotations)
RT_com_annotations_string = [str(element) for element in RT_com_annotations_string]

wordcloud_KyivPost = WordCloud(width=1000, height=500, max_words=80).generate_from_frequencies(Counter(KyivPost_annotations_string))
wordcloud_RT_com = WordCloud(width=1000, height=500, max_words=80).generate_from_frequencies(Counter(RT_com_annotations_string))

plt.figure(figsize=[50,20])
plt.subplot(1,2,1)
plt.imshow(wordcloud_KyivPost, interpolation="bilinear")
plt.axis("off")
plt.title("Kyiv Post Top Annotations", fontdict = {'fontsize' : 60})
plt.subplot(1,2,2)
plt.imshow(wordcloud_RT_com, interpolation="bilinear")
plt.axis("off")
plt.title("RT_com Top Annotations", fontdict = {'fontsize' : 60})
plt.show()

#%%
##### separate entities between KyivPost and RT_com
tweet_entities = context_data.groupby('tweet_id')['entity_name'].unique().reset_index()
tweet_entities.entity_name.explode().value_counts()

entities_source = tweet_entities.merge(cleaned_data[['id', 'source']], left_on='tweet_id', right_on='id', how='left')

KyivPost_entities = [x.tolist() for x in entities_source.loc[entities_source.source=='KyivPost','entity_name']]
KyivPost_entities_string = itertools.chain.from_iterable(KyivPost_entities)
KyivPost_entities_string = [str(element) for element in KyivPost_entities_string]
pd.DataFrame.from_dict(Counter(KyivPost_entities_string), orient='index').reset_index().rename(columns={0:"counts"}).sort_values("counts",ascending=False)

RT_com_entities = [x.tolist() for x in entities_source.loc[entities_source.source=='RT_com','entity_name']]
RT_com_entities_string = itertools.chain.from_iterable(RT_com_entities)
RT_com_entities_string = [str(element) for element in RT_com_entities_string]
pd.DataFrame.from_dict(Counter(RT_com_entities_string), orient='index').reset_index().rename(columns={0:"counts"}).sort_values("counts",ascending=False)

wordcloud_KyivPost = WordCloud(width=1000, height=500, max_words=30).generate_from_frequencies(Counter(KyivPost_entities_string))
wordcloud_RT_com = WordCloud(width=1000, height=500, max_words=30).generate_from_frequencies(Counter(RT_com_entities_string))

plt.figure(figsize=[50,20])
plt.subplot(1,2,1)
plt.imshow(wordcloud_KyivPost, interpolation="bilinear")
plt.axis("off")
plt.title("Kyiv Post Top Entities", fontdict = {'fontsize' : 60})
plt.subplot(1,2,2)
plt.imshow(wordcloud_RT_com, interpolation="bilinear")
plt.axis("off")
plt.title("RT_com Top Entities", fontdict = {'fontsize' : 60})
plt.show()

#%%
##### domains associated with each tweet
tweet_domains = context_data.groupby('tweet_id')['entity_name'].unique().reset_index()
tweet_domains.head(n=10)
##### domains frequency by domain
temp = context_data.entity_name.value_counts()
temp.iloc[0:30]


##### mention associate with each tweet
mention_data.username.value_counts().reset_index().rename(columns={"index":"mentioned_user","username":"counts"})
temp = mention_data.username.value_counts()
temp[0:30]

##### hashtag associate with each tweet
hashtag_data.tag.value_counts().reset_index().rename(columns={"index":"hashtag","username":"counts"})
temp = hashtag_data.tag.value_counts()
data_plot = temp[0:30].reset_index().rename(columns={'index':'hashtag','tag':'counts'})
data_plot.head(20)

plt.figure(figsize=[80,20])
plt.bar(data_plot['hashtag'][0:10],data_plot['counts'][0:10])
plt.title("Top 10 Hashtags",fontdict={'fontsize':80})
plt.xticks(rotation=45,fontsize=70)
plt.yticks(fontsize=70)
plt.show()

#%%
##### Count co-occurance and plot text network
cleaned_data = pd.read_csv(ROOT_DIR+"/Data/cleaned_data.csv",dtype="object",sep='\t')
cleaned_data = cleaned_data.assign(text_to_analyze = cleaned_data.text.str.replace(r'http\S+', '',regex=True).replace(r'@\w+','',regex=True).replace(r'\n','',regex=True).replace(r"[^a-zA-Z0-9]"," ", regex=True).replace(r"[#:;',\'”\-’\.]"," ",regex=True).replace(r'\ss\s',' ',regex=True))
KyivPost_tweets_text = cleaned_data.loc[cleaned_data.source=="KyivPost",'text_to_analyze']
RT_com_tweets_text = cleaned_data.loc[cleaned_data.source=="RT_com",'text_to_analyze']

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# from nltk.stem.porter import PorterStemmer
# english_stemmer = PorterStemmer()
# english_stemmer = nltk.stem.SnowballStemmer('english')
# nltk.download('omw-1.4')

## define lemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

# class StemmedCountVectorizer(CountVectorizer):
    # def build_analyzer(self):
        # analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        # return lambda doc:([english_stemmer.stem(w) for w in analyzer(doc)])

## apply CountVectorizer on the corpus, count frequency for each of the bigram

CountModel = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words='english',ngram_range=(2,2))
X = CountModel.fit_transform(KyivPost_tweets_text)
counts = pd.DataFrame(X.toarray(), columns=CountModel.get_feature_names_out())
count_values= X.toarray().sum(axis=0)
vocab = CountModel.vocabulary_
KyivPost_bigram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1:'bigram'})
KyivPost_bigram[['token1','token2']] = KyivPost_bigram['bigram'].str.split(" ",1, expand=True)

#%%
## Plot the text network: KyivPost text networks with node "Ukrain"
Graph = nx.Graph()
for index,row in KyivPost_bigram.loc[KyivPost_bigram.bigram.str.contains(r'ukrain')].head(30).iterrows():
    Graph.add_edge(row['token1'],row['token2'],weight=row['frequency']*10)
fig, ax = plt.subplots(figsize=(18,11))
pos = nx.spring_layout(Graph, k=5, seed=20221004)
nx.draw_networkx(Graph, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='blue',
                 with_labels = False,
                 ax=ax)
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='blue', alpha=0.25),
            horizontalalignment='center', fontsize=15)
plt.title("KyivPost Text Network with Ukraine Nodes")
plt.show()

#%%
## Plot the text network: KyivPost text networks with node "Russia"
Graph = nx.Graph()
for index,row in KyivPost_bigram.loc[KyivPost_bigram.bigram.str.contains(r'russia')].head(30).iterrows():
    #Graph.add_edge(k.split(" ")[0],k.split(" ")[1], weight=v*10)
    Graph.add_edge(row['token1'],row['token2'],weight=row['frequency']*10)
fig, ax = plt.subplots(figsize=(18,11))
pos = nx.spring_layout(Graph, k=5, seed=20221004)
nx.draw_networkx(Graph, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='blue',
                 with_labels = False,
                 ax=ax)
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='blue', alpha=0.25),
            horizontalalignment='center', fontsize=15)
plt.title("KyivPost Text Network with Russia Nodes")
plt.show()

#%%
## Plot the text network: RT_com text networks with node "Ukrain"
Y = CountModel.fit_transform(RT_com_tweets_text)
counts = pd.DataFrame(Y.toarray(), columns=CountModel.get_feature_names_out())
count_values= Y.toarray().sum(axis=0)
vocab = CountModel.vocabulary_
RT_com_bigram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1:'bigram'})
RT_com_bigram[['token1','token2']] = RT_com_bigram['bigram'].str.split(" ",1, expand=True)

Graph = nx.Graph()
for index,row in RT_com_bigram.loc[RT_com_bigram.bigram.str.contains(r'ukrain')].head(30).iterrows():
    Graph.add_edge(row['token1'],row['token2'],weight=row['frequency']*10)
fig, ax = plt.subplots(figsize=(18,11))
pos = nx.spring_layout(Graph, k=5, seed=20221004)
nx.draw_networkx(Graph, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='blue',
                 with_labels = False,
                 ax=ax)
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='blue', alpha=0.25),
            horizontalalignment='center', fontsize=15)
plt.title("RT_com Text Network with Ukraine Nodes")
plt.show()

#%%
## Plot the text network: RT_com text networks with node "Russia"
Graph = nx.Graph()
for index,row in RT_com_bigram.loc[RT_com_bigram.bigram.str.contains(r'russia')].head(30).iterrows():
    #Graph.add_edge(k.split(" ")[0],k.split(" ")[1], weight=v*10)
    Graph.add_edge(row['token1'],row['token2'],weight=row['frequency']*10)
fig, ax = plt.subplots(figsize=(18,11))
pos = nx.spring_layout(Graph, k=5, seed=20221004)
nx.draw_networkx(Graph, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='blue',
                 with_labels = False,
                 ax=ax)
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='blue', alpha=0.25),
            horizontalalignment='center', fontsize=15)
plt.title("RT_com Text Network with Russia Nodes")
plt.show()

#%%
#### Sentiment Analysis

cleaned_data = pd.read_csv(ROOT_DIR+"/Data/cleaned_data.csv",dtype="object",sep='\t')
cleaned_data = cleaned_data.assign(text_to_analyze = cleaned_data.text.str.replace(r'http\S+', '',regex=True).replace(r'@\w+','',regex=True).replace(r'\n','',regex=True))
cleaned_data = cleaned_data[['id','text','referenced_text','all_text','text_to_analyze']]
cleaned_data = cleaned_data.applymap(str)
cleaned_data.head()
cleaned_data.dtypes

cleaned_data.loc[0,'text']
cleaned_data.loc[0,'text_to_analyze']

##### nltk sentiment analysis
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import tokenize
## ! nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
# lines_list = tokenize.sent_tokenize()
temp = analyzer.polarity_scores(cleaned_data.text_to_analyze[3])
pd.DataFrame(temp,index=[0])

sentiment_result = pd.concat([pd.DataFrame(analyzer.polarity_scores(element),index=[0]) for element in cleaned_data.text_to_analyze],keys=cleaned_data.id)
sentiment_result = sentiment_result.reset_index(level=1,drop=True).reset_index(level=0)
sentiment_result

#%%
#### textblob sentiment analysis
from textblob import TextBlob
temp = TextBlob(cleaned_data.text_to_analyze[3])
temp.sentiment
sentiment_result = pd.concat([pd.DataFrame(TextBlob(element).sentiment).T for element in cleaned_data.text_to_analyze],keys=cleaned_data.id)
sentiment_result = sentiment_result.reset_index(level=1,drop=True).rename(columns={0:"polarity",1:"subjectivity"}).reset_index(level=0)
sentiment_result

#%%
#### flair sentiment analysis
from flair.models import TextClassifier
from flair.data import Sentence
analyzer2 = TextClassifier.load('en-sentiment')
result = []
for element in cleaned_data.text_to_analyze[0:10]:
    sentence = Sentence(element)
    analyzer2.predict(sentence)
    result.append(sentence.labels)

sentiment_result = pd.DataFrame(result)[0].astype(str).str.split(r"→",expand=True).rename(columns={0:'label',1:'confidence'})
sentiment_result[['sentiment','confidence']] = sentiment_result['confidence'].str.split("\s+\(", expand=True).rename(columns={0:'label',1:'confidence'})
sentiment_result['confidence'] = sentiment_result['confidence'].str.replace(r'\)',"", regex=True).astype(float)
sentiment_result = sentiment_result[['label', 'sentiment', 'confidence']]
sentiment_result.describe(include='all')
sentiment_result

#%%
#### BERTweet
import torch
from transformers import AutoModel,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

line = cleaned_data.text[3]

input_ids = torch.tensor([tokenizer.encode(line)])
bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

with torch.no_grad():
    features = bertweet(input_ids)
features[1]

#%%
#### openai
# with open(ROOT_DIR+"/Code/twitter_credential_true.yaml", 'r') as stream:
    # OPENAI_API_KEY = yaml.safe_load(stream)['openapi_api-keys']
OPENAI_API_KEY=""
openai.api_key = os.getenv(OPENAI_API_KEY)

#### Use openapi to analytize the top-5 liked tweets
cleaned_data = pd.read_csv(ROOT_DIR+"/Data/cleaned_data.csv",dtype="object",sep='\t')
cleaned_data[['retweet_count','reply_count','like_count','quote_count']] = cleaned_data[['retweet_count','reply_count','like_count','quote_count']].astype(int)
text_subset = cleaned_data.loc[cleaned_data.like_count>30].sort_values('like_count',ascending=False)[['text','like_count']]
text_subset = text_subset['text']
text_to_analyze = text_subset.str.replace(r'http\S+', '',regex=True).replace(r'@\w+','',regex=True).replace(r'\n','',regex=True)
text_to_analyze = text_to_analyze.to_frame().assign(order=range(1,len(text_to_analyze)+1,1)).applymap(str)
text_to_analyze = text_to_analyze.assign(prompt = text_to_analyze.order+". \\"+text_to_analyze.text)

prompt = 'Classify the sentiment in these tweets:\n\n' +text_to_analyze.loc[:,'prompt'].iloc[0:5].str.cat(sep='\n')

#%%
openai.api_key = OPENAI_API_KEY
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt=prompt,
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
response
text_to_analyze.iloc[4,0]
pd.Series(response['choices'][0]['text'].split('\n'))
pd.Series(response['choices'][0]['text'].split('\n'))
