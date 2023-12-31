# -*- coding: utf-8 -*-
"""topic_modelling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P2GEeEQPB5aEsH0ST_8mffLtUMZx-ZpX
"""

import pandas as pd
import numpy as np
from tqdm import tqdm, notebook
tqdm.pandas()

cnn_file = '/content/drive/My Drive/CNN_coronavirus1.xlsx'
df = pd.read_excel(cnn_file,index_col=0)
print(df.head())

import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from collections import Counter

# Commented out IPython magic to ensure Python compatibility.
import re, string
from pprint import pprint

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import spacy

!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

df[df.duplicated(subset='content')]

df.drop_duplicates(subset='content',inplace=True)
indexnames = df[df['content'].apply(lambda x: str(x).isdigit())].index
# print(indexnames)
df.drop(indexnames,inplace=True)
print(type(df.loc[5216,'content']))

stop_sentences = ['contributed to this', 
                  'This material may not be published, broadcast, rewritten, or redistributed', 
                  'FOX News Network, LLC', 
                  'All rights reserved',
                  'All market data delayed',
                  'Sign up here',
                  "Get all the stories you need-to-know from the most powerful name in news delivered first thing every morning to your inbox Subscribed You've successfully subscribed to this newsletter! This material may not be published, broadcast, rewritten, or redistributed",
                  "LIMITED TIME OFFER, ",
                  r"Learn about all the*",
                  r'Get[ ]*latest[ ]*news',
                  r"delivered[ ]*daily[ ]*inbox",
                  r"Check out what's clicking[ ,\w]*",
                  r"successfully subscribed to[ ,\w]*",
                  r"Thank you for making us your first choice[ ,\w]*",
                  r"Flash top[ \w,]*headlines",
                  r"CLICK HERE*",
                  "Mobile users click here",
                  r"Fun stories about [\w,]* and more",
                  r"Stay up-to-date on the biggest [\w,]* news with our weekly recap",
                  r"originally appeared on [\w,]*",
                  r"originally published on [\w,]*",
                  r"Get a daily look at[ \w,]*",
                  "Fox Nation",
                  "The FOX NEWS RUNDOWN",
                  "subscribe and download",
                  "FOX platforms",
                  "FOX NOW",
                  "FOX NEWS APPFox News",
                  "Fox News First",
                  "copyright",
                  "Follow below on the Fox News live blog",
                  "Kim Komando Show",
                  # stop sentences from CNN
                  'Watch the latest videos on Covid-19.',
                  r'live[ ]*coverage[ ]*of[ \w]*',
                  "Note: The prices above reflect the retailer's listed price at the time of publication.",
                  "Read the full story here.",
                  r"CNN Coronavirus",
                  r"A version of this article first appeared",
                  "You can sign up for free right here",
                  "At CNN, we start with the facts.",
                  "Visit CNN's home for Facts First.",
                  "delivered to your inbox daily.",
                  "Sign up here."
                  ]
stop_sentences = [*map(lambda x: x.lower(), stop_sentences)]

def contain_stop_sentences(sentence):
    '''
    extract the sentences that contain stop sentences
    '''
    check_status = [*map(lambda x: bool(re.search(x, sentence)), stop_sentences)]
    return(any(check_status))

def extract_no_stop_sentences(text_data):
    '''
    Applied on dataframe's column level,
    to delete the stop sentences from a whole news article
    '''
    single_news = ' '.join(''.join(text_data).lower().split('\xa0')) #exclude \xa0
    single_news_sentences = sent_tokenize(single_news)
    new_sentences = []
    for x in single_news_sentences:
        if contain_stop_sentences(x) is False:
            new_sentences.append(x)
    single_new_news = ' '.join(new_sentences)
    
    return(single_new_news)

new_news_list = df['content'].apply(lambda x: extract_no_stop_sentences(str(x)))
df.loc[:,'true_content'] = new_news_list

def make_lower(text):
    return text.lower()

def remove_punctuation(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) 
    text = re.sub("\'", "", text)
    return re.sub(r'[^\w\s]', ' ', text)

def strip_extraspace(text):
    return ' '.join(text.split())

def remove_digits(text):
    return re.sub('\d', ' ', text)

def replace_word(text,word,replacement):
    return text.replace(word,replacement)

def remove_words(text,wordlist):
    for word in wordlist:
        if word in text.split():
            text = re.sub(r'\b{}\b'.format(word), '', text)  
    return text

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.append('coronavirus')
stop_words.append('fox')
stop_words.append('cnn')

def clean_text(text):
    text = make_lower(text)
    text = replace_word(text,'covid-19','covid') 
    text = replace_word(text,'corona virus','coronavirus') 
    text = replace_word(text,'covid','coronavirus') 
    text = replace_word(text,'fox news','fox') 
    text = replace_word(text,'new york','newyork')
    text = replace_word(text, 'begin video clip', '')
    text = replace_word(text, 'commercial break', '')
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = remove_words(text,stop_words)
    
    return text

df['clean_content'] = df['true_content'].progress_apply(lambda x:clean_text(x))
data = df.clean_content.values.tolist()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
data_words[:2]

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
nltk.download('wordnet')

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
lemmatizer = WordNetLemmatizer()

# Do lemmatization keeping only noun, adj, vb, adv
all_news = []
for one_news_list in notebook.tqdm(data_words):
    one_news = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in one_news_list]
    all_news.append(one_news)
    
data_lemmatized = all_news

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_lemmatized, min_count=5, threshold=100) # higher threshold fewer phrases.

# Faster way to get a sentence clubbed as a bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

data_words_bigrams = make_bigrams(data_lemmatized)

# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)

# Create Corpus
texts = data_words_bigrams

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# # Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

import os       #importing os to set environment variable
def install_java():
  !apt-get install -y openjdk-8-jdk-headless -qq > /dev/null      #install openjdk
  os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"     #set environment variable
  !java -version       #check java version
install_java()

!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
!unzip mallet-2.0.8.zip

os.environ['MALLET_HOME'] = '/content/mallet-2.0.8'
mallet_path = '/content/mallet-2.0.8/bin/mallet'

ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=5, id2word=id2word)

# Show Topics
print(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word,alpha=50/num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words_bigrams, start=3, limit=24, step=3)

# Commented out IPython magic to ensure Python compatibility.
# Show elbow graph
import matplotlib.pyplot as plt
# %matplotlib inline

limit=24; start=3; step=3;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

optimal_model = model_list[5]
model_topics = optimal_model.show_topics(formatted=False)
print("Topics for the chosen LDA model:\n")
pprint(optimal_model.print_topics(num_words=10))

def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel = optimal_model, corpus = corpus, texts = data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.sample(10)
df_dominant_topic.to_excel('LDA_cnn_topics.xlsx', index = False)

# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet

sent_topics_sorteddf_mallet.to_excel('LDA_cnn_topics_rep.xlsx', index = False)

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']].drop_duplicates()
topic_num_keywords = topic_num_keywords.set_index('Dominant_Topic')

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

df_dominant_topics.sort_values('Num_Documents', ascending = False).to_excel('LDA_foxcnn_colab_topics.xlsx', index = True)

pprint(df_dominant_topics.Topic_Keywords)

df.head()
df_dominant_topic_date = pd.concat([df['published_date'].reset_index(), df_dominant_topic], axis = 1)
df_dominant_topic_date.head()

df_dominant_topic_date.to_excel('cnn_dominant_topic_date.xlsx', index = False)

agg_topicandtime = pd.pivot_table(df_dominant_topic_date, index=['published_date'], values = ['Document_No'], columns = ['Dominant_Topic'], aggfunc = 'count')
agg_topicandtime.head()
agg_topicandtime.to_excel('cnn_agg_topicandtime.xlsx')

optimal_model.save('LDA_CNN_mallet')
optimal_model = gensim.models.ldamodel.LdaModel.load('LDA_CNN_mallet')

# For further analysis and improvement of model
# import operator
# def ret_top_model():
#     """
#     Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
#     quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
#     evaluating the topic model until this threshold is crossed. 
    
#     Returns:
#     -------
#     lm: Final evaluated topic model
#     top_topics: ranked topics in decreasing order. List of tuples
#     """
#     top_topics = [(0, 0)]
#     while top_topics[0][1] < 0.90:
#         print("Getting model ready\n")
#         lm = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word,chunksize= 100,alpha=0.01,dtype=np.float64,passes=1,num_topics=10)
#         print("Model ready\n")
#         coherence_values = {}
#         for n, topic in lm.show_topics(num_topics=-1, formatted=False):
#             topic = [word for word, _ in topic]
#             cm = CoherenceModel(topics=[topic], texts=data_words_bigrams, dictionary=id2word, window_size=10)
#             coherence_values[n] = cm.get_coherence()
#             print("Getting coherence: {}\n".format(coherence_values[n]))
#         print("Out of the for loop\n")
#         top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
#         print("Top topics: ",top_topics[0][1])
#     return lm, top_topics

# lm, top_topics = ret_top_model()
# print(top_topics[:5])

