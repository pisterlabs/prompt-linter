from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
#import package needed
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import ast 
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy



#load dataframe
tws_df = pd.read_csv('/home/zhoulita/scratch/covid_data_unprocessed/dataFrame/tweet_df_loc_total.csv',low_memory=False,lineterminator='\n')

def get_week_index(df):
    return int((df['day_index']-1)/7)
tws_df['week_index']= tws_df.apply(get_week_index,axis = 1)
#remove unnecessary column
tws_df.drop(['lang', 'referenced_tweets','author_id', 'source', 'possibly_sensitive', 'public_metrics','entities', 'context_annotations', 'in_reply_to_user_id', 'attachments'], axis=1, inplace=True)
tws_df.drop(['Unnamed: 0', 'Unnamed: 0.1','geo_x', 'id_y', 'withheld'], axis=1, inplace=True)



#load the us places with the state code
us_pls_df = pd.read_csv("/home/zhoulita/scratch/covid_19_tweets/us_places.csv")
us_pls_df.drop(['Unnamed: 0', 'full_name','place_type', 'country_code', 'country','geo','name','Lontitude','Latitude'], axis=1, inplace=True)



total_df = pd.merge(tws_df,us_pls_df,how = 'inner',left_on = 'geo_id',right_on = 'id')
total_df.drop(['place_type', 'country','place_type', 'Lontitude', 'geo_y','Latitude','id'], axis=1, inplace=True)



#text cleaning
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)



#define a function to clean the text
def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)#remove the @
    text = re.sub(r'#','',text) #remove the #
    text = re.sub(r'RT[\s]+','',text) #remove retweet
    text = re.sub(r'https?:\/\/\S+','',text) #remove http 
    text =  re.sub('\n','',text)
    text = re.sub('\S*@\S*\s?', '', text)#remove email
    text = re.sub('\s+', ' ', text)
    text = remove_emoji(text)
    return text

total_df['processed_text'] = total_df['text'].apply(clean_text)


#use regex to clean and filter out the useful info from tweets related to family 
regex_election = re.compile(r'(?i)Trump|Biden|Election|democratic|republican|party|President|campaign|elector|candidate') 
regex_mask = re.compile(r'(?i)mask|Mask')

def mask_related_text(text):
    if regex_mask.search(text):
        return True
    return False

def election_related_text(text):
    if regex_election.search(text):
        return True
    return False

total_df['election_related'] = total_df['processed_text'].apply(election_related_text)
total_df['mask_related'] = total_df['processed_text'].apply(mask_related_text)
total_df = total_df[(total_df["mask_related"]==True)&(total_df["election_related"]==False)]



def process_to_phrase(string):
    return gensim.utils.simple_preprocess(str(string), deacc=True)

total_df['processed_text_phrase'] = total_df['processed_text'].apply(process_to_phrase)



data = total_df['processed_text_phrase'].values.tolist()


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)




def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemm`atization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(data)  # processed Text Data!


# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]




# use the for loop to find out the best number of topics to use for the highest coherence number
topic_num_coherence_df = pd.DataFrame()
for i in range(3,16):
    topic_num = i
    print("Number of Topics:",topic_num)
    temp_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topic_num, 
                                           update_every=1,
                                           random_state=50,
                                           passes=20,
                                           per_word_topics=True)
    
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=temp_model, texts=data_ready, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    #attach to the data frame
    topic_num_coherence_df = topic_num_coherence_df.append(pd.Series([int(i),coherence_lda]), ignore_index=True)
    
    print('Coherence Score: ', coherence_lda)

topic_num_coherence_df.columns = ['Topics_Num', 'Coherence_score']


#visualize the score and number of Topics
plt.figure(figsize=(20,8))
plt.plot(topic_num_coherence_df['Topics_Num'],topic_num_coherence_df['Coherence_score'],'b')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Scores')
plt.title('Determine the optimal number of Topics')
plt.show()
plt.savefig("mask_topic_num.png")




