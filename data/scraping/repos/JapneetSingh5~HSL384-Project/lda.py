import csv
import os
import nltk
import pickle
# remove numeric, stop words and short words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import sys
import csv
import pandas as pd
# tokenize the lyrics
from nltk.tokenize import RegexpTokenizer
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import datetime

csv.field_size_limit(sys.maxsize)



def get_df(filename):
    # Read the csv file
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    df = pd.DataFrame(data)

    df.columns = df.iloc[0]

    df = df.iloc[1:]

    return df

def get_profanities():
    # https://www.freewebheaders.com/youtube-blacklist-words-free-and-youtube-comment-moderation/

    with open('profanity.txt', 'r') as file:
            # split by comma
            profanities = file.read().split(',')
            profanities = [w.strip() for w in profanities]

    return profanities

def remove_profanity(tokens, profanities):

    return [w for w in tokens if not w in profanities]

def get_stopwords():
    stop_words = set(stopwords.words('english'))
    song_specific =  ['lyrics', 'verse']
    pronouns = ['the', 'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    pronouns = pronouns + song_specific
    pronouns = set(pronouns)
    stop_words = stop_words.union(pronouns)

    return stop_words

def remove_stopwords(tokens, stop_words):
    return [w for w in tokens if w not in stop_words]

def remove_punctuation(tokens):
    return [w for w in tokens if w not in string.punctuation]

def remove_numeric(tokens):
    return [w for w in tokens if not w.isnumeric()]

def remove_short_words(tokens):
    return [w for w in tokens if len(w) > 2]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens]

def preprocess_pipeline(tokens,stopwords):
    # lower case
    tokens = [w.lower() for w in tokens]
    tokens = remove_stopwords(tokens,stopwords)
    tokens = remove_punctuation(tokens)
    tokens = remove_numeric(tokens)
    tokens = remove_short_words(tokens)
    tokens = lemmatize(tokens)

    # # remove profanity
    # tokens = remove_profanity(tokens)
    
    return tokens

def tokenize_lyrics(df):

    # tokenise
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords = get_stopwords()

    df['tokens'] = df['Lyrics'].apply(tokenizer.tokenize)

    df["processed_tokens"] = df['tokens'].apply(lambda x : preprocess_pipeline(x,stopwords))

    # remove any song with 0 tokens
    df = df[df['processed_tokens'].map(len) > 0]

    return df

def get_tokens(df):

    # create list of all tokens 
    all_tokens = df['processed_tokens'].tolist()

    return all_tokens

def create_dict(tokens, min_count, max_perc):

    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below = min_count, no_above = max_perc)

    return dictionary

def bow_model(dictionary, all_tokens):

    gensim_corpus = [dictionary.doc2bow(song) for song in all_tokens]
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    return (gensim_corpus, id2word)

def lda_model(params):
    
    lda_model = LdaModel(
    corpus=params["gensim_corpus"],
    id2word= params["id2word"],
    chunksize= params["chunksize"],
    alpha='auto',
    eta='auto',
    iterations=params["iterations"],
    num_topics= params["num_topics"],
    passes= params["passes"]
    )

    return lda_model

def visualise(lda_model,params,dictionary,filename):
    
    vis_data = gensimvis.prepare(lda_model, params["gensim_corpus"], dictionary)
    #pyLDAvis.display(vis_data)
    pyLDAvis.save_html(vis_data, f'./Lyrics_LDA_k_{filename}_'+ str(params["num_topics"]) +'_'+str(params["chunksize"]) + '_final.html')

def bifurcate_df(df, param):

    uniq_vals = list(set(df[param].to_list()))
    ans = []

    for val in uniq_vals:
        df_temp = df[df[param] == val]
        ans.append(df_temp)

    return ans

def get_topic_back(gensim_corpus, lda):

    topics = []
    for doc in gensim_corpus:
        prob = lda[doc]
        topic = max(prob, key=lambda item: item[1])[0]

        # to match topic number of pyldavis
        topics.append(int(topic)+1)
    
    return topics

# MAIN

if __name__ == "__main__":

    FILENAME = "filtered_comprehensive.csv"
    df = get_df(FILENAME)

    # remove any row with lyrics and gender NA
    df.dropna(subset=["lower_title", "Artist" , "Lyrics","gender", "year"],inplace=True)

    print("Tokenising lyrics")
    # get tokens
    df = tokenize_lyrics(df)

    # df.to_csv('tokenised_comprehensive.csv')

    print("Creating dictionary")
    # get all tokenss
    all_tokens = get_tokens(df)

    # create dictionary
    dictionary = create_dict(all_tokens, 100, 0.8)

    # create bow
    gensim_corpus, id2word = bow_model(dictionary, all_tokens)

    # create params
    params = {
        "gensim_corpus": gensim_corpus,
        "id2word": id2word,
        "chunksize": 1000,
        "iterations": 600,
        "num_topics": 4,
        "passes": 20
    }

    print("Creating LDA model")
    # create lda model
    lda_model = lda_model(params)

    print("Visualising")
    # visualise
    visualise(lda_model,params,dictionary,"test")

    print("Getting Topics...")

    topics  = get_topic_back(gensim_corpus, lda_model)

    df["topic"] = topics

    # Save csv

    df.to_csv(f'topics_'+ str(params["num_topics"]) +'_'+str(params["chunksize"]) + '_final.html')
    df[['title_x','artist_x','year_x','topic','unique_tokens']].to_csv(f'compact_song_topics_'+ str(params["num_topics"]) +'_'+str(params["chunksize"]) + '_final.html')