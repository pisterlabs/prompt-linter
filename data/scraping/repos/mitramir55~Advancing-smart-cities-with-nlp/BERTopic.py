import os
import json
import pandas as pd
import csv
import datetime
import dateutil.parser
import unicodedata
import time
import regex as re

from bertopic import BERTopic
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence



DATE = datetime.datetime.today().strftime("%b_%d_%Y")
print(DATE)

print("BERTopic is running ------------------------")

# cleaning -------------------------------------------
FOLDER_PATH = "---"
df = pd.read_csv(FOLDER_PATH + "calgary_filtered_July_25.csv")

df = df[~df.full_text.isna()]
df.reset_index(drop=True, inplace=True)

# stopwords
# nltk stopwords
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk_stopwords = stopwords.words('english')

# spacy stopwords
import spacy
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = list(nlp.Defaults.stop_words.copy())

words_to_remove =  [
    'absolutely', "actually", "there", "they", "if", "that", "arent",
    "wont", "dont", "shouldnt", "cant", "couldnt", "musnt", "dont",
    "etc", "\bor\b", "\bmr\b", "\bmrs\b", # small words like or, mr
    "for", "kinda", "cuz", "really", "ain't", " +(ya) +",
    " +(go) +", " +(come) +",
    "i've", "you've", "we've", "they've",
    "they'd", "i'd", "you'd", "they'd", "we'd",
    "getting", "got", "get", "jus", "shitting", "fucking", "huh", "uh",
    "mean", "did", "told", "tell", "anything", "everything", "thats",
    "pm ", "want", "it"
                    ]

total_stopwords = set(
    nltk_stopwords + spacy_stopwords + words_to_remove
    )



def remove_st(t):
    return " ".join([i for i in t.split() if i not in total_stopwords])



# IMPORTANT!
# this is a different cleaning than what we have for other methods!
# in here, we only want words and nothing else.


def clean(t):
    """
    cleans a given tweet
    """
    t = t.lower()

    # list of some extra words: add gradually to this list
    extra = ["&gt;", "&lt;", "&amp;", "”", "“", "#", '\n', ] # "\"", ','
    for patt in extra: t = re.sub(patt, '', t)

    # URL removal: Go on untill you hit a space
    t = re.sub(r"\S*https?:\S*", "", t)

    # t = remove_specific_emojis(t)

    # remove stopwords
    t = remove_st(t)

    # removes all @s and the text right after them; mentions
    # Question: should we remove hashtags too?
    t = re.sub(r'@\w*\_*' , '', t)

    # not all the punctuation is removed.
    # All except for the ones we have in the brackets
    # first one removes all the punctuations, second one only saves dot and hyphen
    t = re.sub(r'[^a-z]', ' ', t)
    # t = re.sub(r'[^ \w\.\-]', ' ', t)

    # removes all single letters surrounded with space except letters I and a
    t = re.sub(r' +(?![ia])[a-z] +', ' ', t)

    # substitute extra space with only one space
    t = re.sub(r' \s+', ' ', t)

    return t



df.loc[:, "cleaned_text"] = df.loc[:, "full_text"].apply(lambda x: clean(x))
df = df[df.loc[:, 'cleaned_text'].apply(lambda x: len(x)>=14)].reset_index(drop=True)

# removing duplicates
df = df[~df.cleaned_text.duplicated()].reset_index(drop=True)

# for bertopic evaluation we need a tsv file of all the text in one column
FILE_NAME = 'corpus.tsv'

docs = df.loc[:, "cleaned_text"]
docs.to_csv(FOLDER_PATH + "corpus.tsv", index=False, sep='\t', header=False)
# data = pd.read_csv(FOLDER_PATH + "corpus.tsv", header=None, sep='\t')

# dataset
data = Dataset()
data.load_custom_dataset_from_folder(FOLDER_PATH)
all_words = [word for words in data.get_corpus() for word in words]



def bertopic_differ_n_topics(nr_topics):

    print('nr_topics = ', nr_topics, ' ------------------------')
    # topics bert
    topic_model = BERTopic(nr_topics = nr_topics)
    
    start = time.time()
    topics, probs = topic_model.fit_transform(docs)
    end = time.time()
    computation_time = float(end - start)
    print('computation_time = ', computation_time)


    MODEL_PATH = FOLDER_PATH + "BERTopics - {nr_topics}/"
    topic_model.save(MODEL_PATH + f"bertopic_model_nr_topics_{nr_topics}", serialization="safetensors")


    # the format for octis
    
    # what is this doing?
    bertopic_topics = [
        [
            vals[0] if vals[0] in all_words else all_words[0]
            for vals in topic_model.get_topic(i)[:10]
        ]
        for i in range(len(set(topics)) - 1)
    ]

    output_tm = {"topics": bertopic_topics}
    

    topk = 10
    npmi = Coherence(texts=data.get_corpus(), topk=topk, measure="c_npmi")
    topic_diversity = TopicDiversity(topk=topk)

    npmi_s = npmi.score(output_tm)
    diversity = topic_diversity.score(output_tm)

    return npmi_s, diversity


topic_diversity_list_bertopic = []
npmi_score_list_bertopic = []


for k in [10, 20, 30, 100]:
    npmi_s, diversity = bertopic_differ_n_topics(k)
    
    print("Topic diversity: "+str(diversity))
    topic_diversity_list_bertopic.append(diversity)
    
    print("Coherence: "+str(npmi_s))
    npmi_score_list_bertopic.append(npmi_s)


