import datetime
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
import regex as re
from octis.dataset.dataset import Dataset
DATE = datetime.datetime.today().strftime("%b_%d_%Y")
print(DATE)

print("NMF is running ------------------------")

# cleaning -------------------------------------------
FOLDER_PATH = "---"
df = pd.read_csv(FOLDER_PATH + "calgary_filtered_2020_2023_Jul_17_2023.csv")

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
FILE_PATH = "/home/mitrasadat.mirshafie/Thesis/June 26th - first round/Topic modeling/NMF/"
FILE_NAME = 'corpus.tsv'

docs = df.loc[:, "cleaned_text"]
docs.to_csv(FILE_PATH + "corpus.tsv", index=False, sep='\t', header=False)

data = pd.read_csv(FILE_PATH + "corpus.tsv", header=None)

dataset = Dataset()
dataset.load_custom_dataset_from_folder(FILE_PATH)


from octis.models.NMF import NMF
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


def ctm_model_output(i):

    model = NMF(num_topics=i, random_state=42)
    model_output_ctm = model.train_model(dataset)

    topic_diversity = TopicDiversity(topk=10) # Initialize metric
    topic_diversity_score = topic_diversity.score(model_output_ctm)
    print("Topic diversity: "+str(topic_diversity_score))


    # Initialize metric
    npmi = Coherence(texts=dataset.get_corpus(), topk=10, measure='c_npmi')
    npmi_score = npmi.score(model_output_ctm)
    print("Coherence: "+str(npmi_score))



for i in [10, 20, 30, 100]:
    ctm_model_output(i)