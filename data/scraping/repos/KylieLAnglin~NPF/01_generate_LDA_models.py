# %%
import os

import pandas as pd
import numpy as np
import string

import gensim
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
from numpy import array
from tqdm import tqdm

from NPF.teachers_and_covid import start
from NPF.library import process_text
from NPF.library import topic_modeling

PASSES = 5
WORDS_TO_VIEW = 10
SEED = 4205

NO_BELOW = [0, 100, 500, 1000]
NO_ABOVE = [0.25, 0.5, 1]
NUM_TOPICS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# %%
df = pd.read_csv(start.CLEAN_DIR + "tweets_relevant.csv")
docs = list(df.text)


# %% Tokenize, lowercase
def make_lower(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


df["tweet_text_clean"] = df.text.apply(make_lower)
df["tweet_text_clean"] = df.text.apply(remove_punctuation)

docs = [
    process_text.process_text_nltk(
        text,
        lower_case=True,
        remove_punct=True,
        remove_stopwords=True,
        lemma=True,
        string_or_list="list",
    )
    for text in df.tweet_text_clean
]

# %%
# remove https
docs_clean = []
for doc in docs:
    new_doc = []
    for term in doc:
        if not "https" in term:
            new_doc.append(term)
    docs_clean.append(new_doc)


# %%

grid = []
for no_below in NO_BELOW:
    for no_above in NO_ABOVE:
        for num_topics in NUM_TOPICS:
            grid.append(
                {"no_below": no_below, "no_above": no_above, "num_topics": num_topics}
            )
len(grid)
# %%

pbar = tqdm(total=len(grid))
coherences = []
for parameters in grid:
    model_name = (
        "topic_"
        + str(parameters["num_topics"])
        + "_no_below_"
        + str(parameters["no_below"])
        + "_no_above_"
        + str(parameters["no_above"])
    )
    newpath = start.RESULTS_DIR + "topic_models/" + model_name + "/"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    pbar.update(1)
    dictionary = Dictionary(docs)

    dictionary.filter_extremes(
        no_below=parameters["no_below"], no_above=parameters["no_above"]
    )

    corpus = [dictionary.doc2bow(doc) for doc in docs]

    lda = gensim.models.LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=parameters["num_topics"],
        passes=PASSES,
        random_state=SEED,
        per_word_topics=True,
    )

    topic_modeling.create_topic_tables(
        lda=lda,
        corpus=corpus,
        dictionary=dictionary,
        tweets_df=df,
        num_topics=parameters["num_topics"],
        folder_path=newpath,
        num_words_to_view=WORDS_TO_VIEW,
    )
    coherences.append(
        CoherenceModel(
            model=lda, corpus=corpus, dictionary=dictionary, coherence="u_mass"
        ).get_coherence()
    )
pbar.close()

models = pd.DataFrame(grid)
models["coherence"] = coherences

models.to_csv(start.RESULTS_DIR + "topic_models_coherence.csv")
# %%
