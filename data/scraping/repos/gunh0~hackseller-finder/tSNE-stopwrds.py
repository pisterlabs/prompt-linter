# %%
from collections import defaultdict
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from textblob import Word
import sys
# !{sys.executable} -m spacy download en
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import spacy
import logging
import warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('../merged_text_PUBG.csv')

# LIMIT = 10
# df = df[:LIMIT]
print(df.shape)
df
# %%
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text):
    return " ".join([Word(word).lemmatize() for word in re.sub("[^A-Za-z0-9]+", " ", text).lower().split() if word not in stopword])


stopword = stopwords.words('english')
df['clean-text'] = df.text.apply(lambda row: clean_text(str(row)))
# %%


perp_components = defaultdict(dict)
for i in [10, 30, 50, 100]:
    for ngram in [(1, 1), (1, 2), (1, 3), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]:
        vectorizer = TfidfVectorizer(ngram_range=ngram, max_features=2**10)
        text_to_vector = vectorizer.fit_transform(df.text.values.astype('U'))
        print("Ngram ", ngram)
        print("Perplexity ", i)
        X_embedded = TSNE(perplexity=i).fit_transform(text_to_vector)
        ngram_str = str(ngram[0])+"_"+str(ngram[1])
        perp_components[i][ngram_str] = X_embedded
        # sns settings
        sns.set(rc={'figure.figsize': (8, 8)})
        # colors
        palette = sns.color_palette(
            "hls", len(set(df.category.values.tolist())))
        y = df.category.values.tolist()
        # plot
        sns.scatterplot(X_embedded[:, 0],
                        X_embedded[:, 1], hue=y, palette=palette)
        # sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend=False, palette=palette)
        title = "t-SNE- JS group - TfIdf - " + \
            ngram_str+"- tSNE perplexity - "+str(i)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.savefig(title)
        plt.title(title)
        plt.show()

# %%
