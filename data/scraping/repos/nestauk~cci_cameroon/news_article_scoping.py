# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.2
#   kernelspec:
#     display_name: cci_cameroon
#     language: python
#     name: cci_cameroon
# ---

# %% [markdown]
# # News article scoping

# %%
from bertopic import BERTopic

# %%
# Import libraries
import pandas as pd
import numpy as np
from newsfetch.google import google_search
from newsfetch.news import newspaper
import newspaper as nwp
from wordcloud import WordCloud
import re
import gensim
from gensim.utils import simple_preprocess
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
import gensim.corpora as corpora

# from bertopic import BERTopic
from pprint import pprint
from gensim.models import CoherenceModel

# %%
# Articles used for testing API's
testing_urls = {
    "BBC news example": "https://www.bbc.co.uk/news/world-48810070",
    "Guardian post example article": "https://theguardianpostcameroon.com/index.php/2021/11/10/fako-meme-divisionsincessant-land-sales-population-growth-sources-of-tension-among-settlers-indigenes/",
    "Cameroon concord example article": "https://www.cameroonconcordnews.com/first-lady-chantal-biya-meets-wife-of-un-secretary-general/",
    "Cameroon tribune example article": "https://www.cameroon-tribune.cm/article.html/43573/fr.html/concertation-minesup-enseignants-on-fait-le-bilan",
}

# %% [markdown]
# ### News-fetch
# - Wasn't able to get google search feature working (search for keyword in news website)
# - Newspaper feature works well where it gives various information for an input article
#     - Information returned in dict format with headline, keywords, article text, language ect
#     - Different information seems to be missing across articles but the infomation that seems consistent (with the tests) are: headline, article and date published
#         - Caveat being for Cameroon concord the article text appears to be the cookie message

# %%
# Test 'newspaper' feature for single article
news_bbc = newspaper(testing_urls["BBC news example"]).get_dict
news_guardian_post = newspaper(testing_urls["Guardian post example article"]).get_dict
cameroon_concord = newspaper(testing_urls["Cameroon concord example article"]).get_dict
cameroon_tribune = newspaper(testing_urls["Cameroon tribune example article"]).get_dict

# %%
# Testing with article suggestion from news-fetch tutorial
print(news_bbc["headline"])
print(news_bbc["keyword"][0])
print(news_bbc["date_publish"])
print(news_bbc["author"])
print(news_bbc["category"])
print(news_bbc["article"])
print("SUMMARY: " + news_bbc["summary"])

# %%
# Testing with article from Cameroon post
print(news_guardian_post["headline"])
print(news_guardian_post["keyword"][0])
print(news_guardian_post["date_publish"])
print(news_guardian_post["author"])
print(news_guardian_post["category"])
# print(news_guardian_post['article'])
print("SUMMARY: " + news_guardian_post["summary"])

# %%
# Testing with article from Cameroon Concord
print(cameroon_concord["headline"])
print(cameroon_concord["keyword"][0])
print(cameroon_concord["date_publish"])
print(cameroon_concord["author"])
print(cameroon_concord["category"])
print(cameroon_concord["article"])
print("SUMMARY: " + cameroon_concord["summary"])

# %%
# Testing with article from Cameroon tribune
print(cameroon_tribune["headline"])
print(cameroon_tribune["keyword"][0])
print(cameroon_tribune["date_publish"])
print(cameroon_tribune["author"])
print(cameroon_tribune["category"])
print(cameroon_tribune["article"])
print("SUMMARY: " + cameroon_tribune["summary"])

# %% [markdown]
# Trialing news search returns error on chrome browser

# %%
# google = google_search('Alcoholics Anonymous', 'https://timesofindia.indiatimes.com/')

# %% [markdown]
# ### Newspaper

# %%
guardian_post = nwp.build(
    "https://theguardianpostcameroon.com/", memoize_articles=False
)
concordnews = nwp.build(
    "https://www.cameroonconcordnews.com/category/news/cameroon-news/",
    memoize_articles=False,
)
tribune = nwp.build("https://www.cameroon-tribune.cm/", memoize_articles=False)

# %%
# Gave URLS from all parts of the site
# guardian_cameroon = nwp.build('https://www.theguardian.com/world/cameroon', memoize_articles=False)

# %%
tribune_urls = []
for article in tribune.articles:
    tribune_urls.append(article.url)

# %%
tribune_urls[0]

# %% [markdown]
# ### Combining for collecting article info

# %%
# Get pre-collected list of URLs
guardian_post_urls = pd.read_csv("guardian_post_urls.csv", header=None, usecols=[0])
guardian_post_urls = list(guardian_post_urls[0])

# %%
guardian_post_data = []
for url in guardian_post_urls:
    post = newspaper(url).get_dict
    guardian_post_data.append(post)

# %%
guardian_post_articles = [d["article"] for d in guardian_post_data]
guardian_post_titles = [d["headline"] for d in guardian_post_data]
guardian_post_keywords = [d["keyword"] for d in guardian_post_data]

# %%
df = pd.DataFrame(
    list(zip(guardian_post_articles, guardian_post_titles, guardian_post_keywords)),
    columns=["text", "title", "keyword"],
)

# %%
# Remove punctuation
df["text_processed"] = df["text"].map(lambda x: re.sub("[,./!?]", "", x))
# Convert the titles to lowercase
df["text_processed"] = df["text_processed"].map(lambda x: x.lower())
# Print out the first rows of papers
df["text_processed"].head()

# %%
# Join the different processed titles together.
long_string = ",".join(list(df["text_processed"].values))
# Create a WordCloud object
wordcloud = WordCloud(
    background_color="white", max_words=5000, contour_width=3, contour_color="steelblue"
)
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()

# %%
stop_words = stopwords.words("english")
stop_words.extend(["from", "subject", "re", "edu", "use", "cameroon"])


# %%
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


# %% [markdown]
# #### LDA topic modelling

# %%
data = df.text_processed.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

# %%
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])

# %%
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=20,
    random_state=100,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha="auto",
    per_word_topics=True,
)

# Print the Keyword in the 10 topics
doc_lda = lda_model[corpus]

# %%
pprint(lda_model.print_topics())

# %%
# Compute Perplexity
print(
    "\nPerplexity: ", lda_model.log_perplexity(corpus)
)  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(
    model=lda_model, texts=data_words, dictionary=id2word, coherence="c_v"
)
coherence_lda = coherence_model_lda.get_coherence()
print("\nCoherence Score: ", coherence_lda)

# %%
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# %%
# Next: look at free trial version: https://newsapi.ai/
