# %%
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import praw
import os
from dotenv import load_dotenv
import cohere
load_dotenv()

# %% API Credentials
client_id = os.environ.get('client_id')
client_secret = os.environ.get('client_secret')
user_agent = os.environ.get('user_agent')
username = os.environ.get('user_name')
password = os.environ.get('password')
cohere_API = os.environ.get('cohere_API')


# %%


def create_reddit_object():

    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent,
                         username=username,
                         password=password)

    return reddit


def hierarchical_clustering(corpus, embedder):

    corpus_embeddings = embedder.encode(corpus)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / \
        np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage='average', distance_threshold=0.8)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i+1)
        print("Length:", len(cluster))

    return clustered_sentences


def get_comments(subreddit, comments_type, limit):

    title_list = []
    comments_master = []

    reddit = create_reddit_object()
    subred = reddit.subreddit(subreddit)

    if comments_type == 'hot':
        category = subred.hot(limit=limit)

    elif comments_type == 'new':
        category = subred.new(limit=limit)

    elif comments_type == 'controversial':
        category = subred.controversial(limit=limit)

    elif comments_type == 'top':
        category = subred.top(limit=limit)

    elif comments_type == 'gilded':
        category = subred.gilded(limit=limit)

    x = next(category)

    for i in category:

        title_list.append(i.title)
        comment_list = []

        for j in i.comments:

            try:
                comment_list.append(j.body)
            except AttributeError:
                continue

        comments_master.append(comment_list)

    return title_list, comments_master


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range,
                            stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j])
                           for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def get_top_n(docs_df, n):

    top_n_labels = []

    for i in range(len(docs_df)):
        lab = docs_df['Topic'][i]
        topic = top_n_words[lab]
        top_n = []
        for j in topic[:n]:
            top_n.append(j[0])

        top_n_labels.append(' || '.join(top_n))

    return top_n_labels


def preprocess(comments):

    comment_list = []

    for comment in comments:
        if comment == '[deleted]' or comment == '[removed]':
            print('found')
            continue

        else:
            comment_list.append(comment)

    return comment_list


# %% Get Comments
titles, comments = get_comments('PersonalFinanceCanada', 'top', limit=6)

# %%  Perform Clustering
embedder = SentenceTransformer('all-MiniLM-L6-v2')
articles = []

for title, comment in zip(titles, comments):

    comments_preprocess = preprocess(comment)

    clustered_sentences = hierarchical_clustering(
        comments_preprocess, embedder)

    clustered_text = clustered_sentences.values()
    clustered_labels = clustered_sentences.keys()

    clusters = []
    label_num = []

    for text, label in zip(clustered_text, clustered_labels):

        for i in text:
            clusters.append(i)
            label_num.append(label)

    docs_df = pd.DataFrame(clusters, columns=["Doc"])
    docs_df['Topic'] = label_num
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(
        ['Topic'], as_index=False).agg({'Doc': ' '.join})

    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(clusters))

    top_n_words = extract_top_n_words_per_topic(
        tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df)

    cluster_label_col = get_top_n(docs_df, 10)

    docs_df['Topic Label'] = cluster_label_col

    articles.append(docs_df)
# %%
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
for article in articles:
    cluster_sum = []
    cluster_label = []
    for cluster in np.unique(article['Topic']):

        same_cluster = article[article['Topic'] == cluster]
        combined = ".".join(same_cluster['Doc'])

        print("length:", len(combined))

        try:
            summary = summarizer(combined, max_length=250,
                                 min_length=30, do_sample=False)
        except IndexError:
            print("Index Error")
            continue
        cluster_sum.append(summary)
        cluster_label.append(cluster)

# %%
df_sum = pd.DataFrame({'summary': cluster_sum, 'label': cluster_label})

# %% Using Cohere
co = cohere.Client(cohere_API)
article = articles[0]
cluster = np.unique(article['Topic'])[0]
cluster_sum = []
cluster_label = []
same_cluster = article[article['Topic'] == cluster]
combined = ".".join(same_cluster['Doc'])
prompt = "Passage:" + combined + "TLDR:"
# prompt = f"""Passage:I hate to keep saying this but maybe this is a good story for CTV News consumer reports. It might get the issue out further and you could start a petition then go to your local government.  u/PatForanCTVNews.In Quebec we have DuProprio, which is actually really popular for selling without an agent. Is there no such thing in the ROC?.Leave a google review for the agent, the only way things are going to change is if they fear for their brand..It’d be great if someone or OP shared a letter they sent to their MP asking for transparency in the real estate industry, that way we can all use it as a template to send to our respectful MPs.I am sorry for this. REs are a scam. The only thing we can do is get on our MPPs to fix legislation and take power away from them..Canada has a long and proud history of monopolies. You can either embrace it or pack your tent and head north..Call your mp, both provincial and federal, call your mayor, call your premiere, start a petition, tell your friends. There needs to be change..Hi highly suggest and hope u share ur story to CBC marketplace. They find issues for consumers and really make the government act - cus the gov just doesnt care enough and slow to act without a little bad press on them..The protectionism we have in some industries in Canada is a freaking joke!  \nWe're talking about the single largest transaction that most people would make in their lifetime and the entire process is obfuscated AF TLDR:"""

response = co.generate(
    prompt=prompt,
    max_tokens=100,
)
print(response.generations[0].text)
# Response
# a system of hidden rules, massive costs, and massive conflicts of interest. Call your MP..In Toronto we had the Competition Bureau take on the real estate boards and they won. Maybe something similar could be done in the rest of the country. At least get the issue in front of government..Is there anything we can do as a group to sue the real estate board or get a class action lawsuit going? That would be the only way to change things..In this country you are paying for the


# %%
