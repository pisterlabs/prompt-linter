# https://github.com/MaartenGr/BERTopic_evaluation/blob/main/notebooks/Evaluation.ipynb

import pandas as pd
import numpy as np
from bertopic import BERTopic
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import pickle

import collections
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/")
ROOT.exists()
ARXIV_PATH = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/arxivdata/arxiv-metadata-oai-snapshot.json")
ARXIV_PATH.exists()
ARTIFACTS = ROOT / Path("artifacts/model2M")


def setup(n=10000):
    with open(ARTIFACTS / "sentence_list_full.pkl", "rb") as file:
        sentence_list_full = pickle.load(file)
    with open(ARTIFACTS / "timestamp_list_full.pkl", "rb") as file:
        timestamp_list_full = pickle.load(file)
    embeddings = np.load(ARTIFACTS / "embeddings2M.npy")
    with open(ARTIFACTS / "vocab2m.pkl", "rb") as file:
        vocab = pickle.load(file)
    vocab_reduced = [word for word, frequency in vocab.items() if frequency >= 50]
    reduced_embeddings = np.load(ARTIFACTS / "reduced_embeddings2M.npy")
    reduced_embeddings_2d = np.load(ARTIFACTS / "reduced_embeddings2M_2d.npy")
    clusters = np.load(ARTIFACTS / "hdb_clusters2M.npy")
    return (
        sentence_list_full[0:n],
        timestamp_list_full[0:n],
        embeddings[0:n],
        vocab_reduced[0:n],
        reduced_embeddings[0:n],
        reduced_embeddings_2d[0:n],
        clusters[0:n],
    )


(
    docs,
    timestamps,
    embeddings,
    vocab,
    reduced_embeddings,
    reduced_embeddings_2d,
    clusters,
) = setup(n=10000)

from bertopic import BERTopic

params = {
    "nr_topics": [(i+1) * 10 for i in range(5)],
    "min_topic_size": 15,
    "verbose": True
}

from sentence_transformers import SentenceTransformer
from ScienceNOW.sciencenow.models.trainer import Trainer
from ScienceNOW.sciencenow.models.dataloader import DataLoader

dataset, custom = "arxiv", True
data_loader = DataLoader(dataset)
# data_loader.load_docs()
data, timestamps = docs, timestamps
# timestamps = [str(stamp) for stamp in timestamps]
model = SentenceTransformer("all-mpnet-base-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")
_, timestamps = data_loader.load_docs()
#

embeddings = model.encode(data, show_progress_bar=True)

custom_path = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/ScienceNOW/artifacts/model10k")

# Extract vocab to be used in BERTopic
# vocab = collections.Counter()
# tokenizer = CountVectorizer().build_tokenizer()
# for doc in tqdm(docs):
#     vocab.update(tokenizer(doc))

# with open(custom_path / "vocab.txt", "w") as file:
#     for item in vocab:
#         file.write(item+"\n")
# file.close()

# docs = [doc.replace("\n", " ") for doc in docs]
# assert all("\n" not in doc for doc in docs)

# with open(custom_path / "corpus.tsv", "w") as file:
#     for document in docs:
#         file.write(document + "\n")
# file.close()

# now run every set of parameters three times to help combat randomness introduced by UMAP
from tqdm import tqdm
results = []
for i in tqdm(range(3)): # repeat every experiment three times
    params = {
        "nr_topics": [(j+1)*10 for j in range(5)], # 10 - 50 topics => 15 runs total
        "min_topic_size": 15, # 48 minutes computation time in total for 10k docs 10 time bins
        "verbose": True,
    }
    trainer = Trainer(dataset=dataset,
                      model_name="BERTopic",
                      params=params,
                      bt_embeddings=embeddings,
                      custom_dataset=custom,
                      bt_timestamps=timestamps,
                      topk=5,
                      bt_nr_bins=10,
                      verbose=True)
    res = trainer.train(save=f"DynamicBERTopic_arxiv_{i+1}")
    results.append(res)

# params = {
#     "nr_topics": 50,
#     "min_topic_size": 15,
#     "verbose": True,
#     #"embeddings": embeddings,
#     #"timestamps": timestamps,
#     #"topk": 5,
#     "verbose": True,
# }

# model = BERTopic(**params)
# topics, _ = model.fit_transform(data, embeddings)
# # prebin timestamps
# nr_bins = 10
# df = pd.DataFrame({"Doc": docs, "Timestamp": timestamps})
# df["Timestamp"] = pd.to_datetime(df["Timestamp"], infer_datetime_format=True)
# df["Bins"] = pd.cut(df.Timestamp, bins=nr_bins)
# df["Timestamp"] = df.apply(lambda row: row.Bins.left, 1)
# timestamps = df.Timestamp.tolist()
# documents = df.Doc.tolist()

# timestamps = [pd.to_datetime(ts).value for ts in timestamps]
# timestamps = [str(timestamp) for timestamp in timestamps]


# topics_over_time = model.topics_over_time(
#     docs = data, # docs
#     # topics, # timestamps
#     timestamps = timestamps, # nr_bins 
#     nr_bins=10, # dt_format
#     evolution_tuning=False, # evo_tuning
#     global_tuning=False, # global_tuning
# )


# TODO: Wrap everything to only require configs for experiments
# TODO: Set up Datasets: NLP only, Dynamic NLP + Standard, Online NLP + Standard
# TODO: Experiment with introducing noise papers + bioarxiv papers
# TODO: Format evaluation in a presentable way.
# TODO: Hierarchical Model for coarse overview and fine grained models for every subcategory

class ArxivProcessor:
    def __init__(self, path=ARXIV_PATH, sorted=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = path
        self.sorted = sorted
    def _get_data(self):
        with open(self.path, "r") as file:
            for line in file:
                yield line
    def _process_data(self):
        date_format = "%a, %d %b %Y %H:%M:%S %Z"
        data_generator = self._get_data()
        ids, titles, abstracts, cats, refs, timestamps = [], [], [], [], [], []
        for paper in tqdm(data_generator):
            paper_dict = json.loads(paper)
            ids.append(paper_dict["id"])
            titles.append(paper_dict["title"])
            abstracts.append(paper_dict["abstract"])
            cats.append(paper_dict["categories"])
            refs.append(paper_dict["journal-ref"])
            timestamps.append(paper_dict["versions"][0]["created"])  # 0 should be v1
        # process timestamps so that they can be sorted
        timestamps_datetime = [
            datetime.strptime(stamp, date_format) for stamp in timestamps
        ]
        out = pd.DataFrame(
            {
                "id": ids,
                "title": titles,
                "abstract": abstracts,
                "categories": cats,
                "references": refs,
                "timestamp": timestamps_datetime,
            }
        )
        if self.sorted:
            return out.sort_values("timestamp", ascending=False)
        return out


arxiv = ArxivProcessor()
df = arxiv._process_data()

# Every Arxiv paper falls into one of 6 "Level 0" topics
# Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics

# every paper is already assined one ore more subcategories
unique_cats = df["categories"]
# unroll categories 
unique_cats = [cat.split(" ") for cat in unique_cats]
# unpack nested list and filter to obtain uniques
unique_cats = [j for i in unique_cats for j in i]
unique_cats  = set(unique_cats)
# 176 Level 1 Topics
# AI, Computation and Language, Computational Complexity, etc.

taxonomy = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ru25jan4/taxonomy.txt")

with open(taxonomy, "r") as file:
    arxiv_taxonomy = [line.rstrip() for line in file]

arxiv_taxonomy_list = [line.split(":") for line in arxiv_taxonomy]
arxiv_taxonomy = {line[0]: line[1].lstrip() for line in arxiv_taxonomy_list}

keys = set(arxiv_taxonomy.keys())
other = unique_cats - keys
# 28 categories that are not found in the arxiv taxonomy
# are any papers tagged as ONLY falling into these categories?
outlier_papers = df[df["categories"].isin(other)]
len(outlier_papers) # 106849, a relatively big chunk

# as none of the outlier categories are really needed for downstream analysis we will remove them from
# the set of papers to be analyzed

# lets start with a sample of 10000 papers and see if we can recover the 6 original level 0 cats
nonoutlier_df = df[df["categories"].isin(keys)]
temp = nonoutlier_df[nonoutlier_df["categories"].isin(other)]
len(temp) # 0
sample = nonoutlier_df.sample(n=10000)
from bertopic import BERTopic
data = sample["abstract"].tolist()
timestamps = sample["timestamp"].tolist()
l0_categories = ["Physics", "Mathematics", "Computer Science", "Quantitative Biology", "Quantitative Finance", "Statistics"]

label_map = {
    "stat": "Statistics",
    "q-fin": "Quantitative Finance",
    "q-bio": "Quantitative Biology",
    "cs": "Computer Science",
    "math": "Mathematics"
} # anything else is physics

# simply map them to the first L0 topic they fall into
catstring = [x.split(" ")[0].split(".")[0] for x in sample["categories"]]
sample_categories = [label_map[x] if x in label_map.keys() else "Physics" for x in catstring]

label_to_numeric = {
    "Physics": 0, 
    "Mathematics": 1,
    "Computer Science": 2, 
    "Quantitative Biology": 3, 
    "Quantitative Finance": 4,
    "Statistics":5,
}

numeric_to_label = {
    -1: "Outlier",
    0: "Physics", 
    1: "Mathematics",
    2: "Computer Science", 
    3: "Quantitative Biology", 
    4: "Quantitative Finance",
    5: "Statistics",
}

numeric_sample_categories = [label_to_numeric[x] for x in sample_categories]

topic_model = BERTopic(verbose=True, nr_topics=7).fit(data, y=numeric_sample_categories)

topics = topic_model.topics_

# reduce outliers
new_topics = topic_model.reduce_outliers(data, topics , strategy="c-tf-idf", threshold=0.1)
new_topics = topic_model.reduce_outliers(data, new_topics, strategy="distributions")

from collections import Counter

original = Counter(numeric_sample_categories)
modeled = Counter(topics)
reduced = Counter(new_topics)

# lets visualize results and then obtain clustering accuracy metrics
# wrap this into the evaluation loop 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english")
topic_model.update_topics(data, vectorizer_model=vectorizer_model)
topic_model.update_topics(data, topics=new_topics)
topic_model.set_topic_labels(numeric_to_label)

# visualizing documents
from sentence_transformers import SentenceTransformer
from umap import UMAP
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(data, show_progress_bar=True)
topic_model = BERTopic(verbose=True, nr_topics=7).fit(data, y=numeric_sample_categories, embeddings=embeddings)
topics = topic_model.topics_
# reduce outliers
new_topics = topic_model.reduce_outliers(data, topics , strategy="c-tf-idf", threshold=0.1)
new_topics = topic_model.reduce_outliers(data, new_topics, strategy="distributions")
topic_model.update_topics(data, vectorizer_model=vectorizer_model)
topic_model.update_topics(data, topics=new_topics)
topic_model._outliers = 0
topic_model.set_topic_labels(numeric_to_label)
fig = topic_model.visualize_documents(data, embeddings=embeddings, custom_labels=True)
fig.write_html("L0_umap.html")
# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
fig = topic_model.visualize_documents(data, reduced_embeddings=reduced_embeddings,custom_labels=True)
fig.write_html("L0_umap_reduced.html")



fig = topic_model.visualize_barchart(custom_labels=True)
fig.write_image("L0_barchart.png")

fig = topic_model.visualize_topics(custom_labels=True)
fig.write_image("L0_intertopic_distance.png")
fig.write_html("L0_intertopic_distance.html")

fig = topic_model.visualize_heatmap(custom_labels=True)
fig.write_image("L0_heatmap.png")
fig.write_html("L0_heatmap.html")

# visualizing individual topic representations:
topics_per_class = topic_model.topics_per_class(data, classes=new_topics)
fig = topic_model.visualize_topics_per_class(topics_per_class, custom_labels=True)
fig.write_html("L0_topics_per_class.html")

# Create topics over time
model_ot = BERTopic(verbose=True, nr_topics=7).fit(data, y=numeric_sample_categories)
topics_over_time = model_ot.topics_over_time(data, timestamps, nr_bins=50)
topics = model_ot.topics_
# reduce outliers
new_topics = model_ot.reduce_outliers(data, topics , strategy="c-tf-idf", threshold=0.1)
new_topics = model_ot.reduce_outliers(data, new_topics, strategy="distributions")

vectorizer_model = CountVectorizer(stop_words="english")
model_ot.update_topics(data, vectorizer_model=vectorizer_model)
model_ot.set_topic_labels(numeric_to_label)

fig = model_ot.visualize_topics_over_time(topics_over_time, custom_labels=True)
fig.write_image("L0_over_time.png")
fig.write_html("L0_over_time.html")





#### Online Model experiments on computer science papers
# 
# filtering for just computer science categories
cs_keys = {key for key in keys if key.startswith("cs")}
cs_keys.remove("cs.IT") # was not in data frame
len(cs_keys)
cs_values = list(map(arxiv_taxonomy.get, cs_keys))
cs_map = {key: value for key, value in zip(cs_keys, cs_values)}

cs_df = nonoutlier_df[nonoutlier_df["categories"].isin(cs_keys)]
cs_data = cs_df["abstract"].tolist()
cs_timestamps = cs_df["timestamp"].tolist()

# finding out how many papers are present in the smallest group
cs_groups = cs_df.groupby(cs_df["categories"])
cs_groupcounts = cs_groups.size()
# the category with the smallest number of members is cs.GL with 69 members
# every other category has at least 250 members

numeric_to_label = {key: value for key, value in zip(range(len(cs_keys)), cs_values)}
numeric_to_label[-1] = "Outlier"
label_to_numeric = {key: value for key, value in zip(cs_values,range(len(cs_keys)))}
label_to_numeric["Outlier"] = [-1]

cs_categories = [cs_map[x] for x in cs_df["categories"].tolist()]
numeric_cs_categories = [label_to_numeric[x] for x in cs_categories]

# precalculating embeddings
cs_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cs_embeddings = cs_embedding_model.encode(cs_data, show_progress_bar=True)
cs_umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

topic_model = BERTopic(verbose=True,
    nr_topics=len(set(cs_df["categories"]))+1,
    umap_model=cs_umap_model,
    min_topic_size=50,
    ).fit(cs_data, y=numeric_cs_categories, embeddings=cs_embeddings)

topics = topic_model.topics_

# reduce outliers
new_topics = topic_model.reduce_outliers(cs_data, topics , strategy="c-tf-idf", threshold=0.1)
new_topics = topic_model.reduce_outliers(cs_data, new_topics, strategy="distributions")
# lets visualize results and then obtain clustering accuracy metrics
# wrap this into the evaluation loop 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english")
topic_model.update_topics(cs_data, vectorizer_model=vectorizer_model)
topic_model.update_topics(cs_data, topics=new_topics)
topic_model.set_topic_labels(numeric_to_label)

from collections import Counter
original = Counter(numeric_sample_categories)
modeled = Counter(topics)
reduced = Counter(new_topics)

# we could simply model all the outliers again in an unsupervised manner
# and see if meaningful clusters can be found

# visualize full cs model
topic_model._outliers = 0
fig = topic_model.visualize_documents(cs_data, embeddings=cs_embeddings, custom_labels=True)
fig.write_html("L1_cs_umap.html")

# remove stop words
vectorizer_model = CountVectorizer(stop_words="english")
topic_model.update_topics(cs_data, vectorizer_model=vectorizer_model)
topic_model.set_topic_labels(numeric_to_label)

fig = topic_model.visualize_barchart(custom_labels=True)
fig.write_image("L1_cs_barchart.png")
fig.write_html("L1_cs_barchart.html")

fig = topic_model.visualize_topics(custom_labels=True)
fig.write_image("L1_cs_intertopic_distance.png")
fig.write_html("L1_cs_intertopic_distance.html")

fig = topic_model.visualize_heatmap(custom_labels=True)
fig.write_image("L1_cs_heatmap.png")
fig.write_html("L1_cs_heatmap.html")

# visualizing individual topic representations:
topics_per_class = topic_model.topics_per_class(cs_data, classes=new_topics)
fig = topic_model.visualize_topics_per_class(topics_per_class, custom_labels=True)
fig.write_html("L1_cs_topics_per_class.html")

# Create topics over time
#model_ot = BERTopic(verbose=True, nr_topics=7).fit(data, y=numeric_cs_categories)
topics_over_time = topic_model.topics_over_time(cs_data, cs_timestamps, nr_bins=50)
topics = model_ot.topics_
# reduce outliers
new_topics = model_ot.reduce_outliers(data, topics , strategy="c-tf-idf", threshold=0.1)
new_topics = model_ot.reduce_outliers(data, new_topics, strategy="distributions")

vectorizer_model = CountVectorizer(stop_words="english")
topic_model.update_topics(cs_data, vectorizer_model=vectorizer_model)
model_ot.set_topic_labels(numeric_to_label)

fig = topic_model.visualize_topics_over_time(topics_over_time, custom_labels=True)
fig.write_image("L1_cs_over_time.png")
fig.write_html("L1_cs_over_time.html")
# TODO: fix custom labels, as multimedia is incorrect for sure


#### Stream Clustering
# Experiment with Online Models & DenStream https://riverml.xyz/0.14.0/api/cluster/DenStream/

len(cs_df)
# let's assume documents arrive in chunks of 1000 documents each
cs_doc_chunks = [cs_data[i:i+1000] for i in range(0, len(cs_data), 1000)]
# 184 chunks of 1k docs each

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

# Prepare sub-models that support online learning
from river import stream
from river import cluster

class River:
    def __init__(self, model):
        self.model = model
    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            self.model = self.model.learn_one(umap_embedding)
        labels = []
        for umap_embedding, _ in stream.iter_array(umap_embeddings):
            label = self.model.predict_one(umap_embedding)
            labels.append(label)
        self.labels_ = labels
        return self

# Using DBSTREAM to detect new topics as they come in
cluster_model = River(cluster.DBSTREAM())
vectorizer_model = OnlineCountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

# Prepare model
online_topic_model = BERTopic(
    hdbscan_model=cluster_model, 
    vectorizer_model=vectorizer_model, 
    ctfidf_model=ctfidf_model,
)

online_topics = []
# fitting the first 1000 documents
for doc_chunk in tqdm(cs_doc_chunks[0:30]):
    online_topic_model.partial_fit(doc_chunk)
    topics = online_topic_model.topics_
    online_topics.extend(topics)

# update the topics attribute for further downstream applications
online_topic_model.topics_ = online_topics
import pickle
# with open("online_topics.pkl", "wb") as filepath:
#     pickle.dump(online_topics, filepath)

with open("online_topics.pkl", "rb") as filepath:
    online_topics = pickle.load(filepath)

test = online_topic_model.get_topics()


### Visualize Online Model
online_cs_data = [x for y in cs_doc_chunks[0:30] for x in y]
online_cs_embeddings = cs_embeddings[0:30000]
fig = online_topic_model.visualize_documents(online_cs_data, custom_labels=True)
fig.write_html("online_cs_umap.html")

# remove stop words
vectorizer_model = CountVectorizer(stop_words="english")
topic_model.update_topics(cs_data, vectorizer_model=vectorizer_model)
topic_model.set_topic_labels(numeric_to_label)

fig = online_topic_model.visualize_barchart(custom_labels=True)
fig.write_image("L1_cs_barchart.png")
fig.write_html("online_cs_barchart.html")

fig = online_topic_model.visualize_topics(custom_labels=True)
fig.write_image("L1_cs_intertopic_distance.png")
fig.write_html("online_cs_intertopic_distance.html")

fig = online_topic_model.visualize_heatmap(custom_labels=True)
fig.write_image("L1_cs_heatmap.png")
fig.write_html("online_cs_heatmap.html")






























# topic_model = BERTopic(verbose=True)
# topics, probs = topic_model.fit_transform(docs)
# topics_over_time = topic_model.topics_over_time(docs, timestamps, nr_bins=50)
# # runs about 14sec for 10k articles
# # global_tuning and evolutionary_tuning may have additional effects
# tovert_nonglobal = topic_model.topics_over_time(docs, timestamps, global_tuning=False, evolution_tuning=True, nr_bins=20)
# # we should use binning to keep the number of unique topic representations under control
# topics_over_time_20bins = topic_model.topics_over_time(docs, timestamps, nr_bins=20)

# plot = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
# plot.write_image("topics_over_time_20bins.png")
# plot = topic_model.visualize_topics_over_time(topics_over_time_20bins, top_n_topics=10)
# plot.write_image("topics_over_time_10bins.png")

# similar_topics, similarity = topic_model.find_topics("NLP", top_n=5)
# topic_model.get_topic(similar_topics[0])

# # visualize NLP over time
# plot = topic_model.visualize_topics_over_time(topics_over_time, topics=similar_topics)
# plot.write_image("topics_over_time_nlp.png")

# # custom topic labels
# topic_labels = topic_model.generate_topic_labels(nr_words=2,topic_prefix=False)
# topic_model.set_topic_labels(topic_labels)

# ## Evaluation Pipeline
# import nltk
# import gensim
# import gensim.corpora as corpora
# import pandas as pd
# import numpy as np
# from bertopic import BERTopic
# from sentence_transformers import SentenceTransformer
# from umap import UMAP
# from hdbscan import HDBSCAN
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import CountVectorizer
# from bertopic.vectorizers import ClassTfidfTransformer
# from gensim.models.coherencemodel import CoherenceModel


# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
# hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# vectorizer_model = CountVectorizer(stop_words="english")
# ctfidf_model = ClassTfidfTransformer()

# topic_model = BERTopic(
#   embedding_model=embedding_model,    # Step 1 - Extract embeddings
#   umap_model=umap_model,              # Step 2 - Reduce dimensionality
#   hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
#   vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
#   ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
#   nr_topics=10                        # Step 6 - Diversify topic words
# )

# topics, probabilities = topic_model.fit_transform(docs[0:10000])

# embeddings = embedding_model.encode(docs[0:10000], show_progress_bar=True)
# data = docs[0:10000]
# times = timestamps[0:10000]

# from evaluation import Trainer, DataLoader

# # simple wrapper to fit models multiple times for evaluation 
# for i in range(3):
#     params = {
#         "embedding_model": "all-MiniLM-L6-v2",
#         "nr_topics": [(i+1)*10 for i in range(5)],
#         "min_topic_size": 15,
#         "vectorizer_model": vectorizer_model,
#         "umap_model":umap_model,
#         "verbose": True
#     }

#     trainer = Trainer(dataset=dataset,
#                       model_name="BERTopic",
#                       params=params,
#                       bt_embeddings=embeddings,
#                       custom_dataset=custom,
#                       verbose=True)
#     results = trainer.train(save=f"BERTopic_trump_{i+1}")

# from collections import Counter
# tcount = Counter(topics)

# # TOOD: Hyperparameter Optimization & Comparison of Models

