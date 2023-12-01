from joblib import Parallel, delayed
import joblib
import json

from umap import UMAP
import numpy as np
from hdbscan import HDBSCAN
import hdbscan
from tqdm import tqdm
from langchain.llms import OpenAI

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from .doc_to_faiss import make_data_path, make_db_path, data_index

embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)


def p_usage(i):
    p = make_data_path(i)
    db = FAISS.load_local(make_db_path(p), embeddings)
    r = query_usage_scope(db)
    return i, r


def query_usage_scope(doc_db):
    query = """
    what is the personal information used for?
    """

    docs = doc_db.similarity_search(query)
    search_result = "\n".join(d.page_content for d in docs)

    return llm.predict(
        f"""
    what are the uses of the collected personal information?

    Rephrase each usage under 10 words
    Each usage should begin with a verb followed by 3 to 10 words
    return a comma-separated list of usages, 

    fragments of a privacy policy: {search_result}
    usage of personal information:

    """
    )


class UsagePredictor:
    def train(self, training_doc_embeds):
        umap = UMAP(
            n_components=16,
            n_neighbors=10,
        )

        clusterer = HDBSCAN(min_cluster_size=30, min_samples=15, prediction_data=True)

        umap.fit(np.array(training_doc_embeds))
        t_umap = umap.transform(np.array(training_doc_embeds))
        clusterer.fit(t_umap)
        t_cluster, t_strength = hdbscan.approximate_predict(clusterer, t_umap)

        self.train_umap = t_umap
        self.train_cluster = t_cluster
        self.train_strength = t_strength

        self.umap = umap
        self.clusterer = clusterer

    def predict(self, new_doc_faiss):
        splited = query_usage_scope(new_doc_faiss).split(",")
        embeds = embeddings.embed_documents(splited)

        this_umap = self.umap.transform(np.array(embeds))
        this_cluster, _ = hdbscan.approximate_predict(self.clusterer, this_umap)
        return this_cluster, splited, embeds


# plt.scatter(t_umap[:,0], t_umap[:,1], c=t_cluster)

# pd.DataFrame([all_usage_chained, t_cluster])

# summary_df = pd.DataFrame(
#     [doc_ids,
#     doc_tokens,
#     t_cluster.squeeze(),
# ]).T.sort_values(2)

# summary_df.columns = ['doc_id', 'token', 'cluster']
# (
#     summary_df
#     .assign(is_outlier=lambda df: df['cluster']==-1)
#     .groupby('doc_id')['is_outlier']
#     .sum()
#     .value_counts()
# )

def main():
    usages_p = Parallel(n_jobs=4)(delayed(p_usage)(i) for i in tqdm(data_index.keys()))
    usages_p = dict(usages_p)

    doc_ids = []
    doc_tokens = []
    doc_embeds = []
    for k, v in tqdm(usages_p.items()):
        splitted = v.split(",")
        embeds = embeddings.embed_documents(splitted)

        doc_ids.extend([k] * len(splitted))
        doc_tokens.extend(splitted)
        doc_embeds.extend(embeds)

    predictor = UsagePredictor()
    predictor.train(doc_embeds)

    return predictor

    # joblib.dump(predictor, "usage_model.joblib")
