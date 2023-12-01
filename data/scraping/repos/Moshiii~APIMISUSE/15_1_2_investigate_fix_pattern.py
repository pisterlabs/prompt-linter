from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import hdbscan
import umap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from dotenv import load_dotenv
import os
import json
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
import openai
import numpy as np


def parse_fix_pattern(input_str):
    input_str = input_str.replace("\n", "").lower().replace(
        "fix pattern:", "fix_pattern:")
    if "fix_pattern:" in input_str:
        output_str = input_str.split("fix_pattern:")[1]
    else:
        print("error: fix_pattern: not found")
        output_str = ""
    return output_str


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
data_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\data_all\\misuse_v3_classification_stage_3_result.json"
# example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\manual\\fix_rules.json"
# example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\data_all\\fix_rules.json"
example_path = "C:\@code\APIMISUSE\data\misuse_jsons\\auto_langchain\data_all\\fix_rules_v4_list.json"

example_dict = {}
with open(example_path, encoding="utf-8") as f:
    data = json.load(f)
    for line in data:
        item = {
            "number": line["number"],
            # "change": line["change"].split("\n"),
            "fix_pattern": line["fix_pattern"],
            "APIs": line["APIs"],
        }
        example_dict[line["number"]] = item

data_dict = {}
with open(data_path, encoding="utf-8") as f:
    data = json.load(f)
    for line in data:
        item = {
            "number": line["number"],
            "Symptom": line["Symptom"],
            "Root_Cause": line["Root_Cause"],
            "Action": line["Action"],
            "Element": line["Element"],
        }
        data_dict[line["number"]] = item

# combine two dict
for key in data_dict.keys():
    if key in example_dict.keys():
        data_dict[key]["fix_pattern"] = example_dict[key]["fix_pattern"]

documents = [x["fix_pattern"] for x in data_dict.values()]
ids = [str(x["number"]) for x in data_dict.values()]
print(len(documents))
print(len(ids))
print(documents[0])
print(ids[0])
# exit()

client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                  persist_directory="C:\@code\APIMISUSE\chroma_db"
                                  ))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-ada-002"
)
# API_misuse_fix_pattern_rules
collection = client.get_or_create_collection(
    "fix_rules_pattern_v4_enbedding_900", embedding_function=openai_ef)

# collection.add(
#     documents=documents,
#     ids=ids,
# )
# print("finished")


# sub_category = "Data Conversion Error"
sub_category = "Device Management Error"
sub_category = "Deprecation Management Error"
sub_category = "Algorithm Error"
mask = [x["Root_Cause"] == sub_category for x in data_dict.values()]
# mask = [True for x in data_dict.values()]
print(len(mask))
documents = [data["documents"][i] for i in range(len(data["documents"])) if mask[i]]
print(documents)

# data_dict = {k: v for k, v in data_dict.items(
# ) if v["Root_Cause"] == sub_category}
# fix_patterns = [x["fix_pattern"] for x in data_dict.values()]
# # for x in data_dict.values():
# # print(x["fix_pattern"])


# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(fix_patterns)
# feature_names = vectorizer.get_feature_names_out()


# def extract_top_keywords(tfidf_matrix, feature_names, top_n=5):
#     keywords = []
#     for i in range(len(fix_patterns)):
#         tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
#         sorted_tfidf_scores = sorted(
#             tfidf_scores, key=lambda x: x[1], reverse=True)
#         top_keywords = [word for word, score in sorted_tfidf_scores[:top_n]]
#         keywords.append(top_keywords)
#     return keywords


# top_keywords_per_sentence = extract_top_keywords(tfidf_matrix, feature_names)
# kw_list = []
# for i, sentence in enumerate(fix_patterns):
#     # print(f"Sentence {i + 1}: {sentence}")
#     # print(f"Keywords: {', '.join(top_keywords_per_sentence[i])}\n")
#     kw_list.extend(top_keywords_per_sentence[i])
# # get counter
# kw_counter = Counter(kw_list)
# print(kw_counter.most_common(25))


# data = collection.get(ids = [str(x["number"]) for x in data_dict.values()],include=["embeddings","documents"])
# print(len(data["embeddings"]))
# print(len(data["documents"]))
# print(np.array(data["embeddings"][0]).shape)

# data_embedding = [np.array(x) for x in data["embeddings"]]
# # filter data_embedding by mask
# data_embedding = [data_embedding[i] for i in range(len(data_embedding)) if mask[i]]
# documents = [data["documents"][i] for i in range(len(data["documents"])) if mask[i]]
# print(len(data_embedding))


# umap_embeddings = umap.UMAP().fit_transform(data_embedding)

# # Perform HDBSCAN clustering
# clusterer = hdbscan.HDBSCAN(min_samples=3)
# labels = clusterer.fit_predict(umap_embeddings)
# print(Counter(labels))


# centroids = []
# for i in range(max(labels)+1):
#     cluster = [umap_embeddings[j] for j in range(len(labels)) if labels[j] == i]
#     cluster = np.array(cluster)
#     centroid = np.mean(cluster, axis=0)
#     centroids.append(centroid)
#     # get the closest instance and print the document
#     closest_index = np.argmin(np.linalg.norm(cluster-centroid, axis=1))
#     print(documents[closest_index])
#     # print("")

# plt.scatter(umap_embeddings[:, 0],
#             umap_embeddings[:, 1], c=labels, cmap='rainbow')
# plt.colorbar()
# plt.show()
