import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from denseclus import DenseClus
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

from copy import deepcopy
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import ConversationChain
from sklearn.metrics import silhouette_score

os.environ["OPENAI_API_KEY"] = "your-api-key"


class Outlaw:
    def __init__(self):
        chat_model = ChatOpenAI(model_name="gpt-4-1106-preview")
        self.convo = ConversationChain(llm=chat_model)
        # self.df = pd.read_csv(file_path)
        # self.dummy_df = pd.get_dummies(self.df)
        # self.desc = open(desc_path, "r").read()

    def set_data(self, df_path, desc_path):
        self.df = pd.read_csv(df_path)
        self.dummy_df = pd.get_dummies(self.df)
        self.desc = open(desc_path, "r").read()

    def clustering(self, use_kmean=True):
        print("-- Clustering --")
        clf = DenseClus()
        clf.fit(self.df)
        if use_kmean:
            kmeans = KMeans(n_clusters=8, random_state=42, n_init=20)
            kmeans.fit(clf.mapper_.embedding_)
            self.label = kmeans.predict(clf.mapper_.embedding_)
        else:
            self.label = clf.score()

        # Result file
        result = deepcopy(self.df)
        result["class"] = self.label
        result.to_csv("result.csv", index=False)
        self.result = result

    def train_tree(self):
        print("-- Tree training --")
        # train test split
        xtrain, xtest, ytrain, ytest = train_test_split(
            self.dummy_df,
            self.label,
            test_size=0.2,
            random_state=42,
            stratify=self.label,
        )

        # declare a decision tree classifier instance and train it on the available data
        clt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(xtrain, ytrain)
        rules = sklearn.tree.export_text(clt)

        # evaluate how well our classifier performs
        y_pred = clt.predict(xtest)
        acc = accuracy_score(ytest, y_pred)
        pre = precision_score(ytest, y_pred, average="weighted")
        rec = recall_score(ytest, y_pred, average="weighted")
        f1 = f1_score(ytest, y_pred, average="weighted")
        self.acc = [acc, pre, rec, f1]

        for i in range(self.df.shape[1]):
            rules = rules.replace(f"feature_{i} ", self.dummy_df.columns[i] + " ")
        self.rules = rules

    def tune_clustering(self, emb, k_min=3, k_max=10):
        print("-- Tuning --")
        clf = DenseClus()
        clf.fit(self.df)
        max_silhouette = 0
        k = 0
        for n_clusters in range(k_min, k_max):
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_model.fit(clf.mapper_.embedding_)
            labels = kmeans_model.labels_
            cluster_centers = kmeans_model.cluster_centers_
            silhouette_avg = silhouette_score(clf.mapper_.embedding_, labels)

            if silhouette_avg > max_silhouette:
                max_silhouette = silhouette_avg
                k = n_clusters

    def extract_stat(self):
        print("-- Statistics extracting --")
        stat_df = self.result.groupby("class").describe().xs("mean", level=1, axis=1)

        # numerical feature
        h, w = stat_df.shape
        col_name = stat_df.columns
        stat_num = ""
        for i in range(h):
            txt = f"Cluster {i}: "
            for j in range(w):
                txt += col_name[j]
                txt += " = "
                txt += str(round(stat_df.iloc[i, j], 2))
                txt += ", "
            stat_num = stat_num + txt + "\n"

        # categorical feature
        cat_df = self.result.select_dtypes("object")
        col_name = cat_df.columns
        stat_cat = ""
        for i in range(h):
            txt = f"\nCluster {i}:"
            for j in range(cat_df.shape[1]):
                dic = (
                    self.result[self.result["class"] == i]
                    .select_dtypes("object")[col_name[j]]
                    .value_counts(normalize=True)
                    .to_dict()
                )
                txt += f"\n{col_name[j]} :"
                for k in dic:
                    txt += f" {k} = {round(dic[k],2)}, "
            stat_cat = stat_cat + txt + "\n"
            self.stat_num = stat_num
            self.stat_cat = stat_cat

    def start(self):
        self.clustering()
        # model.label = pd.read_csv('/content/result.csv')['class'].values
        # model.result = pd.read_csv('/content/result.csv')

        self.train_tree()
        self.extract_stat()

    def response(self):
        print("-- LLM generating --")
        text = f"""You are business analyst expert of the bank and your responsibility is to provide suggestion how bank should treat each customer group in detail.
    you will be given the information needed for the task, those are decision tree based clustering providing rules used to cluster each group, feature statistics (mean or proportion) of each group and feature description to explain what each feature is.

    -- Needed information --
    Decision tree rules :
    {self.rules}

    Numerical feature statistics :
    {self.stat_num}

    Categorical feature statistics :
    {self.stat_cat}

    Feature description :
    {self.desc}

    Please provide policy suggestion step by step about how the bank should do with each group in detail.
    The response should be in the format below.

    Class 0
    Characteristics :
    - (Characteristics 1)
    - (Characteristics 2)
    ... and so on

    Policy suggestion :
    - (Policy suggestion 1)
    - (Policy suggestion 2)

    (Always repeat this format for all class)
    """
        return self.convo(text)["response"]

    def chat(self, user_input):
        return self.convo(user_input)["response"]
