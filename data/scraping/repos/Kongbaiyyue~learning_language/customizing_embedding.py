from typing import List, Tuple
from networkx import dfs_tree  # for type hints

import numpy as np  # for manipulating arrays
import pandas as pd  # for manipulating data in dataframes
import pickle  # for saving the embeddings cache
import plotly.express as px  # for plots
import random  # for generating run IDs
from sklearn.model_selection import train_test_split  # for splitting train & test data
import torch  # for matrix optimization

from openai.embeddings_utils import get_embedding, cosine_similarity  # for embeddings

reaction_type = {"<RX_6>": 0, "<RX_2>": 1, "<RX_1>": 2, "<RX_3>": 3, "<RX_7>": 4, "<RX_9>": 5, "<RX_5>": 6, "<RX_10>": 7, "<RX_4>": 8, "<RX_8>": 9}

# input parameters
embedding_cache_path = "data/uspto50_embedding_cache.pkl"  # embeddings will be saved/loaded here
default_embedding_engine = "babbage-similarity"  # text-embedding-ada-002 is recommended
num_pairs_to_embed = 1000  # 1000 is arbitrary
local_dataset_path = "data/uspto_50.csv"  # download from: https://nlp.stanford.edu/projects/snli/


def process_input_data(df: pd.DataFrame) -> pd.DataFrame:
    # you can customize this to preprocess your own dataset
    # output should be a dataframe with 3 columns: text_1, text_2, label (1 for similar, -1 for dissimilar)
    df_temp = {
        "text_1": [],
        "text_2": [],
        "label": [],
    }

    df_len = len(df)
    for i in range(df_len):
        j = i + 1
        while j < df_len:
            if i < j:
                df_temp["text_1"].append(df["products_mol"][i])
                df_temp["text_2"].append(df["products_mol"][j])
                if df["reaction_type"][i] == df["reaction_type"][j]:
                    df_temp["label"].append(1)
                else:
                    df_temp["label"].append(-1)
            j += 1
        print(i)
        if i % 1000 == 0:
            print(i)
    # df = df.head(num_pairs_to_embed)
    return df_temp

# load data
train_path = "data/uspto_50_train.csv"
# test_path = "data/uspto_50_test.csv"
# valid_path = "data/uspto_50_valid.csv"
df = pd.read_csv(train_path)
# df_test = pd.read_csv(test_path)
# df_valid = pd.read_csv(valid_path)


# process input data
df = process_input_data(df)  # this demonstrates training data containing only positives
# df_test = process_input_data(df_test)
# df_valid = process_input_data(df_valid)

df = pd.DataFrame(df)
# df_test = pd.DataFrame(df_test)
# df_valid = pd.DataFrame(df_valid)


# view data
# df.head()
df.to_csv("E://uspto_50_train_" + "pair.csv", index=False)
# df_test.to_csv(test_path[:-4] + "pair.csv", index=False)
# df_valid.to_csv(valid_path[:-4] + "pair.csv", index=False)