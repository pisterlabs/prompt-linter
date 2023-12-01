import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np
import json
import pandas as pd
import sys

from oepnai_manager import openai_manager


def is_empty_string(s):
    return not bool(s and s.strip())


openai.api_key = os.getenv('OPENAI_API_KEY')


# check empty string
if is_empty_string(os.getenv('OPENAI_API_KEY')):
    print("OPENAI_API_KEY is empty")
    exit()


arguments = sys.argv

if len(arguments) < 1:
    print("Please input command")
    exit()
# find query in arguments "init" or "update"
raw_data_path = "./datas/raw_datas.csv"
raw_data_embedding_path = "./embeddings/embeddings_context.csv"

text_pattern_tiles = "กระเบื้อง{tile_pattern} มีขนาด{tile_size}, ยี่ห้อ {brand}, มีสี{tile_color}, มีพื้นผิวที่{tile_surface}, ใช้{tile_type}, ชื่อของสินค้าคือ: {tile_name}"
text_pattern_content = "บริบท: {context}, meaning: {mean}"

if arguments[1] == "init":
    openai_manager.create_embedding_data(raw_data_path,
                                         raw_data_embedding_path, text_pattern_content)
    exit()
if arguments[1] == "update":
    openai_manager.update_embedding_data(raw_data_path,
                                         raw_data_embedding_path, text_pattern_content)
    exit()

if arguments[1] == "run" and len(arguments) == 3:
    query = arguments[2]

    indexes_sort, similarities = openai_manager.get_similarity_data(
        query, raw_data_embedding_path)

    print("indexes:", indexes_sort)
    df = pd.read_csv(raw_data_embedding_path)
    embeddings = df.embedding.apply(eval).values.tolist()

    display_data = df.drop(columns=["embedding"])
    # print content
    print("A:\n", display_data.iloc[indexes_sort[0]]["context"])

    # print product

    # products = display_data.iloc[indexes_sort[0:3]]
    # with open('product_message.json', 'r') as f:
    #     message = dict()
    #     message['type'] = 'carousel'

    #     contents = []
    #     # for loop with index
    #     for i in range(0, len(products)):
    #         type = products.iloc[i]["tile_name"]
    #         print(type)
    #         contents.append(type)

    #     message['contents'] = contents
