# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/data_connection/text_embedding/
https://www.langchain.com.cn/modules/models/text_embedding

The Embeddings class is a class designed for interfacing with text embedding models. 
There are lots of embedding model providers (OpenAI, Cohere, Hugging Face, etc) - this class is designed to provide a standard interface for all of them.
"""
import os
import torch as th
from langchain.embeddings import (OpenAIEmbeddings, HuggingFaceEmbeddings)


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 1-使用 OpenAI 模型
embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
len(embeddings), len(embeddings[0])

embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
embedded_query[:5]

# ----------------------------------------------------------------------------------------------------------------
# 2-使用 Hugging Face 模型
'''
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers
'''
checkpoint = "all-mpnet-base-v2"

embeddings_model = HuggingFaceEmbeddings(
    model_name=os.path.join(path_model, checkpoint),
    cache_folder=os.path.join(path_model, checkpoint),
    # model_kwargs={"device": "cuda"},
    # encode_kwargs={"normalize_embeddings": False}
    )

text = "This is a test document."
embeddings = embeddings_model.embed_documents(texts=[text])
len(embeddings), len(embeddings[0])

embedded_query = embeddings_model.embed_query(text="What was the name mentioned in the conversation?")




