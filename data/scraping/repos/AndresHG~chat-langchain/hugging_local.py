# -*- coding: utf-8 -*-
# @Time : 5/15/23 11:17 PM
# @Author : AndresHG
# @File : nlpcloud_local.py
# @Email: andresherranz999@gmail.com

import pickle

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Load vectorstore
with open("../vectorstores_faiss/vectorstore_light.pkl", "rb") as f:
    vectorstore = pickle.load(f)
    print("Vectirstore loaded")

# Construct a ConversationalRetrievalChain with a streaming llm for combine docs
# and a separate, non-streaming llm for question generation
llm = HuggingFaceHub(
    repo_id="declare-lab/flan-alpaca-large",
    model_kwargs={"temperature": 0, "max_length": 512},
    # callback_manager=None
)
print("HuggingFace model loaded")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    input_key="question",
)

query = "Which class should I pick if I like spells?"
result = qa.run(query)
print("Result: ", result)


