# -*- coding: utf-8 -*-
# @Time : 5/15/23 11:17 PM
# @Author : AndresHG
# @File : nlpcloud_local.py
# @Email: andresherranz999@gmail.com

import pickle
import time

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain.llms import OpenAI, HuggingFaceHub, LlamaCpp, NLPCloud
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)
from langchain.chains.question_answering import load_qa_chain

# Load vectorstore
embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"}
)
vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)

# Construct a ConversationalRetrievalChain with a streaming llm for combine docs
# and a separate, non-streaming llm for question generation
llm = NLPCloud(
    verbose=True,
)

question_generator = LLMChain(
    llm=llm, prompt=CONDENSE_QUESTION_PROMPT
)
doc_chain = load_qa_chain(
    llm, chain_type="stuff", prompt=QA_PROMPT
)

qa = ConversationalRetrievalChain(
    retriever=vectorstore.as_retriever(),
    combine_docs_chain=doc_chain,
    question_generator=question_generator,
)
print("NLPCloud model loaded")

chat_history = []
query = "Which class should I pick if I like spells?"
result = qa.run({"question": query, "chat_history": chat_history})
print("Result: ", result)

time.sleep(3)
chat_history.append((query, result))
query = "I like fireballs!"
result = qa.run({"question": query, "chat_history": chat_history})
print("Result: ", result)
