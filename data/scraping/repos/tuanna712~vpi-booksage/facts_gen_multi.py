import os
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from underthesea import word_tokenize
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings

#---CLAUDE-RESPONSE----------------------------------------------
def responding_claude(question, context):
    #---PROMPT-------------------------------------------------------
    prompt = f"You are a helpfull assistant that excel in answering questions from a given context.\
            You will be provided with the question which is delimited by XML tags and the \
            context delimited by triple backticks. \
            Base on this context, please answer the question for Vietnamese citizen in formal style.\
            If the context does not contain relevant information,\
            please answer that context is not enough information\
            <tag>{question}</tag>\
            ````\n{context}\n```"
    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    HUMAN_PROMPT = f"{prompt}\n\nHuman: "
    AI_PROMPT = "\n\nAssistant:"
    completion = client.completions.create(
        model="claude-2",
        max_tokens_to_sample=2000,
        temperature=0.1,
        prompt=f"{HUMAN_PROMPT} {AI_PROMPT}",
    )
    return completion.completion
