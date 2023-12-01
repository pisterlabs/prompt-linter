from flask import Flask, request, jsonify
from flask import Flask, render_template, request, url_for

from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import time
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas
import openai
import numpy as np
import glob
import datetime

from api_key import api_key
os.environ["OPENAI_API_KEY"] = api_key

openai.api_key = api_key

ips = []
ips_times = []

ips_ref = []
ips_times_ref = []

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



def logic(question):

    df = pandas.read_csv(f"embs0.csv")

    embs = []
    for r1 in range(len(df.embedding)): # Changing the format of the embeddings into a list due to a parsing error
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs

    bot_message = ""
    product_embedding = get_embedding( # Creating an embedding for the question that's been asked
        question
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding)) # Finds the relevance of each piece of data in context of the question
    df.to_csv("embs0.csv")

    df2 = df.sort_values("similarity", ascending=False) # Sorts the text chunks based on how relevant they are to finding the answer to the question
    df2.to_csv("embs0.csv")
    df2 = pandas.read_csv("embs0.csv")
    #print(df2["similarity"][0])

    from langchain.docstore.document import Document

    comb = [df2["combined"][0]]
    docs = [Document(page_content=t) for t in comb] # Gets the most relevant text chunk

    prompt_template = question + """

    {text}

    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT) # Preparing the LLM

    output = chain.run(docs) # Formulating an answer (this is where the magic happens)

    return output


app = Flask(__name__, template_folder="templates", static_folder="static")
@app.route("/")
def home():
    return render_template("webapp.html") # Renders the webpage

@app.route('/chat', methods=['POST']) # Listens to incoming requests

def chat(): # The function that recieves questions and sends answers from the chatbot

    user_message = request.json['message'] # Receives the question
    pm = request.json['pm'] # Recieves the previous question that has been asked by the user (needed for follow up questions)

    response = logic(user_message) # Creates a response from the ChatBot
    if ("sorry" in response.lower()) or ("provide more" in response.lower()) or ("not found" in response.lower()) or ("does not mention" in response.lower()) or ("does not reference" in response.lower()) or ("no information" in response.lower()) or ("not enough information" in response.lower()) or ("unable to provide" in response.lower()) or ("the guidelines do not" in response.lower()):
        response = logic(str(pm + ' ' + user_message)) # If the ChatBot isn't able to answer the question, it uses the previous question to make an answer in case it's a follow up question

    response = response.replace("<", "").replace(">", "") # Cleans the response
    return jsonify({'message': response}) # Finally returns the response


if __name__ == "__main__":
    app.run(host="localhost", port=8001, debug=True) # Runs the ChatBot on port:8001 (you can use a different one)
