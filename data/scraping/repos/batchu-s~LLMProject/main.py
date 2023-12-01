import requests
# import boto3
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
import pandas as pd
import configparser
from fastapi.encoders import jsonable_encoder
import time
# import json
import numpy as np
from langchain.llms import OpenAI
import os
import openai

def get_bedrock_client():
    """This function returns the amazon Bedrock client

    Returns:
        _type_: Bedrock client
    """
    bedrock_llm = BedrockEmbeddings(
        credentials_profile_name="sumanth",
        model_id="amazon.titan-e1t-medium",
        region_name="us-east-1",
    )
    return bedrock_llm

#--------------------------------------------------------------------------

def get_cohere_client():
    """This function returns the cohere client

    Returns:
        _type_: cohere client
    """
    config = configparser.ConfigParser()
    config.read("config.properties")

    return CohereEmbeddings(
        model="embed-english-light-v2.0",
        cohere_api_key=config['DEFAULT']['COHERE_API_KEY']
    )

#--------------------------------------------------------------------------

def get_openai_client(model: str) -> OpenAI:
    """This function returns the openai client

    Args:
        model (str): model name

    Returns:
        OpenAI: OpenAI client
    """
    return OpenAI(model=model, temperature=0.7)

#--------------------------------------------------------------------------

def get_embedding(text: str, model="text-embedding-ada-002") -> list:
    """This function takes text and returns the embedding of the text

    Args:
        text (str): text chunk from the dataframe

    Returns:
        list: Embedding vector of the text chunk
    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']

#--------------------------------------------------------------------------

if __name__ == "__main__":

    conf = configparser.ConfigParser()
    conf.read("config.properties")
    openai.api_key = conf['DEFAULT']['OPENAI_API_KEY']
    os.environ['OPENAI_API_KEY'] = conf['DEFAULT']['OPENAI_API_KEY']
    ####################################################################
    # load documents
    ####################################################################
    # URL of the Wikipedia page to scrape
    url = 'https://en.wikipedia.org/wiki/Prime_Minister_of_the_United_Kingdom'

    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the text on the page
    article_text = soup.get_text()
    article_text = article_text.replace("\n", "")

    with open('output.txt', 'w', encoding="utf-8") as f:
        f.write(article_text)

    ####################################################################
    # split text
    ####################################################################
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 100,
        chunk_overlap  = 20,
        length_function = len,
    )


    all_split_texts = text_splitter.create_documents([article_text])

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(all_split_texts, embeddings)

    # all_split_texts_json = [json.dumps(jsonable_encoder(doc)) for doc in all_split_texts]
    # text_chunks = [item.page_content for item in all_split_texts]

    # df = pd.DataFrame({'text_chunks': text_chunks})
    
    # df['ada_embedding'] = df.text_chunks.apply(lambda x: get_embedding(x, model="text-embedding-ada-002"))
    # print(df.head())

    # Calculate embeddings for the user's question.
    users_question = "Who is the prime minister of the United Kingdom?"

    results = db.similarity_search(
        query=users_question,
        n_results=5
    )

    # question_embedding = get_embedding(users_question, model="text-embedding-ada-002")

    # # create a list to store calculated cosine similarity
    # cosine_similarity = []

    # for index, row in df.iterrows():
    #    A = row.ada_embedding
    #    B = question_embedding
       
    #    cosine = np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

    #    cosine_similarity.append(cosine)

    # df['cosine_similarity'] = cosine_similarity
    # df.sort_values(by=['cosine_similarity'], ascending=False)

    # print(df.head())

    # Build a prompt template.
    llm = get_openai_client(model="text-davinci-003")
    # context = ""

    # for index, row in df[0:50].iterrows():
    #     context = context + " " + row.text_chunks

    # define the prompt template
    template = """
    You are a chat bot who loves to help people! Given the following context sections, answer the
    question using only the given context. If you are unsure and the answer is not
    explicitly writting in the documentation, say "Sorry, I don't know how to help with that."

    Context sections:
    {context}

    Question:
    {users_question}

    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "users_question"])

    prompt_text = prompt.format(context=results, users_question=users_question)
    print(llm(prompt_text))