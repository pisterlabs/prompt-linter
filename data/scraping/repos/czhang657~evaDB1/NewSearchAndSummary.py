from ast import keyword
import os
import shutil
from typing import Dict, List
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
import pandas as pd
import streamlit as st, tiktoken
from unidecode import unidecode
from time import perf_counter

import evadb

MAX_CHUNK_SIZE = 10000

def receive_user_input():
    print(
        '''Welcome! This app lets you to search for the most relevant papers to the paper you provided and summarize the main tool used in the papers. 
        To use this app, you need to save your pdf of the paper into the same directory as this file and provide your OpenAI API key.'''
    )
    SerperAPI = str(input(
            "Please enter your Serper API key: "))
    timestamps = {}
    t_i = 0

    timestamps[t_i] = perf_counter()
    pdfFile = str(input(
            "Please enter your pdf path name: "))
    url = 'https://api.example.com/data'
    OpenAPI = str(
        input(
            "Please enter your OpenAI API key: "
        )
    )
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    keyWord = str(
        input(
            "Please enter your key word: "
        )
    )
    keyWord2 = str(
        input(
            "Please enter your need, search news or paper (enter s or p): "
        )
    )
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()


    os.environ['pdf'] = pdfFile
    os.environ["SERPER_KEY"] = SerperAPI
    os.environ["OPEN_KEY"] = OpenAPI
    return keyWord, keyWord2

def searchForNews(keyword, cursor):
    search = GoogleSerperAPIWrapper(type="news", tbs="qdr:w1", serper_api_key=os.environ["SERPER_KEY"])
    result_dict = search.results(keyword)
    cursor.drop_table("News111", if_exists=True).execute()
    cursor.drop_table("News", if_exists=True).execute()
    print(cursor.query(
        """CREATE TABLE IF NOT EXISTS News111 (title1 TEXT(50), link TEXT(50), summary TEXT(50));"""
    ).execute())
    cursor.query("INSERT INTO News111 (title1, link, summary) VALUES ('1', '2', '3')").df();
    cursor.query("SELECT * FROM News111").df()
    for news_item in result_dict['news']:
        title = news_item['title']  # Limit the title to 50 characters
        link = news_item['link']    # Limit the link to 50 characters
        loader = UnstructuredURLLoader(urls=[news_item['link']])
        data = loader.load()
        # print(title)
        # print(link)
        llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo', openai_api_key=os.environ["OPEN_KEY"])
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(data)
        # print(summary)
        # print(cursor.query("SELECT * FROM News").df())
        cursor.query(f"INSERT INTO News111 (title1, link, summary) VALUES ('{title}', '{link}', '{summary}')").df();
        cursor.query("SELECT * FROM News111").df()

    cursor.query("DROP FUNCTION IF EXISTS Similarity;").execute()
    Similarity_function_query = """CREATE FUNCTION Similarity
                    INPUT (Frame_Array_Open NDARRAY UINT8(3, ANYDIM, ANYDIM),
                           Frame_Array_Base NDARRAY UINT8(3, ANYDIM, ANYDIM),
                           Feature_Extractor_Name TEXT(100))
                    OUTPUT (distance FLOAT(32, 7))
                    TYPE NdarrayFunction
                    IMPL './similarity.py'"""
    
    cursor.query(Similarity_function_query).execute()

    query = f"""
    SELECT Frame_Array_Open, Frame_Array_Base, Feature_Extractor_Name, Similarity(Frame_Array_Open, Frame_Array_Base, Feature_Extractor_Name) AS distance
    FROM News
    ORDER BY distance ASC
    LIMIT 1;
    """

    # Execute the query using the cursor
    print(cursor.query(query).execute())



def LoadPaper(keyword, cursor):
    timestamps = {}
    t_i = 0

    timestamps[t_i] = perf_counter()
    pdf = os.environ['pdf']
    cursor.query("DROP Table IF EXISTS MyPDF;").execute()
    LoadQuery = f'''LOAD PDF '{pdf}' INTO MyPDF;'''
    cursor.query(LoadQuery).execute()
    context_list = []
    context = ""
    res_batch = cursor.query(
        f"""SELECT data FROM MyPDF
        ORDER BY Similarity(SentenceFeatureExtractor('{keyword}'),features)
        LIMIT 5;"""
    ).execute()
    for i in range(len(res_batch)):
        context_list.append(res_batch.frames["MyPDF.data"][i])
    context = "\n".join(context_list)
    question = """You are given a block of disorganized text extracted from an academic paper. The goal is to get a table of 5 relevant academic papers: their titles, links and summaries."""
    t_i = t_i + 1
    timestamps[t_i] = perf_counter()
    cursor.query(f"""
    SELECT ChatGPT(
    '{question}'
    )

    FROM '{context}';
    """).df()



if __name__ == "__main__":
    user_input, need = receive_user_input()
    if (need == 's'):
        cursor = evadb.connect().cursor()
        searchForNews(user_input, cursor)
    elif (need == 'p'):
        cursor = evadb.connect().cursor()
        LoadPaper(user_input, cursor)
    else:
        print("Wrong input, please try again.")
