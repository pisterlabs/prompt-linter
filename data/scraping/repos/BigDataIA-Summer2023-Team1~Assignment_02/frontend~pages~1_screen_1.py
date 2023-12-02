import os
import time
import requests

import streamlit as st

from uuid import uuid4
from dotenv import load_dotenv

from sqlalchemy import text, create_engine
from google.cloud.sql.connector import Connector

import openai
import tiktoken
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv("../.env")


def connect_to_sql():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/app/storage-key.json"  # "/app/storage-key.json"

    return Connector().connect(
        instance_connection_string=os.environ['INSTANCE_CONNECTION_NAME'],
        driver=os.environ["DB_DRIVER"],
        user=os.environ["DB_USER"],
        password=os.environ['DB_PASS'],
        db=os.environ["DB_NAME"]
    )


def get_sql_client(conn):
    return create_engine("postgresql+pg8000://", creator=conn).connect()


def fetch_companies_from_db():
    conn = get_sql_client(connect_to_sql)

    q = text('SELECT DISTINCT company from earnings_transcript_meta_data')

    # Execute the query
    result = conn.execute(q)

    # Fetch all the rows returned by the query
    rows = result.fetchall()

    companies_list = []
    # Process the rows
    for row in rows:
        companies_list.append(row.company)

    conn.close()

    return companies_list


def fetch_years_from_db():
    conn = get_sql_client(connect_to_sql)

    q = text("SELECT DISTINCT DATE_PART('year', financial_year) AS year FROM earnings_transcript_meta_data")

    # Execute the query
    result = conn.execute(q)

    # Fetch all the rows returned by the query
    rows = result.fetchall()

    years_list = []
    # Process the rows
    for row in rows:
        years_list.append(int(row.year))

    conn.close()

    return years_list


@st.cache_data
def fetch_company_metadata(company, year):
    conn = get_sql_client(connect_to_sql)

    formatted_query = "SELECT company, ticker, financial_year, quarter, uri FROM earnings_transcript_meta_data where company = '{}' and DATE_PART('year', financial_year) = '{}'".format(
        company, year)
    print(formatted_query)
    q = text(formatted_query)

    # Execute the query
    result = conn.execute(q)

    # Fetch all the rows returned by the query
    rows = result.fetchall()

    metadata = []
    # Process the rows
    for row in rows:
        metadata.append({"company": row.company, "ticker": row.ticker, "quarter": row.quarter, "uri": row.uri,
                         "timestamp": row.financial_year})

    conn.close()

    return metadata


def get_earnings_call_data(url):
    response = requests.get(url)

    # Check the status code to ensure the request was successful
    if response.status_code == 200:
        # Print the response content
        return response.text
    else:
        print("Error:", response.status_code)
        return ""


@st.cache_data
def text_summarization(api_key, prompt):
    openai.api_key = api_key

    return openai.Completion.create(
        model="text-davinci-003",
        prompt="Brief the companies financial earnings transcript \n\n {}".format(prompt),
        temperature=0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )


tiktoken.encoding_for_model('gpt-3.5-turbo')


def tiktoken_len(query):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        query,
        disallowed_special=()
    )
    return len(tokens)


def text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )


def openai_embeddings(openai_key):
    return OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=openai_key
    )


@st.cache_data
def index_vectors(openai_key, pinecone_key, pinecone_environment, index_name, query, company_metadata):
    chunks = text_splitter().split_text(query)

    record_metadata = [{
        "chunk": j, "text": chunk_text, **company_metadata
    } for j, chunk_text in enumerate(chunks)]

    pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_environment
    )

    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536
        )

    if "vector_ids" in st.session_state and len(st.session_state["vector_ids"]) > 0:
        p_index = pinecone.Index(index_name)
        p_index.delete(ids=st.session_state["vector_ids"])

    index = pinecone.GRPCIndex(index_name)
    time.sleep(30)

    if len(chunks) > 0:
        ids = [str(uuid4()) for _ in range(len(chunks))]
        st.session_state["vector_ids"] = ids
        embeds = openai_embeddings(openai_key).embed_documents(chunks)

        index.upsert(vectors=zip(ids, embeds, record_metadata))
        print("Indexing Completed")


@st.cache_data
def semantic_search(openai_key, pinecone_key, pinecone_environment, pinecone_index_name, query):
    xq = openai_embeddings(openai_key).embed_query(query)

    pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_environment
    )

    index = pinecone.GRPCIndex(pinecone_index_name)
    time.sleep(30)

    xc = index.query(xq, top_k=3, include_metadata=True)

    res = []
    for match in xc["matches"]:
        res.append(
            {"text": match["metadata"]["text"], "source": match["metadata"]["source"], "score": match["score"] * 100})

    return res


class Document:
    def __init__(self, page_content, document_metadata):
        self.page_content = page_content
        self.metadata = document_metadata


@st.cache_data
def langchain_summerization(openai_key, pinecone_key, pinecone_environment, pinecone_idx, query, document_metadata):
    pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_environment
    )

    embeddings = openai_embeddings(openai_key)
    vectordb = Pinecone.from_documents([Document(page_content=query, document_metadata=document_metadata)], embeddings,
                                       index_name=pinecone_idx)

    llm = ChatOpenAI(
        openai_api_key=openai_key,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    chain = load_summarize_chain(llm, chain_type="stuff")
    search = vectordb.similarity_search(" ")
    summary = chain.run(input_documents=search, question="Write a concise summary within 200 words.")

    return summary


def generative_answering(openai_key, pinecone_key, pinecone_environment, pinecone_idx):
    pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_environment
    )

    # switch back to normal index for langchain
    index = pinecone.Index(pinecone_idx)

    vectorstore = Pinecone(
        index, openai_embeddings(openai_key).embed_query, 'text'
    )

    # completion llm
    llm = ChatOpenAI(
        openai_api_key=openai_key,
        model_name='gpt-3.5-turbo-16k',
        temperature=0.0,
        max_tokens=4000
    )

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )


# Streamlit app
st.subheader('Text Summarization')

st.session_state['vector_ids'] = []

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password")
    pinecone_api_key = st.text_input("Pinecone API key", type="password")
    pinecone_env = st.text_input("Pinecone environment")
    pinecone_index = st.text_input("Pinecone index name")

with st.form("login", clear_on_submit=False):
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        companies_list = fetch_companies_from_db()

        company_name = st.selectbox(
            'Company Name', companies_list)
        st.write('You selected:', company_name)

    with col2:
        years = fetch_years_from_db()

        financial_year = st.selectbox(
            'Financial Year', years)

        st.write('You selected:', financial_year)

    show_data_btn = st.form_submit_button("Show Meta Data")

if show_data_btn:
    data = fetch_company_metadata(company_name, financial_year)
    st.session_state['metadata'] = data

    st.write("Metadata")
    st.json(data)

    st.write("Earnings Call Data")
    source_blob_url = "https://storage.googleapis.com/earnings-call-data-new-bucket/dataset/{}".format(
        data[0]["uri"].split("/")[
            -1])  # TODO: Let user choose the file from UI and fetch that specifc call data from the GCS
    earnings_call_data = get_earnings_call_data(source_blob_url)

    st.session_state['earnings_call_data'] = earnings_call_data

    st.text_area("", value=earnings_call_data, height=300)

    if openai_api_key and pinecone_api_key and pinecone_env and pinecone_index:
        # Summary from OpenAI
        earnings_call_data_summary = text_summarization(openai_api_key, earnings_call_data)

        st.write("Earnings Call Data Summary: ")
        st.text_area("", value=earnings_call_data_summary.choices[0].text, height=300)
        st.session_state["earnings_call_data_summary"] = earnings_call_data_summary.choices[0].text

        metadata = {'company': data[0]["company"], 'ticker': data[0]["ticker"],
                    'quarter': data[0]["quarter"], 'source': data[0]["uri"]}

        index_vectors(openai_api_key, pinecone_api_key, pinecone_env, pinecone_index, earnings_call_data, metadata)

        # Summary from Langchain and Pinecone
        langchain_summary = langchain_summerization(openai_api_key, pinecone_api_key, pinecone_env, pinecone_index,
                                                    earnings_call_data, metadata)
        st.write("Earnings Call Data Summary by Langchain & Pinecone: ")
        st.write(langchain_summary)

        st.session_state["langchain_earnings_call_data_summary"] = langchain_summary

    else:
        st.warning(f"Please provide the missing fields.")

input_query = st.text_area("Your Query Here: ", "")
search_btn = st.button("Search")

if search_btn:
    # Fetch metadata from session state
    if "metadata" in st.session_state and len(st.session_state["metadata"]) > 0:
        st.write("Metadata")
        st.json(st.session_state["metadata"])

    # Display companies earnings call transcript
    if "earnings_call_data" in st.session_state and st.session_state["earnings_call_data"] != "":
        st.write("Earnings Call Data")
        st.text_area("", value=st.session_state["earnings_call_data"], height=300)

    # Summary from OpenAI
    if "earnings_call_data_summary" in st.session_state and st.session_state["earnings_call_data_summary"] != "":
        st.write("Earnings Call Data Summary by OpenAI: ")
        st.text_area("", value=st.session_state["earnings_call_data_summary"], height=300)

    # Summary from Langchain and Pinecone
    if "langchain_earnings_call_data_summary" in st.session_state and st.session_state["langchain_earnings_call_data_summary"] != "":
        st.write("Earnings Call Data Summary by Langchain & Pinecone: ")
        st.text_area("", value=st.session_state["langchain_earnings_call_data_summary"], height=300)

    # Semantic Search results
    search_results = semantic_search(openai_api_key, pinecone_api_key, pinecone_env, pinecone_index, input_query)
    st.write("Semantic Search Results using OpenAI embeddings & Pinecone")
    st.json(search_results)

    # Generative Question and Answering
    qa_with_sources = generative_answering(openai_api_key, pinecone_api_key, pinecone_env, pinecone_index)
    response = qa_with_sources(input_query)
    st.write("Generative Question and Answering using Langchain & Pinecone")
    st.json(response)
