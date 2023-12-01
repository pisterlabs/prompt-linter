""" This is a file for custom tools that you can use in the LLM agent
"""
import os
import psycopg2
import openai

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool

from src.docsearch.docsearch import (
    docsearch_create_indexes_from_files,
    docsearch_query_indexes,
)
from src.langchain_summary import produce_meta_summary, summarise_articles
from src.newsearch.refinitiv_query import (
    create_rkd_base_header,
    parse_freetext_headlines,
    parse_news_stories_texts,
    retrieve_freetext_headlines,
    retrieve_news_stories,
)

# load environment variables
load_dotenv()

# set global variables
RKD_USERNAME = os.getenv("REFINITIV_USERNAME")
RKD_PASSWORD = os.getenv("REFINITIV_PASSWORD")
RKD_APP_ID = os.getenv("REFINITIV_APP_ID")

# openai configuration
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
openai.api_type = os.getenv('AZURE_OPENAI_API_TYPE')
openai.api_base = os.getenv('AZURE_OPENAI_API_BASE')

# Create an instance of Azure OpenAI; find completions is faster than chat
CHAT_LLM = AzureOpenAI(
    deployment_name="text-davinci-003",
    model_name="text-davinci-003",
    temperature=0,
    best_of=1,
)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=7_000, chunk_overlap=400)

# Create database connection
host = os.getenv('PG_HOST')
dbname = os.getenv('PG_DB_NAME')
user = os.getenv('PG_USER')
password = os.getenv('PG_PASSWORD')
sslmode = os.getenv('PG_SSLMODE')

# Construct connection string
conn_string = f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
conn = psycopg2.connect(conn_string)


@tool("Refinitiv freetext news search summary tool", return_direct=True)
def refinitiv_freetext_news_summary_tool(input: str) -> str:
    """
    Queries the Refinitiv News API for news articles related to the free text input
    which have happened in the last num_weeks_ago.
    Then summarises the news articles and returns the summary of enriched headlines.
    """
    base_header = create_rkd_base_header(RKD_USERNAME, RKD_PASSWORD, RKD_APP_ID)

    # freetext headline search; set last_n_weeks as 2; queries both headline and body
    # for english text (Refinitiv is better for English than Chinese queries).
    freetext_results = retrieve_freetext_headlines(base_header, input, 2, "both", "EN")
    freetext_news_articles = parse_freetext_headlines(freetext_results)

    # load full news stories related to those headlines
    news_stories = retrieve_news_stories(
        base_header, [article.id for article in freetext_news_articles]
    )
    news_stories_texts = parse_news_stories_texts(news_stories)

    # summarise the news stories
    article_summaries = summarise_articles(
        chat_llm=CHAT_LLM,
        text_splitter=TEXT_SPLITTER,
        article_headlines=[a.headline for a in freetext_news_articles],
        article_texts=news_stories_texts,
    )

    # produce meta summary
    meta_summary = produce_meta_summary(CHAT_LLM, TEXT_SPLITTER, article_summaries)
    return meta_summary


@tool("document question answering", return_direct=True)
def document_question_answering(input: str) -> str:
    """
    Answers questions related to HSBC knowledge documents and gives answers
    from HSBC's perspective on topics.
    """

    # initialise docsearch variables
    NUM_DIMENSIONS = 1536
    EMBEDDINGS_MODEL = OpenAIEmbeddings(model="text-embedding-ada-002")

    ENDPOINT = os.getenv("FORM_RECOGNISER_ENDPOINT")
    CREDENTIAL = AzureKeyCredential(os.getenv("FORM_RECOGNISER_KEY"))
    DOC_ANALYSIS_CLIENT = DocumentAnalysisClient(ENDPOINT, CREDENTIAL)

    # text splitter with smaller chunk size because docs are larger
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=3_000, chunk_overlap=300)

    # TODO: Right now loads from a sample directory; need to find a way to load
    # from a vector database or otherwise? would this be pre-loaded??
    # Then can replace FAISS lookup for sample documents.
    FILES_DIR = "./data/pdf_img_samples/"
    uploaded_files = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR)]

    # create faiss index and index_doc_store
    faiss_index, index_doc_store = docsearch_create_indexes_from_files(
        NUM_DIMENSIONS,
        uploaded_files,
        DOC_ANALYSIS_CLIENT,
        EMBEDDINGS_MODEL,
        TEXT_SPLITTER,
    )

    # query faiss index
    result = docsearch_query_indexes(
        input, faiss_index, index_doc_store, EMBEDDINGS_MODEL, CHAT_LLM
    )
    return result


@tool("hsbc knowledge search tool")
def hsbc_knowledge_tool_pgvector(input: str) -> str:
    """useful for when you need to answer questions about hsbc related knowledge"""
    try:
        # get embedding from input
        response = openai.Embedding.create(input=input, engine="text-embedding-ada-002")
        embeddings = response['data'][0]['embedding']
        # create cursor
        cur = conn.cursor()
        # execute query
        cur.execute(f"SELECT content FROM hsbc_homepage_content ORDER BY embedding <-> '{embeddings}' LIMIT 1;")
        # retrieve records
        records = cur.fetchall()
        # close cursor
        cur.close()
        # return answer
        return records[0][0]
    except Exception as e:
        print(e)
        return "Sorry, I don't understand your question. Please try again."



@tool("reject tool", return_direct=True)
def reject_tool(input: str) -> str:
    # LLM agent sometimes will not reject question not related to HSBC, hence adding this tools to stop the thought/action process
    """useful for when you need to answer questions not related to HSBC"""
    return """
    I'm sorry, but as a customer service chatbot for HSBC Hongkong, I am only able to assist with questions related to HSBC Hongkong products and services. 
    Is there anything else related to HSBC Hongkong that I can help you with?
    """
