import os
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory 
from langchain.vectorstores import MongoDBAtlasVectorSearch
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from fastapi import FastAPI
import datetime
from pydantic import BaseModel

class QueryRequest(BaseModel):
    start_date: str
    end_date: str
    country: str
    state: str
    query: str
    chat_history: list

# uvicorn Tumen_Chatbot_development_edition:app --reload
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# The local machine should have the following environment variables:
ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
ATLAS_USER = os.environ["ATLAS_USER"]

# This function is used to parse the filters into the format that can be used by MongoDB
def parse_parameters(start_date, end_date, country, state):
    must_conditions = []
    if state != 'null':
        filter = {
            "text": {
                "path": "state",
                "query": state
            }
        }
        must_conditions.append(filter)

    if country != 'null':
        filter = {
            "text": {
                "path": "country",
                "query": country
            }
        }
        must_conditions.append(filter)

    start_date = '1999-01-01' if start_date == 'null' else start_date
    end_date = '2999-01-01' if end_date == 'null' else end_date
    filter = {
        'range': {
            'path': 'messageDatetime',
            'gte': datetime.datetime.strptime(start_date, "%Y-%m-%d"),
            'lte': datetime.datetime.strptime(end_date, "%Y-%m-%d")+datetime.timedelta(days=1),
        }
    }
    must_conditions.append(filter)

    conditions = {
        "compound": {
            "must": must_conditions
        }
    }

    return conditions

# This function calls the chatbot and returns the answer and prints all the relevant metadata
@app.post("/query")
def query(query_request: QueryRequest):
    start_date = query_request.start_date
    end_date = query_request.end_date
    country = query_request.country
    state = query_request.state
    query = query_request.query
    chat_history = query_request.chat_history
    '''

    Args:
        start_date: string, e.g. '2022-01-01'
        end_date: string e.g. '2022-01-02'
        country: string e.g. 'Switzerland'
        state: string e.g. 'Zurich'
        query: string e.g. 'Can I get free clothes in Zurich?'
        chat_history: array

    Returns:

    '''
    
    # initialize the connection to MongoDB Atlas
    client = MongoClient(
        "mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(
            ATLAS_USER, ATLAS_TOKEN))
    db_name, collection_name = "scrape", "telegram"
    collection = client[db_name][collection_name]

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print('OpenAI API key not found in environment variables.')
        exit()

    # create the embedding and vector search objects
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectors = MongoDBAtlasVectorSearch(
        collection=collection, text_key='messageText',
        embedding=embeddings, index_name='telegram_embedding'
    )

    # create the memory object
    memory = ConversationBufferMemory( 
    memory_key='chat_history', 
    return_messages=True, 
    output_key='answer')

    # create the large leanguage model object
    llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo-16k', openai_api_key=api_key)

    # create the prompt template for chatbot to use
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    Combine the information from the context with your own general knowledge to provide a comprehensive and accurate answer. 
    Please be as specific as possible, also you are a friendly chatbot who is always polite.
    {context}
    Question: {question}"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    # generate conditions
    must_conditions = parse_parameters(start_date, end_date, country, state)
    print(must_conditions)
    # create a chatbot chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectors.as_retriever(search_type = 'mmr',
                                       search_kwargs={
                                                'k': 100, 'lambda_mult': 0.25,
                                                "pre_filter": {
                                                   "compound": {
                                                       "must": must_conditions
                                                   }
                                                },
                                       }),
        memory = memory,
        return_source_documents=True,
        return_generated_question=True,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # create the chat
    answer = chain({"question": query, "chat_history": chat_history})
    # for i in range(10):
    #     print(answer["source_documents"][i].metadata['state'])
    #     print(answer["source_documents"][i].metadata['country'])
    #     print(answer["source_documents"][i].metadata['messageDatetime'])
    #print(answer["source_documents"][0].page_content)
    return answer["answer"], answer['chat_history']