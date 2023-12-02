from langchain.retrievers import PineconeHybridSearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import SpladeEncoder
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain import SerpAPIWrapper
from langchain.agents import AgentType
# from fastapi.responses import StreamingResponse
from langchain.callbacks.manager import CallbackManager
from langchain.tools.human.tool import HumanInputRun
# from pydantic import BaseModel
# from langchain.utilities import GoogleSearchAPIWrapper
import pinecone
import openai
import time
# from fastapi import FastAPI
from lcserve import serving
import os


@serving(websocket=True)
def get_answers(question: str, **kwargs) -> str:
    streaming_handler = kwargs.get('streaming_handler')

    # initialize the pinecone index
    pinecone.init(
        api_key="f112db94-1b02-44ec-b1d7-a4cf165fad28",  # app.pinecone.io
        environment="us-east1-gcp"  # next to api key in console
    )
    index = pinecone.Index("criminal-laws")

    # Getting our encoding models ---> dense + sparse
    embeddings = HuggingFaceEmbeddings(model_name="msmarco-distilbert-base-tas-b")
    splade_encoder = SpladeEncoder()

    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=splade_encoder, index=index)

    # setting up open ai env
    os.environ["OPENAI_API_KEY"] = "your key"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = "your org key"
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", verbose=True, streaming=True,
                     callback_manager=CallbackManager([streaming_handler]))

    # Creating a langchain document retriver chain
    dnd_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever,
                                         callback_manager=CallbackManager([streaming_handler]))

    os.environ["SERPAPI_API_KEY"] = "your serp key"
    search = SerpAPIWrapper()

    # Adding tools to help the agent
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions"
        ),
        Tool(
            name="criminal-laws",
            func=dnd_qa.run,
            description="useful for when you need to know something about Criminal law. Input: an objective to know more about a law and it's applications. Output: Correct interpretation of the law. Please be very clear what the objective is!",
            return_direct=True
        )
    ]

    HumanInputRun.description = "useful for when you want more context to answer the question perfectly, or when user is aasking about something that's to specific and not general or when some vague question is being asked to you"
    tools.append(HumanInputRun())

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    if agent.run(question):
        return ("")


