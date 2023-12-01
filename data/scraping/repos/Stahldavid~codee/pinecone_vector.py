from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
import pinecone 
from langchain.tools import BaseTool, StructuredTool, Tool, tool

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Define the input schema for the Search Pinecone Vectorstore tool
class SearchInput(BaseModel):
    query: str = Field()



class PineconeVector:
    question: str = Field()
    is_single_input = True
    name = "Pinecone Vector"
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

        pinecone.init(
            api_key="pinecone_api_key",  # find at app.pinecone.io
            environment="us-east1-gcp"  # next to api key in console
        )

        index = pinecone.Index("langchain-chat")

        # Initialize OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()

        # Initialize a Pinecone vector store
        vectorstore = Pinecone(index=index, embedding_function=embeddings.embed_query, text_key="text")

        from langchain.agents.agent_toolkits import (
            create_vectorstore_agent,
            VectorStoreToolkit,
            VectorStoreInfo,
        )
        vectorstore_info = VectorStoreInfo(
            name="Robotics and Control Systems Knowledge Base",
            description="A collection of information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller",
            vectorstore=vectorstore,
        )

        self.toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
        # Create a RetrievalQA chain using the Pinecone vector store
        self.pinecone_search_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=vectorstore.as_retriever())

        tools = [
            Tool.from_function(
            func=self.pinecone_search_chain.run,
            name = "Pinecone_Vectorstore",
            description="A collection of information related to ROS2, Webots, impedance/admittance control, T-motor AK-series actuators, and MIT mini cheetah controller",
        # coroutine= ... <- you can specify an async method if desired as well
    ),
]
    

    

