from config import openaiapi
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory


def doc_loader(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents


def inputs(file_path):
    documents = doc_loader(file_path)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openaiapi)
    vectorstore = Chroma.from_documents(documents, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return vectorstore, memory


def chatbot(file_path, query):
    vectorstore, memory = inputs(file_path)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openaiapi)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    result = qa({"question": query})
    return result


file_path = "/Users/saumya/Documents/Government/files/Producer Enterprise/Ujjala/extracted/Draft 6th PMC Minutes - Dairy Value Chain.txt"
input_text = "is there any point included about planning for the utilisation of left-over funds in the document? Please give an exhaustive description of the context and then explain why you reached youur answer "
response = chatbot(file_path, input_text)
answer = response["answer"]
print(f"Question: {input_text}")
print("\n")
print(f"Answer: {answer}")


"""from config import openaiapi
from summarise import load_and_convert_documents
from llama_index import VectorStoreIndex, LLMPredictor, ServiceContext
import logging
import sys
from llama_index.node_parser import SimpleNodeParser
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
import openai

openai.api_key = openaiapi


def qa(file_path, input_text):
    # openai.api_key = openaiapi
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    docs = load_and_convert_documents(file_path)
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(docs)
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openaiapi)
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = VectorStoreIndex(nodes, service_context=service_context)
    query_index = index.as_query_engine()
    response = query_index.query(input_text)
    tools = [
        Tool(
            name="LlamaIndex",
            func=response,
            description="useful for when you want to answer questions about a specific topic related to the government. The input to this tool should be a complete english sentence.",
            return_direct=True,
        ),
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openaiapi)
    agent_executor = initialize_agent(
        tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
    )
    return agent_executor.run(input=input_text)


file_path = "/Users/saumya/Documents/Government/files/Producer Enterprise/Ujjala/extracted/Draft 6th PMC Minutes - Dairy Value Chain.txt"
input_text = "is there any point included about comparison of year-wise financial achievements vs their targets in the document?"
answer = qa(file_path, input_text)
print(answer)"""
