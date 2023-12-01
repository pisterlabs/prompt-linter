import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import OnlinePDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import functions_framework


@functions_framework.http
def qa_chrome_ext(request):
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
      # Allows GET requests from any origin with the Content-Type
      # header and caches preflight response for an 3600s
      headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "3600"
      }
      return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {
      "Access-Control-Allow-Origin": "*"
    }

    # Get the path to the PDF file
    # data = request.get_json()
    url = request.args.get("url")
    if url[-4:] != ".pdf":
        return ("Please select a PDF tab.", 400, headers)
    loader = OnlinePDFLoader(url)
    text = loader.load()

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("API_KEY"))
    knowledge_base = FAISS.from_documents(chunks, embeddings)

    # Create the conversational chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.environ.get("API_KEY"))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, knowledge_base.as_retriever(), memory=memory)

    # Send user input
    user_input = request.args.get("message")
    response = qa({"question": user_input})

    return (response["answer"], 200, headers)
