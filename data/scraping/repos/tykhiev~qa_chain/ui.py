import gradio as gr
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI
import os


# Load the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load the PDF document


def load_document(file):
    loader = PyPDFLoader(file.name)
    document = loader.load()
    return document

# Load the question answering system


def load_qa_system(document):
    # Split document into chunks and state embeddings
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents=document)

    # State embedding
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # # Create vector store to use as index(save)
    vectordb = Chroma.from_documents(
        docs, embeddings, persist_directory="/chroma/db")
    vectordb.persist()
    vectordb_saved = Chroma(embedding_function=embeddings,
                            persist_directory="/chroma/db")
    # Expose this index in a retriever interface
    retriever = vectordb_saved.as_retriever(
        search_type="similarity", kwargs={"k": 2})

    # Create a question answering chain
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(), retriever)

    return qa



# Chat function


def chat(qa, query):
    chat_history = []
    response = qa({"question": query, "chat_history": chat_history})
    chat_history.append(f"User: {query}\nAI: {response['answer']}\n")
    return response["answer"]


# Define the Gradio interface
file_input = gr.File(label="Upload PDF Document")
text_input = gr.Textbox(label="Ask something")


chat_history = []


def chatbot_interface(file, query):
    global chat_history

    if file is not None:
        document = load_document(file)
        qa_system = load_qa_system(document)

        response = ""
        for question in query.split('\n'):
            response = chat(qa_system, question)
            chat_history.append(f"User: {question}\nAI: {response}\n")

        return ''.join(chat_history)

    else:
        return "Please upload a PDF document."


gr.Interface(fn=chatbot_interface, inputs=[
             file_input, text_input], outputs="text").launch(debug=True)
