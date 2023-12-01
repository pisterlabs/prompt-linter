from langchain.agents import Tool
from langchain.agents import tool
from files_uploader import PdfLoader, vectorstore
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from API_keys import OPENAI_API_KEY 
import json
from pdf_viewer import pdfViewer
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from API_keys import WOLFRAM_ALPHA_APPID



pdf_loader = PdfLoader("files")
pdf_loader.load_files_in_pinecone(index_name='test')

#function to get metadata from a list of documents
def get_metadata(documents_list):
    data = { }
    for i in range(len(documents_list)):
        document = documents_list[i]
        data[f'chunk_{i}'] = document.metadata
    return data

    
@tool
def document_search(query: str):
    """Answers a question about a file in the database."""

    number_of_chunks = 2
    doc = vectorstore.similarity_search(query,k=number_of_chunks)
    data = get_metadata(doc)
    pdfViewer(data)
    chain = load_qa_with_sources_chain(OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY), chain_type="stuff")
    chain({"input_documents": doc, "question": query}, return_only_outputs=True)
    return chain({"input_documents": doc, "question": query}, return_only_outputs=True)


@tool
def wolfram(query: str):
    """Answers a question about a math problem."""
    wolfram = WolframAlphaAPIWrapper(wolfram_alpha_appid=WOLFRAM_ALPHA_APPID)
    return wolfram.run(query)

tools = [
    Tool(
        name = "Document Search",
        func=document_search,
        description="useful for question answering about a specific document in the database. When you don't know the answer to a query this tool will help you find the answer."
    ),
        Tool(
        name = "Wolfram Alpha",
        func=wolfram.run,
        description="useful for question answering about a math problem. Always use this tool when there is a math operation to do."
        )
]

