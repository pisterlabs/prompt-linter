import os
from doc_vector_store import load_pinecone
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

from azure_file_loader import azureLoader
from embedding import get_embeddings



OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]


# define a questions and answers bot 
def qaBot(query, index_name):

    
 
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    existDocsearch = load_pinecone(index_name)
    docs = existDocsearch.similarity_search(query, include_metadata=True)

    return chain.run(input_documents=docs, question=query)


if __name__ == "__main__":
    # test the bot 
    query = "What is the benefits of this stablecoins"
    document_path = "curve-stablecoin.pdf"
    print(qaBot(query, document_path))




