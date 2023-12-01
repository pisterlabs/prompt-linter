#%%
import os
from k import *
import gradio as gr
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.storage.file_system import LocalFileStore
from langchain.document_loaders.pdf import PyPDFLoader, PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = k

# where our embeddings will be stored
#store = LocalFileStore("./cache/")

def rag(file, prompt) -> str:
    loader = PyPDFLoader(file.name)

    # by default the PDF loader both loads and splits the documents for us
    pages = loader.load_and_split()


    #region llm
    # instantiate embedding model
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embeddings)
    query = "What did the president say about Ketanji Brown Jackson"

    medium_qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        retriever=db.as_retriever(),
        return_source_documents=True,
        verbose=True
    )
    #endregion

    response_rag = medium_qa_chain({"query":prompt})
    return response_rag
# %%

demo = gr.Interface(
    rag,
    inputs=["file","text"],
    outputs=["text"],
    title="Tax Return Analytics",
    description="Tax Return Deloitte",
    article="Jesus C",
    css=".gradio-container {background-color: neutral}",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(show_api=True, debug=True)