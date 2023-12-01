from utils import *
import json

# ingest PDF files
from langchain.document_loaders import PyPDFLoader


# Load GOOG's 10K annual report (92 pages).
url = "https://abc.xyz/investor/static/pdf/20230203_alphabet_10K.pdf"

class savedPDFWorker:

    def __init__(self):
        pass
    def setFolder(self, folderName):
        self.folderName = folderName

    def get_AI_response(self, message):

        loader = PyPDFLoader(url)
        documents = loader.load()

        PROJECT_ID = ""
        LOCATION = "us-central1"
        vector_save_directory = 'D:\\Documents\\NotesHelper\\'+ self.folderName #CHANGE THIS

        # Store docs in local vectorstore as index
        # it may take a while since API is rate limited
        from langchain.vectorstores import Chroma

        vector_read_from_db = Chroma(persist_directory=vector_save_directory,
                                    embedding_function=embeddings)

        # Expose index to the retriever
        retriever = vector_read_from_db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        # Create chain to answer questions
        from langchain.chains import RetrievalQA

        # Uses LLM to synthesize results from the search index.
        # We use Vertex PaLM Text API for LLM
        qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
        )

        query = message
        result = qa({"query": query})
        return result
