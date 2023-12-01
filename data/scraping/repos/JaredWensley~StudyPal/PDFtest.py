# use defined utilities
from utils import *

# ingest PDF files
from langchain.document_loaders import PyPDFLoader

# Load GOOG's 10K annual report (92 pages).
def_url = "https://abc.xyz/investor/static/pdf/20230203_alphabet_10K.pdf"

class newPDFWorker:
    def __init__(self) -> None:
        pass

    def setFolder(self, folderName):
        self.folderName = folderName

    def embedPDF(self, url):

        loader = PyPDFLoader(url)
        documents = loader.load()

        # from google.colab import auth as google_auth
        # google_auth.authenticate_user()

        PROJECT_ID = ""
        LOCATION = "us-central1"

        import vertexai
        x = vertexai.init(project=PROJECT_ID, location=LOCATION)

        # split the documents into chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        print(f"# of documents = {len(docs)}")

        vector_save_directory = 'D:\\Documents\\NotesHelper\\'+ self.folderName #CHANGE THIS

        # Store docs in local vectorstore as index
        # it may take a while since API is rate limited
        from langchain.vectorstores import Chroma

        # create DB file from results
        chroma_db = Chroma.from_documents(docs,
                                        embeddings,
                                        persist_directory=vector_save_directory)

        chroma_db.persist()

        # Read from the created chroma DB (sqlite file)
        vector_read_from_db = Chroma(persist_directory=vector_save_directory,
                                    embedding_function=embeddings)

        # Expose index to the retriever, will search based on question
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

        #query = "What was Alphabet's net income in 2022?"
        #result = qa({"query": query})
        print("Done")
