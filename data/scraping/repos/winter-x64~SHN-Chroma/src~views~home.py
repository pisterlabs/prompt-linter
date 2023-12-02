from flask import Blueprint, render_template

# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import TextLoader


# import chromadb
# from chromadb.utils import embedding_functions

# chroma_client = chromadb.Client()


home = Blueprint("home", __name__)


@home.route('/')
def homepage():
    return render_template("home/home.html")


@home.route('/upload_text')
def upload_text():

    #     data = ""
    #     # load the document and split it into chunks
    #     # loader = TextLoader(data)

    #     # documents = loader.load()

    #     # split it into chunks
    #     # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #     # docs = text_splitter.split_documents(documents)

    #     embedding_function = embedding_functions.DefaultEmbeddingFunction()

    #     embedded_data = embedding_function(data)

    #     # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #     # db = Chroma.from_documents(docs, embedding_function)

    #     collection = chroma_client.create_collection(name="collection_myHelper")
    #     collection.add(
    #         embeddings= embedded_data,
    #     )
    #     # print results
    vart = "Uploaded"

    return render_template("uploads/upload_txt.html", vart=vart)


@home.route('/upload_pdf')
def upload_pdf():

    #     data = ""
    #     # load the document and split it into chunks
    #     loader = TextLoader(data)

    #     documents = loader.load()

    #     # split it into chunks
    #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #     docs = text_splitter.split_documents(documents)

    #     embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #     db = Chroma.from_documents(docs, embedding_function)

    #     # print results
    #     vart = "Uploaded"

    #     return render_template("home/home.html", vart=vart)

    # @home.route('/search')
    # def search():
    #     query = "What did the president say about Ketanji Brown Jackson"

    #     db = ""

    #     docs = db.similarity_search(query)

    #     # print results
    #     vart = docs[0].page_content
    vart = "Searched"
    return render_template("uploads/upload_pdf.html", vart=vart)
