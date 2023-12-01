import os
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY")

PINECONE_API_KEY = config("Pinecone_API")
PINECONE_ENVIRONMENT = config("Pinecone_env")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import pinecone
from django.conf import settings
from langchain.chains.mapreduce import MapReduceChain
import logging
from . models import Vectorstore,KnowledgeDocument,Chunk
import uuid
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader,UnstructuredURLLoader,UnstructuredPDFLoader, PyMuPDFLoader, PDFPlumberLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from .parsers import DeepgramParser
llm = OpenAI(temperature=0)

pinecone.init(
    api_key=PINECONE_API_KEY ,  # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT  # next to api key in console
)

logger = logging.getLogger(__name__)

os.environ["PATH"] += os.pathsep + 'C:/Users/Sahit/ffmpeg-2023-06-26-git-285c7f6f6b-essentials_build/bin'

CACHE_KEY_PREFIX = "pinecone_index:"

def get_full_path(file_path):
    """
    Returns the full path to the file located at file_path.
    """
    return os.path.join(settings.MEDIA_ROOT, file_path)



def get_loader(document_type, url=None, file=None):
    loaders = {
        'YouTube': lambda url: GenericLoader(YoutubeAudioLoader([url], settings.MEDIA_ROOT), DeepgramParser()),
        'PDF': PDFPlumberLoader,
        'Web': UnstructuredURLLoader,
        # Add other document types and their loaders here...
    }

    loader_func = loaders.get(document_type)
    if not loader_func:
        raise ValueError(f"No loader found for document type {document_type}")

    if document_type == "YouTube":
        return loader_func(url)
    else:
        return loader_func(file)

def load_and_split_documents(loader):
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    
    return documents, texts

def run_summary_chain(documents):
    summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = summary_chain.run(documents)
    print(f"Documents input: {documents}")  # DEBUGGING
    print(f"Generated summary by run_summary_chain: {summary}")  # DEBUGGING
    return summary

class IndexingContext:
    def __init__(self, user, namespace, knowledgebase, document, texts):
        self.user = user
        self.namespace = namespace
        self.knowledgebase = knowledgebase
        self.document = document
        self.texts = texts
        self.index_name = 'test'  


def index_document(ctx: IndexingContext):
    # Generate embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    # Generate UUIDs for each text
    ids = [str(uuid.uuid4()) for _ in range(len(ctx.texts))]

    # List of documents
    texts = ctx.texts

    print(texts)

    # Add texts to Pinecone
    print("texts", texts)
    pinecone_index = Pinecone.from_documents(documents=texts, embedding=embeddings, ids=ids, index_name=ctx.index_name, namespace = ctx.knowledgebase.namespace)

    # Create Vectorstore entry in the database
    vectorstore = Vectorstore.objects.create(user=ctx.user, knowledgebase=ctx.knowledgebase, document=ctx.document, index=ctx.index_name)

    # Create Chunk entries in the database
    for i, doc in enumerate(ctx.texts):
        chunk_uuid = ids[i]
        content = doc.page_content  # Use page_content or another appropriate attribute
        metadata = doc.metadata
        Chunk.objects.create(vectorstore=vectorstore, uuid=chunk_uuid, content = content, metadata = metadata)

