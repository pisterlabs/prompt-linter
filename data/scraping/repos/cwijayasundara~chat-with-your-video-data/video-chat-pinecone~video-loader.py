import os
import openai
import sys
import pinecone
from langchain.chat_models import ChatOpenAI

from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

sys.path.append('../..')
_ = load_dotenv(find_dotenv())


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


openai.api_key = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = ''
PINECONE_ENV = 'us-west4-gcp-free'
index_name='image-store-research'

url = "https://www.youtube.com/watch?v=BunESRhYhec"

save_dir = "docs/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)
docs = loader.load()
print("There are " + str(len(docs)) + " documents in the list of docs")

# step 2: split the video into chunks
r_text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=50,
    length_function=len
)

splits = r_text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

doc_db = Pinecone.from_documents(
    splits,
    embeddings,
    index_name=index_name
)

# combine contextual compression with self-query
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

#  combining compression and self-query
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=doc_db.as_retriever(search_type="mmr")
)
question = "What are we going to learn from this course?"
compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)
