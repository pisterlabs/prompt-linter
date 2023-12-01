import os
import pickle
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextVectorizer:
    def __init__(self, url_list):
        self.url_list = url_list
        self.loader = UnstructuredURLLoader(urls=url_list)
        self.data = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_text(self):
        self.docs = self.text_splitter.split_documents(self.data)

    def create_embeddings(self, model_kwargs={'device': 'cpu'}):
        model_folder="src/models/sentence-transformers_all-mpnet-base-v2"
        if os.path.exists(model_folder):
            model_id = model_folder
        else:
            model_id='sentence-transformers/all-mpnet-base-v2'
        self.hf_embedding = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs=model_kwargs,
            cache_folder="src/models"
        )

    def create_vector_index(self):
        self.vectorindex = FAISS.from_documents(self.docs, self.hf_embedding)
        file_path="src/data/vector_index.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(self.vectorindex, f)

class QnAProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        file_path="src/data/vector_index.pkl"
        with open(file_path, "rb") as f:
            self.vectorindex = pickle.load(f)

    def answer_question(self, query):
        os.environ['OPENAI_API_KEY'] = self.api_key
        llm = OpenAI(temperature=0.9, max_tokens=500)
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=self.vectorindex.as_retriever()
        )
        return chain({"question": query}, return_only_outputs=True)