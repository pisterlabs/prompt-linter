from multiprocessing import Pool
import os
import trafilatura
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAIChat
from langchain.docstore.document import Document


class WebQuery:
    def __init__(self, openai_api_key=None) -> None:
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        self.llm = OpenAIChat(
            temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")
        self.chain = None
        self.db = None

    def ask(self, question: str) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            docs = self.db.get_relevant_documents(question)
            response = self.chain.run(input_documents=docs, question=question)
        return response

    def ingest(self, url: str) -> str:
        with Pool(1) as p:
            result = p.apply(extract_content, (url,))
        documents = [Document(page_content=result, metadata={"source": url})]
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(
            splitted_documents, self.embeddings).as_retriever()
        self.chain = load_qa_chain(
            OpenAIChat(temperature=0, model_name="gpt-3.5-turbo-16k"), chain_type="stuff")
        return "Success"

    def forget(self) -> None:
        self.db = None
        self.chain = None


def extract_content(url):
    return trafilatura.extract(trafilatura.fetch_url(url))
