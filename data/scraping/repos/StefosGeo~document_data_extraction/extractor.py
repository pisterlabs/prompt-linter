import json
import logging
from collections import defaultdict
from langchain import FAISS
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from .documents import Document
from .questions import get_questions


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentSearch:
    def __init__(self, document: Document, embeddings):
        self._document = document
        self._embeddings = embeddings
        self._docsearch = self._initialize_search()

    def _initialize_search(self):
        text = self._document.read().clean().text
        return FAISS.from_texts(self._split_text(text), self._embeddings)

    @staticmethod
    def _split_text(text):
        text_splitter = CharacterTextSplitter(separator=".", chunk_size=6000, chunk_overlap=400, length_function=len)
        return text_splitter.split_text(text)

    def similarity_search(self, query, k=2):
        return self._docsearch.similarity_search(query, k=k)


class OpenAIQA:
    def __init__(self):
        self._chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0), chain_type="stuff")

    def ask(self, docs, prompt):
        return self._chain.run(input_documents=docs, question=prompt)


class DataExtractor:
    def __init__(self, document: Document):
        self._document = document
        self._embeddings = OpenAIEmbeddings()
        self._doc_search = DocumentSearch(document, self._embeddings)
        self._qa = OpenAIQA()
        self._query_instruction = " Give the answer in json format. example {out: 45}"
        self.total_tokens = 0
        self.total_cost = 0

    def qa(self):
        results = defaultdict()
        questions = get_questions(self._document.language)
        for question in questions:
            with get_openai_callback() as cb:
                docs = self._doc_search.similarity_search(question.value, k=2)
                prompt = question.value + self._query_instruction
                response = self._qa.ask(docs, prompt)
                results[question.name] = json.loads(response)["out"]
                self._print_logs(cb, question, results)
                self.total_tokens += cb.total_tokens
                self.total_cost += cb.total_cost
        return results

    @staticmethod
    def _print_logs(cb, question, results):
        logger.info(f"[Extractor] Query: {question.name} | Response: {results[question.name]}")
        logger.info(f"[Extractor]   Total Tokens: {cb.total_tokens}")
        logger.info(f"[Extractor]   Prompt Tokens: {cb.prompt_tokens}")
        logger.info(f"[Extractor]   Completion Tokens: {cb.completion_tokens}")
        logger.info(f"[Extractor]   Total Cost (USD): ${cb.total_cost}")