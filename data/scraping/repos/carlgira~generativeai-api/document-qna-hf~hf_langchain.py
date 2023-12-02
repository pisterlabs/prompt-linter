from langchain import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import UnstructuredWordDocumentLoader, OnlinePDFLoader
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer
from abc import ABC, abstractmethod
import os

CHUNK_SIZE = int(os.environ['DOC_CHUNK_SIZE'])
CHUNK_OVERLAP = int(os.environ['DOC_CHUNK_OVERLAP'])
MAX_NEW_TOKENS = int(os.environ['DOC_MAX_NEW_TOKENS'])
MAX_NUM_TOKENS = int(os.environ['LLM_MAX_NUM_TOKENS'])
HF_MODEL_NAME = os.environ['HUGGINGFACEHUB_MODEL']

question_prompt_template = """Use ONLY the following pieces of context to answer the question at the end.
Return any relevant text verbatim.
Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""

QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer, combine them, and delete repeated information. 
QUESTION: {question}
=========
Contex :{context}
=========
FINAL ANSWER:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["context", "question"]
)


class LangChainBacked(ABC):
    def __init__(self, model, max_tokens, max_new_tokens):
        self.embeddings = HuggingFaceHubEmbeddings()
        self.llm = HuggingFaceHub(repo_id=model, model_kwargs={"temperature": 0.01, "max_new_tokens": max_new_tokens,
                                                               "max_tokens": max_tokens})
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.question_prompt_num_of_tokens = self.count_tokens(question_prompt_template)
        self.combine_prompt_num_of_tokens = self.count_tokens(combine_prompt_template)
        self.db = None

    def read_document(self, file_name):
        loader = None
        if file_name.endswith('.txt'):
            loader = TextLoader(file_name)

        if file_name.endswith('.pdf'):
            loader = OnlinePDFLoader(file_name)

        if file_name.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_name)

        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        split_docs = text_splitter.split_documents(documents)
        for d in split_docs:
            d.metadata['count'] = self.count_tokens(d)

        return split_docs

    def read_documents(self, list_of_file_name):
        docs = []
        for file_name in list_of_file_name:
            docs.extend(self.read_document(file_name))
        return docs

    @abstractmethod
    def load_db(self, **kwargs):
        pass

    @abstractmethod
    def load_doc_to_db(self, docs, **kwargs):
        pass

    def answer_query(self, query, **kwargs):
        self.load_db(**kwargs)
        return self._combine_and_evaluate_question(query)

    def count_tokens(self, text):
        if isinstance(text, Document):
            return len(self.tokenizer.encode(text.page_content))
        return len(self.tokenizer.encode(text))

    def _evaluate_question(self, docs, query, question_num_of_tokens, question_prompt):
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=question_prompt)
        num_of_tokens = sum([d.metadata['count'] for d in docs])
        if num_of_tokens + question_num_of_tokens + MAX_NEW_TOKENS < MAX_NUM_TOKENS:
            return [Document(page_content=chain.run(input_documents=docs, question=query))]

        res = []
        docs_copy = docs.copy()
        context = []
        tokens_context = 0
        while len(docs_copy) > 0:
            d = docs_copy[0]
            d_num_tokens = d.metadata['count']
            if tokens_context + d_num_tokens > self.max_tokens - self.max_new_tokens - question_num_of_tokens:
                t = chain.run(input_documents=context, question=query)
                res.append(Document(page_content=t, metadata={'count': self.count_tokens(t)}))
                context.clear()
                tokens_context = 0

            context.append(d)
            tokens_context += d_num_tokens
            docs_copy.pop(0)

        return res

    def _combine_and_evaluate_question(self, query):
        docs = self.db.similarity_search(query, raw_response=True)

        res = self._evaluate_question(docs, query, self.question_prompt_num_of_tokens, QUESTION_PROMPT)
        print('evaluate')

        if len(res) == 1:
            return res[0].page_content.replace('\n\n\n', '')

        print('combine')
        res = self._evaluate_question(docs, query, self.combine_prompt_num_of_tokens, COMBINE_PROMPT)

        return ' '.join([d.page_content for d in res]).replace('\n\n\n', '')


class OpenSearchBackend(LangChainBacked):
    def __init__(self, opensearch_url, model=HF_MODEL_NAME, max_tokens=MAX_NUM_TOKENS, max_new_tokens=MAX_NEW_TOKENS):
        self.opensearch_url = opensearch_url
        LangChainBacked.__init__(self, model, max_tokens, max_new_tokens)

    def load_db(self, **kwargs):
        return self._load_db(**kwargs)

    def _load_db(self, opensearch_index, verify_certs=True):
        self.db = OpenSearchVectorSearch(index_name=opensearch_index, embedding_function=self.embeddings,
                                         opensearch_url=self.opensearch_url, verify_certs=verify_certs)

    def load_doc_to_db(self, docs, **kwargs):
        self._load_doc_to_db(docs, **kwargs)

    def _load_doc_to_db(self, docs, opensearch_index, verify_certs=True):
        OpenSearchVectorSearch.from_documents(docs, self.embeddings, opensearch_url=self.opensearch_url,
                                              index_name=opensearch_index, verify_certs=verify_certs)


class ChromaBacked(LangChainBacked):
    def __init__(self, model=HF_MODEL_NAME, max_tokens=MAX_NUM_TOKENS, max_new_tokens=MAX_NEW_TOKENS):
        LangChainBacked.__init__(self, model, max_tokens, max_new_tokens)

    def load_db(self, **kwargs):
        return self._load_db(**kwargs)

    def _load_db(self, file_name):
        persist_directory = 'db_' + file_name
        self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

    def load_doc_to_db(self, docs, **kwargs):
        self._load_doc_to_db(docs, **kwargs)

    def _load_doc_to_db(self, docs, file_name):
        persist_directory = 'db_' + file_name
        self.db = Chroma.from_documents(docs, self.embeddings, persist_directory=persist_directory)
        self.db.persist()


