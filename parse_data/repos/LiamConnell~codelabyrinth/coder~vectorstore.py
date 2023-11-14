import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langchain import LLMChain, PromptTemplate
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.tools import Tool
from langchain.vectorstores.pgvector import PGVector

from coder.db import Database
from coder.utils import format_doc

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("POSTGRES_DRIVER", "psycopg2"),
    host=os.environ.get("POSTGRES_HOST", "localhost"),
    port=int(os.environ.get("POSTGRES_PORT", "5432")),
    database=os.environ.get("POSTGRES_DATABASE", "postgres"),
    user=os.environ.get("POSTGRES_USER", "postgres"),
    password=os.environ.get("POSTGRES_PASSWORD"),
)


class VectorStore:
    def __init__(self, connection_string=None, embedding_fn=None):
        self.connection_string = connection_string or CONNECTION_STRING
        self.embedding_fn = embedding_fn or OpenAIEmbeddings()
        self.db = Database()

    def list_collections(self) -> list[str]:
        collections = self.db.fetch_query("SELECT name from langchain_pg_collection")
        collections = [c[0] for c in collections]
        return collections

    def create_collection(self, collection_name: str, collection_metadata: dict = None):
        PGVector(self.connection_string, self.embedding_fn, collection_name, collection_metadata)

    def delete_collection(self, collection_name: str):
        PGVector(self.connection_string, self.embedding_fn, collection_name).delete_collection()

    def vectorstore(self, collection_name: str) -> PGVector:
        return PGVector(self.connection_string, self.embedding_fn, collection_name)

    def add_docs(self, collection_name, docs: list[Document]):
        PGVector.from_documents(documents=docs, collection_name=collection_name,
                                embedding=self.embedding_fn, connection_string=CONNECTION_STRING)

    def similarity_search(self, collection_name, *args, **kwargs):
        return PGVector(self.connection_string, self.embedding_fn, collection_name).similarity_search(*args, **kwargs)

    def mmr_search(self, collection_name, query, k=5, lambda_param=.5):
        query_vector = self.embedding_fn.embed_query(query)
        result = self.db.fetch_query(f"""
            select e.document, e.embedding, e.embedding <=> vector('{self.embedding_fn.embed_query(query)}') as score
            from langchain_pg_embedding e 
            join langchain_pg_collection lpc on e.collection_id = lpc.uuid
            where lpc.name='{collection_name}'
            order by e.embedding <=> vector('{query_vector}')
            limit {k*2};
        """)
        doc_set = []
        doc_vector = {}
        doc_score = {}
        for (doc, vector, score) in result:
            doc_set.append(doc)
            doc_vector[doc] = np.array(json.loads(vector)).astype('float64')
            doc_score[doc] = score
        calculate_similarity = cosine_similarity

        selected = []
        while (len(selected) < k) and doc_set:
            remaining = [doc for doc in doc_set if doc not in selected]
            mmr_score = lambda x: (
                lambda_param * calculate_similarity(doc_vector[x], np.array(query_vector)) -
                (1 - lambda_param) * max([calculate_similarity(doc_vector[x], doc_vector[y]) for y in selected])
                if len(selected) > 0 else 0
            )
            selected.append(max(remaining, key=mmr_score))
            doc_set.remove(selected[-1])
        return selected

    def similarity_search_with_evaluation(self, collection_name, query, k=5):
        documents = self.similarity_search(collection_name, query, k*2)

        summarize_prompt_template = "Please summarize the following document:\n{formatted_document}"
        SUMMARIZE_PROMPT = PromptTemplate(template=summarize_prompt_template, input_variables=["formatted_document"])

        relevance_prompt_template = "How relevant is this document to the question, either showing how to solve it or showing the relevant parts of the codebase to operate on, or showing how similar features are implemented? Answer with a score between 0 and 100. Answer with the number only.\n\nDocument:\n{formatted_document}\n\nSummary:\n{summary}\n\nQuestion: {question}\n\nScore: "
        RELEVANCE_PROMPT = PromptTemplate(template=relevance_prompt_template, input_variables=["formatted_document", "summary", "question"])

        # llm = ChatOpenAI(temperature=0)
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        summarize_chain = LLMChain(llm=llm, prompt=SUMMARIZE_PROMPT)
        relevance_chain = LLMChain(llm=llm, prompt=RELEVANCE_PROMPT)

        def evaluate_document_relevancy(document: Document, question: str) -> float:
            formatted_document = format_doc(document)
            summary = summarize_chain.apply([{"formatted_document": formatted_document}])[0]['text']
            relevance_score = relevance_chain.apply([{
                "formatted_document": formatted_document, "summary": summary, "question": question
            }])[0]['text']
            print(relevance_score.strip())
            return float(relevance_score.strip())

        eval_doc_fn = functools.partial(evaluate_document_relevancy, question=query)

        with ThreadPoolExecutor(max_workers=20) as ex:
            evaluations = list(ex.map(eval_doc_fn, documents))

        doc_evals = sorted(zip(documents, evaluations), key=lambda x: x[1], reverse=True)

        return [doc for doc, e in doc_evals[:k]]

    def get_tool(self, collection_name: str, **kwargs):
        # fn = functools.partial(self.similarity_search, collection_name=collection_name, **kwargs)
        fn = lambda q: self.similarity_search(collection_name, q)
        # TODO: extra verbiage for codebases
        description = (
            "Useful for when you need to answer questions about {collection_name}. "
            "Whenever you need information about {collection_name} "
            "you should ALWAYS use this. "
            "Input should be a fully formed question."
        )
        return Tool.from_function(
            func=fn,
            name=f"Retrieve {collection_name.replace(' ', '_')} Documents",
            description=description.format(collection_name=collection_name)
        )


def cosine_similarity(vector_a, vector_b):
    print(vector_a.dtype, vector_b.dtype)
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity
