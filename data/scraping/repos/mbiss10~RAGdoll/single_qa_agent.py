"""
Class containing the QA agent.
Run this file directly to interactively ask the agent questions, or import
the agent class into another file to use it there.
"""

from langchain.chat_models import ChatOpenAI
from langchain.retrievers import TFIDFRetriever
from langchain.schema import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import pickle
import os
import prompts as prompts
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = API_KEY


class SingleQaAgent():
    def __init__(self,
                 llm,
                 vectorstore_path,
                 prompt=None,
                 vectorstore_k=6,
                 vectorstore_sim_score_threshold=None,
                 passages_path=None,
                 tfidf_k=6,
                 ):
        # FAISS-based vectorstore 
        self.vectorstore = self.init_db(vectorstore_path)
        self.vectorstore_sim_score_threshold = vectorstore_sim_score_threshold
        self.vectorstore_k = vectorstore_k

        # tf-idf retriever for hybrid document retrieval
        self.tfidf_retriever = None
        if passages_path is not None:
            self.tfidf_retriever = self.init_tfidf_retriever(
                passages_path, tfidf_k)

        if prompt:
            self.chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=prompt)
        else:
            self.chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        
        # store history of questions asked so that we can inspect later for evaluation
        self.history = []

    def init_db(self, vectorstore_path):
        with open(vectorstore_path, "rb") as f:
            return pickle.load(f)

    def init_tfidf_retriever(self, passages_path, tfidf_k):
        data = None
        with open(passages_path, "rb") as f:
            data = pickle.load(f)

        texts, sources = list(zip(*data))
        res = TFIDFRetriever.from_texts(texts, k=tfidf_k)
        res.docs = [Document(page_content=t, metadata={"source": s})
                    for t, s in zip(texts, sources)]
        return res

    def get_docs_vectorstore(self, query):
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(query, k=self.vectorstore_k)
        res = []
        for (doc, score) in docs_and_scores:
            if score < self.vectorstore_sim_score_threshold:
                # log whenever we avoid using a document due to its similarity score being too low.
                # this is relatively rare, but it's interesting to know when the treshold is useful.
                print("Vectorstore-retrieved doc below similarity score threshold.")
                print(f"Similarity score: {score}\nThreshold score: {self.vectorstore_sim_score_threshold}\nDoc:\n{doc}")
            else:
                res.append(doc)
        return res


    # return_only_outputs determines whether or not to print all of the documents
    # that were retrieved and provided to the model
    def ask(self, query, return_only_outputs=True):
        docs = self.get_docs_vectorstore(query)

        if self.tfidf_retriever is not None:
            tf_idf_docs = self.tfidf_retriever.get_relevant_documents(query)
            docs.extend(tf_idf_docs)

        res = self.chain({"input_documents": docs, "question": query},
                         return_only_outputs=return_only_outputs)

        self.history.append((query, res))

        print(f"[RAGDoll]: {res['output_text']}\n")

        if not return_only_outputs:
            print("[Passages Consulted]:")
            for idx, source in enumerate(res["input_documents"]):
                print(f"--- [{idx+1}] {source.page_content}")

        return res


if __name__ == "__main__":
    # Run the QA agent in interactive mode

    # ChatOpenAI uses gpt-3.5-turbo
    llm = ChatOpenAI(temperature=0)
    agent = SingleQaAgent(llm,
                          "./data/dev/db_cs_with_sources.pkl",
                          prompt=prompts.TURBO_PROMPT,
                          vectorstore_k=8,
                          vectorstore_sim_score_threshold=0.7,
                          passages_path="./data/dev/cs_data.pkl",
                          tfidf_k=10)

    while True:
        query = input("> ")
        agent.ask(query, return_only_outputs=False)
