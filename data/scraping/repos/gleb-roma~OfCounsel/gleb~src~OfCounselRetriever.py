from langchain.schema.retriever import BaseRetriever
from langchain.schema.document import Document
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

import requests


class BaseOfCounselRetriever(BaseRetriever):
    api_params: Dict[str, Any] = {}

    RAG_SERVER_URL = "https://ladybird-winning-shiner.ngrok-free.app"
    AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiaWF0IjoxNTE2MjM5MDIyfQ.TzOHUu0Xi4oMx6F1SBGYwqqH_a2i9x7NJcD0mA-ucR0"

    def _call_search_endpoint(self, url, query, limit=5, ccpa_only=True):
        headers = {"Authorization": f"Bearer {self.AUTH_TOKEN}"}
        response = requests.get(
            url,
            params={
                "query": query,
                "limit": limit,
                "ccpa_only": ccpa_only,
                "params": self.api_params,
            },
            headers=headers,
        )
        if response.status_code == 200:
            results = response.json()
            return results
        else:
            raise Exception(
                f"Error {response.status_code} calling {url}: {response.text}"
            )

    def _results_to_docs(self, results):
        result_docs = []
        for r in results:
            if r["node"]["text"]:
                result_docs.append(
                    Document(
                            page_content=r["node"]["text"] or "",
                            metadata={**r["node"], "source": r["node"]["id"]},
                    )
                )
        return result_docs


class SimilarityOfCounselRetriever(BaseOfCounselRetriever):

    def _sim_search(self, query, limit=10, ccpa_only=True):
        '''
        query: a string
        limit: number of results to return
        ccpa_only: only return results that are part of the CCPA - if False, will return all results from statutes in the database (currently all of CIV)
        '''
        url = f"{self.RAG_SERVER_URL}/rag/sim_search/"
        return self._call_search_endpoint(url, query, limit, ccpa_only)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        _get_relevant_documents is function of BaseRetriever implemented here

        :param query: String value of the query

        """
        results = self._sim_search(query)
        return self._results_to_docs(results)


class PlainTextOfCounselRetriever(BaseOfCounselRetriever):

    def _text_search(self, query, limit=5, ccpa_only=True):
        url = f"{self.RAG_SERVER_URL}/rag/text_search/"
        return self._call_search_endpoint(url, query, limit, ccpa_only)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        _get_relevant_documents is function of BaseRetriever implemented here

        :param query: String value of the query

        """
        results = self._text_search(query)
        return self._results_to_docs(results)


class OfCounselRetriever(BaseOfCounselRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_params = kwargs.get("api_params", {})

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        _get_relevant_documents is function of BaseRetriever implemented here

        :param query: String value of the query

        """
        url = f"{self.RAG_SERVER_URL}/rag/"
        results = self._call_search_endpoint(url, query)
        if self.api_params.get('mode') == 'path_similarity':
            if self.api_params.get('flatten_path'):
                # returns all nodes from path (root to leaf) as individual documents
                result_docs = []
                for r in results:
                    for n in r["path_nodes"]:
                        result_docs.append(
                            Document(
                                    page_content=n["text"],
                                    metadata={**n, "source": n["id"]},
                            )
                        )
                return result_docs
            else:
                # combines text from all nodes in path into one document
                result_docs = []
                for r in results:
                    result_docs.append(
                        Document(
                                page_content=r["node"]["path_text"],
                                metadata={**r["node"], "source": r["node"]["id"]},
                        )
                    )
                return result_docs
        else:
            return self._results_to_docs(results)


# sim_retriever = SimilarityOfCounselRetriever()

############################################################################################################
### Test  ##################################################################################################
#
# from langchain.chat_models import ChatOpenAI
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.prompts import PromptTemplate
# from langchain.chains import create_qa_with_sources_chain
# from langchain.chains import RetrievalQA
#
#
# chat = ChatOpenAI(model_name='gpt-4', temperature=0.2)
#
# qa_chain = create_qa_with_sources_chain(chat)
#
# doc_prompt = PromptTemplate(
#     template="Content: {page_content}\nSource: {source}",
#     input_variables=["page_content", "source"],
# )
#
# final_qa_chain = StuffDocumentsChain(
#     llm_chain=qa_chain,
#     document_variable_name="context",
#     document_prompt=doc_prompt,
# )
#
# retrieval_qa = RetrievalQA(
#     retriever=SimilarityOfCounselRetriever(),
#     combine_documents_chain=final_qa_chain
# )
#
# retrieval_qa_ofcounsel = RetrievalQA(
#     retriever=OfCounselRetriever(mode="path_similarity", flatten_path=True),
#     combine_documents_chain=final_qa_chain
# )
#
# # simple test
#
# query = "What are the General Duties of Businesses that Collect Personal Information?"
#
# answer = retrieval_qa_ofcounsel.run(query) # took 24s
#
# print(answer)

