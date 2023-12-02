from typing import List

from langchain.schema import BaseRetriever, Document


class IWXRetriever(BaseRetriever):
    vector_stores = []
    search_k = 4

    def set_search_k(self,value):
        self.search_k = value

    def initialise(self, vector_stores, search_k=4):
        self.vector_stores = vector_stores
        self.search_k = search_k

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.get_documents(query)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_documents(query)

    def get_documents(self, query):
        search_results = None
        for vector_store in self.vector_stores:
            if search_results is None:
                search_results = vector_store.similarity_search(query, k=self.search_k)
            else:
                search_results = search_results + vector_store.similarity_search(query, k=self.search_k)
        print(search_results)
        return search_results
