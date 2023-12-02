from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores import VectorStore
from langchain.schema.retriever import Document
from typing import List
from langchain.pydantic_v1 import Field
from langchain.document_transformers import LongContextReorder
from langchain.document_loaders import TextLoader
import copy


class CustomRetriever(VectorStoreRetriever):
    """
    A custom retriever that extends the VectorStoreRetriever class.
    It retrieves relevant documents based on a query and filters them based on specified criteria.

    Args:

        vectorstore (VectorStore): The vectorstore to retrieve documents from.
        base_path (str): The base path of the documents.
        search_type (str): The type of search to perform. Can be "similarity" or "mmr".
        retrieve_type (str): The type of retrieval to perform. Can be "chunk", "parent" or "custom".
        max_elements (int): The maximum number of documents to retrieve.
        filter (dict): A dictionary specifying the filtering criteria with the following keys:\n
            "casetype": A list of case types to include in the filtered list. Eks.: ["FLYKN", "KOLLKN", "PRKN", "SJTKN"].\n
            "year": A list of years to include in the filtered list. Eks.: [2016, 2017, 2018, 2019, 2020, 2021, 2022].\n
            "exclude_case": A list of case numbers to exclude from the filtered list. Eks.: ["2022-00365", "2017-00001"]\n
            "har_mindretall": A boolean indicating whether to include only documents with dissenting opinions. Eks.: True\n
            "tjenesteyter_avviser": A boolean indicating whether to include only documents where the service provider does not comply with the decision. Eks.: True\n
        search_kwargs (dict): A dictionary specifying the search arguments. The dictionary can contain the following keys:
            "k": The number of documents to retrieve from the vectorstore.
    """

    vectorstore: VectorStore
    base_path: str = ""
    search_type: str = "similarity"
    retrieve_type: str = "chunk"
    max_elements: int = 4
    filter: dict = Field(default_factory=dict)
    search_kwargs: dict = Field(default_factory=dict)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents based on a query and filters them based on specified criteria.

        Args:
            query (str): The query to retrieve relevant documents for.

        Returns:
            List[Document]: A list of `Document` objects that satisfy the filtering criteria.
        """
        chunks, scores = zip(
            *self.vectorstore.similarity_search_with_relevance_scores(
                query=query, **self.search_kwargs
            )
        )
        [
            chunk.metadata.update({"score": score})
            for chunk, score in zip(chunks, scores)
        ]

        result_docs = copy.deepcopy(chunks)
        parent_docs = self.get_parent_docs(chunks)

        if self.retrieve_type == "chunk":
            pass
        elif self.retrieve_type == "parent":
            result_docs = parent_docs
        elif self.retrieve_type == "custom":
            result_docs = self.custom_retrieve(parent_docs)

        result_docs = self.add_metadata(result_docs, parent_docs)
        result_docs = self.filter_docs(result_docs, self.filter)

        if self.retrieve_type == "chunk":
            result_docs = result_docs[: self.max_elements]
        else:
            # Retrieve new list of documents, where each source is only represented once.
            new_result_docs = []
            # Keep track of which sources have been added to the new list, and what position they have in the new_result_docs.
            added_sources = {}
            index = 0
            for doc in result_docs:
                if doc.metadata["source"] not in added_sources.keys():
                    doc.metadata["chunk_count"] = 1
                    new_result_docs.append(doc)

                    added_sources[doc.metadata["source"]] = index
                    index += 1
                    if len(new_result_docs) == self.max_elements:
                        break
                else:
                    new_result_docs[added_sources[doc.metadata["source"]]].metadata[
                        "chunk_count"
                    ] += 1

            result_docs = new_result_docs

        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(result_docs)
        return reordered_docs

    def get_parent_docs(self, chunks: List[Document]):
        """
        Retrieves the parent documents for a list of `Document` objects.

        Args:
            docs (List[Document]): The list of `Document` objects to retrieve parent documents for.

        Returns:
            List[Document]: A list of `Document` objects that are the parent documents of the input `Document` objects.
        """
        parent_docs = []
        for chunk in chunks:
            text_loader = TextLoader(self.base_path + chunk.metadata["source"])

            parent_doc = text_loader.load()[0]
            parent_doc.metadata = copy.deepcopy(chunk.metadata)

            parent_docs.append(parent_doc)

        return parent_docs

    def add_metadata(slef, docs: List[Document], parent_docs: List[Document]):
        """
        Adds metadata to a list of `Document` objects based on their parent documents and relevance scores.

        Args:
            docs (List[Document]): The list of `Document` objects to add metadata to.
            parent_docs (List[Document]): The list of parent `Document` objects.

        Returns:
            List[Document]: A list of `Document` objects with added metadata.
        """
        for doc in docs:
            parent_doc = [
                parent_doc
                for parent_doc in parent_docs
                if parent_doc.metadata["source"] == doc.metadata["source"]
            ][0]
            doc.metadata["har_mindretall"] = (
                "mindretall" in parent_doc.page_content.lower()
            )
            doc.metadata["tjenesteyter_avviser"] = (
                "tjenesteyter følger ikke vedtaket i saken"
                in parent_doc.page_content.lower()
            )

        return docs

    def custom_retrieve(self, parent_docs: List[Document]):
        """
        Retrieves a custom set of documents based on their parent documents.

        Args:
            parent_docs (List[Document]): The list of parent `Document` objects to retrieve custom documents for.

        Returns:
            List[Document]: A list of `Document` objects that are a custom set of documents based on their parent documents.
        """
        results = copy.deepcopy(parent_docs)
        for doc in results:
            cutoff = doc.page_content.find("Nemnda bemerker")
            doc.page_content = doc.page_content[cutoff:]

        return results

    def filter_docs(self, docs: List[Document], filter: dict) -> List[Document]:
        """
        Filters a list of `Document` objects based on a set of criteria specified in a dictionary.

        Args:
            docs (List[Document]): The list of `Document` objects to filter.
            filter (dict): A dictionary specifying the filtering criteria. The dictionary can contain the following keys:
                - "casetype": A list of case types to include in the filtered list.
                - "year": A list of years to include in the filtered list.
                - "exclude_case": A list of case numbers to exclude from the filtered list.
                - "har_mindretall": A boolean indicating whether to include only documents with dissenting opinions.
                - "tjenesteyter_avviser": A boolean indicating whether to include only documents where the service provider does not comply with the decision.

        Returns:
            List[Document]: A list of `Document` objects that satisfy the filtering criteria.
        """
        filtered_docs = []
        for doc in docs:
            doc.metadata["source"] = doc.metadata["source"].split("/")[-1]
            casetype, year, number = doc.metadata["source"].split(".")[
                0].split("-")

            if "casetype" in filter and casetype not in filter["casetype"]:
                continue

            if "year" in filter and int(year) not in filter["year"]:
                continue

            if (
                "exclude_case" in filter
                and f"{year}-{number}" in filter["exclude_case"]
            ):
                continue

            if (
                "har_mindretall" in filter
                and filter["har_mindretall"] != doc.metadata["har_mindretall"]
            ):
                continue

            if (
                "tjenesteyter_avviser" in filter
                and filter["tjenesteyter_avviser"]
                != doc.metadata["tjenesteyter_avviser"]
            ):
                continue

            filtered_docs.append(doc)

        return filtered_docs
