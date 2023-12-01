from langchain.tools import BaseTool
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from virtual_assistant.utilities import GoogleSerperAPIWrapper, PINECONE_INDEX_NAME

class BrowsingTool(BaseTool):
    serper: GoogleSerperAPIWrapper

    name =  "Browsing internet tool"

    description = (
        "A wrapper around search and internet browsing. "
        "Useful when you need retrieve lastest information or question context."
        "Input should be a search query."
    )

    def _run(self, query: str):
        search_results =  self.serper.run(query)
        return self._parse_results(search_results, query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BrowsingRun does not support async")

    def _parse_results(self, search_results: dict,  query: str) -> str:
        answer = []
        links =  []

        for r in search_results:
            if 'answer' in r:
                answer.append(r['title'] + ": ")
                answer.append(r['answer'])
                return " ".join(answer)

            if 'link' in r:
                links.append(r['link'])

        if len(answer) > 0:
            return " ".join(answer)

        loader = WebBaseLoader(links)
        documents = loader.load()

        embedding = OpenAIEmbeddings()

        text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        db = Pinecone.from_documents(index_name=PINECONE_INDEX_NAME, embedding=embedding, documents=docs, namespace="browsing_documents")

        results = db.as_retriever().get_relevant_documents(query)

        if len(results) > 0:
            return results[0].page_content

        return "No result from browsing internet."
