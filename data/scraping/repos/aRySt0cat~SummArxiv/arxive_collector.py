from pathlib import Path
from time import sleep

import arxiv
import faiss
import requests
import yaml
from langchain.chains import LLMChain
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import FAISS

import settings
from notion_utils import *


class ArxivCollector:
    """Collect papers from arxiv and post them to Notion.

    Attributes:
        queries (list[str]): arxiv query
        embedding_model (OpenAIEmbeddings): embedding model
        vectorstore (FAISS): vectorstore
        create_summary (LLMChain): LLMChain for summarization
        headers (dict): headers for Notion API
    """

    def __init__(
        self,
        query_file: str | Path,
        vectorstore_path: str | Path = settings.VECTORSTORE_PATH,
    ):
        # arxiv query setting
        with open(query_file, "r") as f:
            config = yaml.safe_load(f)
        self.queries = [self.parse_query(q) for q in config["query"]]

        # embedding setting
        dim = 1536
        self.embedding_model = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

        # load index
        self.vectorstore_path = vectorstore_path
        if self.load():
            print("Successfully loaded index.")
            print("The number of documents:", self.vectorstore.index.ntotal)
        else:
            print("Creating a new index.")
            index = faiss.IndexFlatL2(dim)
            self.vectorstore = FAISS(
                self.embedding_model.embed_query, index, InMemoryDocstore({}), {}
            )

        # summary setting
        prompt_template = PromptTemplate(
            input_variables=["article"], template=settings.SUMMARY_TEMPLATE
        )
        llm_chain = LLMChain(
            llm=OpenAI(
                openai_api_key=settings.OPENAI_API_KEY, temperature=0, max_tokens=1000
            ),
            prompt=prompt_template,
        )
        self.create_summary = llm_chain

        # notion header
        self.headers = {
            "Accept": "application/json",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
            "Authorization": "Bearer " + settings.NOTION_API_KEY,
        }

    def parse_query(self, query: str) -> str:
        query = query.replace("(", "%28").replace(")", "%29").replace('"', "%22")
        return query

    def collect(self) -> None:
        """Collect"""
        for query in self.queries:
            print("query =", query)
            search = arxiv.Search(
                query=query,
                max_results=settings.ARXIV_LIMIT,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )
            for result in search.results():
                meta_data = {
                    "title": result.title,
                    "url": result.entry_id,
                    "date": result.published,
                }
                page_content = result.summary

                embedding = self.embedding_model.embed_query(page_content)
                doc_score = self.vectorstore.similarity_search_with_score_by_vector(
                    embedding, k=settings.TOP_K
                )
                documents = [d for d, _ in doc_score]
                scores = [s for _, s in doc_score]
                # skip if the paper is already posted
                if len(documents) > 0 and (
                    documents[0].metadata["title"] == meta_data["title"]
                    or scores[0] < 1e-3
                ):
                    print(f"{meta_data['title']} is already posted.")
                    continue
                print(f"Posting {meta_data['title']} to Notion.")
                self.post_notion(page_content, meta_data, documents)
                self.vectorstore.add_embeddings(
                    [(page_content, embedding)], [meta_data]
                )
        self.save()

    def post_notion(
        self,
        page_content: str,
        meta_data: dict,
        related_papers: list[Document],
        summarize: bool = True,
    ) -> None:
        """Post the page content, related papers and summary to Notion.

        Args:
            page_content (str): abstract of the paper
            meta_data (dict): metadata of the paper
            related_papers (list[Document]): related papers,
            summarize (bool, optional): whether to summarize the abstract. Defaults to True.
        """
        if summarize:
            summary = self.create_summary.run({"article": page_content})
        else:
            summary = page_content

        summary = parse_output(summary)
        children = [
            {"object": "block", **create_header1("Abstract")},
            *summary,
            {"object": "block", **create_header1("Related Papers")},
        ]
        children += [
            {
                "object": "block",
                **create_linked_text(p.metadata["title"], p.metadata["url"]),
            }
            for p in related_papers
        ]
        print(meta_data["date"])
        data = {
            "parent": {"database_id": settings.NOTION_DB_ID},
            "properties": create_property(
                meta_data["title"], meta_data["url"], meta_data["date"], page_content
            ),
            "children": children,
        }
        response = requests.post(
            settings.NOTION_POST_URL, json=data, headers=self.headers
        )
        if response.status_code != 200:
            print(f"Failed to post to Notion. status code:{response.status_code}")
        else:
            print("Successfully posted to Notion.")
        sleep(0.5)

    def save(self, path: str | Path | None = None):
        if path is None:
            path = self.vectorstore_path
        self.vectorstore.save_local(path)

    def load(self, path: str | Path | None = None) -> bool:
        if path is None:
            path = self.vectorstore_path
        if isinstance(path, str):
            path = Path(path)

        if path.exists():
            self.vectorstore = FAISS.load_local(path, self.embedding_model)
            return True
        else:
            return False
