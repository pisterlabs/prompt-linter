from collections import defaultdict
from dataclasses import dataclass, field
import io
from typing import Dict, List, Optional
from langchain.docstore.document import Document
import fitz
import asyncio

# wrapper which automatically parses PDF into unicode text
from langchain.document_loaders.arxiv import ArxivAPIWrapper
# base search which returns Result objects
from arxiv import SortCriterion, SortOrder, Search, Result
import requests

class ArxivFetch:

    def __init__(self, doc_content_chars_max: Optional[int] = None):
        # load 1 doc max since only care about 1st result
        self._search = ArxivAPIWrapper(
            doc_content_chars_max=doc_content_chars_max, 
            load_all_available_meta=True,
            load_max_docs=1
        )

    @staticmethod
    def url_from_id(paper_id: str) -> str:
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    @classmethod
    def get_pdf_txt(cls, paper_id: str, exclude_references=True):
        resp = requests.get(cls.url_from_id(paper_id), stream=True)
        stream = io.BytesIO(resp.content)

        page_txts = []
        with fitz.Document(stream=stream) as pdf:
            for page in pdf.pages():
                txt = page.get_text()
                page_txts.append(txt)
        
        pdf_contents = "".join(page_txts)

        if exclude_references:
            # TODO: use LLM to classify each "reference" occurence by showing surrounding text
            # find last occurence of 'references' incase the paper content mentions it before the heading
            idx = pdf_contents.lower().rindex("reference")
            pdf_contents = pdf_contents[:idx]
        
        return pdf_contents

    @staticmethod
    def _short_id(url: str):
        """Convert https://arxiv.org/abs/XXXX.YYYYYvZ to XXXX.YYYYY"""
        return url.split("/")[-1].split("v")[0]

    def get_doc_sync(self, paper_id: str) -> tuple[Document, str]:
        """Get a PDF contents for the paper_id, exluding all text after the "References" section.
        This saves a lot of tokens for summarization methods and reduces size in vector store.
        
        Also returns paper's abstract.
        """
        # use arXiv API for the paper title
        search = Search(query=paper_id, max_results=1)
        
        found = False
        for result in search.results():
            found = True

            title = result.title
            abstract = result.summary
            break
        
        if not found:
            raise Exception("Paper not found")
        
        # get the paper text, excluding all text after references
        pdf_txt = self.get_pdf_txt(paper_id, exclude_references=True)
        metadata = {
            "source": paper_id,
            "title": title
        }
        return Document(page_content=pdf_txt, metadata=metadata), abstract
    
    async def get_doc_async(self, paper_id: str):
        return await asyncio.get_event_loop().run_in_executor(None, self.get_doc_sync, paper_id)
    
    def _search_papers(self, query: str):
        search = Search(
            query, 
            max_results=5, 
            sort_by=SortCriterion.Relevance,
            sort_order=SortOrder.Descending
        )
        return search.results()
    
    
    def search_sync(self, query: str) -> List[str]:
        return self._search_papers(query)

    async def search_async(self, query: str):
        loop = asyncio.get_event_loop()
        # use default executor (thread pool)
        return await loop.run_in_executor(None, self._search_papers, query)
        
    

@dataclass
class PaperMetadata:
    title: str
    source: str

    def short_repr(self):
        return f"{self.title} - {self.source}"

@dataclass
class LoadedPapersStore:
    _to_papers: Dict[str, list[PaperMetadata]] = field(default_factory=lambda: defaultdict(list))
    _id_to_title: Dict[str, str] = field(default_factory=dict)

    def get(self, chat_id: str):
        return self._to_papers[chat_id]

    def add_papers(self, chat_id: str, paper_metas: List[PaperMetadata]):
        self._to_papers[chat_id].extend(paper_metas)

    def register(self, papers: List[PaperMetadata]):
        """Create mapping from id to title"""
        for paper in papers:
            self._id_to_title[paper.source] = paper.title

    def get_title(self, paper_id: str):
        return self._id_to_title[paper_id]
    