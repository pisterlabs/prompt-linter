import re
from typing import Any, Coroutine, List, Tuple, Optional
from pydantic import Field
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever, Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
# TODO - which pdfloader is best for engineering drawings? perhaps UnstructuredPDFLoader with mode="elements" and strategy="hi_res"?
from langchain.document_loaders import PyPDFLoader
from loguru import logger

from meche_copilot.schemas import AgentConfig, Source
from meche_copilot.pdf_helpers.get_page_from_sheet import get_page_from_sheet
from meche_copilot.pdf_helpers.get_pages_from_text import get_pages_from_text

from meche_copilot.utils.envars import OPENAI_API_KEY

# TODO - in the future, consider using Grobid to extract text from PDFs since these types of pdfs are engineering drawings and things with structured data and we'd like to retain metadata with the text we lookup

class SpecsRetriever(BaseRetriever):
  """Given reference notes about what specs we are looking for and/or where to find them, return relavent docs

  - If a sheet is mentioned in the ref_notes (regex a letter followed by a dash and three numbers like this M-802), get docs with that page number

  - If a page is mentioned in the ref_notes (regex: p. or page or p or pn or pg. etc), get the docs with that page number

  - If quotes are found (' or ") in the ref_notes find exact matches of that quote
  
  """

  doc_retriever: AgentConfig
  source: Source
  
  chroma_db: Optional[Chroma]

  def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs) -> List[Document]:
    relavent_docs: List[Document] = []
    relavent_page_source: List[Tuple[str, str]] = []

    refresh_source_docs = kwargs.get('refresh_source_docs', None)
    if refresh_source_docs is not None:
        logger.debug(f'refresh_source_docs was passed with value {refresh_source_docs}')
    else:
        logger.debug('refresh_source_docs was not passed')

    self.check_db_contents(refresh_source_docs=refresh_source_docs)

    sheet_regex = r'[A-Za-z]-\d{3}'
    sheet_matches = re.findall(sheet_regex, self.source.notes)
    logger.debug(f"ref_notes mentioned sheets: {sheet_matches}")
    if len(sheet_matches) > 0:
      for sheet in sheet_matches:
        for src_fpath in self.source.ref_docs:
          res = get_page_from_sheet(sheet=sheet, pdf_fpath=src_fpath)
          if res is not None:
            relavent_page_source.append((res[0], src_fpath))
          else:
            logger.warning(f"Could not find sheet {sheet} in {src_fpath}")
    
    page_regex = r'(p\.|page|p|pn|pg\.)\s*(\d+)'
    page_matches = re.findall(page_regex, self.source.notes, re.IGNORECASE)
    logger.debug(f"ref_notes mentioned pages: {page_matches}")
    if len(page_matches) > 0:
      for match in page_matches:
        # NOTE - this is a kinda shitty way of doing this cuz it will get that page for each source rather than in the correct source...do better later
        page_num = match[1]
        for src_fpath in self.source.ref_docs:
          relavent_page_source.append((page_num, src_fpath))
    
    quote_regex = r"(?<=')[^']*(?=')|(?<=\")[^\"]*(?=\")"
    quote_matches = re.findall(quote_regex, self.source.notes)
    logger.debug(f"ref_notes mentioned quotes: {quote_matches}")
    if len(quote_matches) > 0:
      for quote in quote_matches:
        for src_fpath in self.source.ref_docs:
          pages = get_pages_from_text(text=quote, pdf_fpath=src_fpath)
          for pg in pages:
            relavent_page_source.append((pg, src_fpath))

    unique_page_src = list(set(relavent_page_source))
    logger.debug(f"total unique pg/src from ref notes: {unique_page_src}")

    if len(unique_page_src) == 0:
      # use similarity search to get ref docs from vectorstore
      logger.warning("Reference notes did not mention any specific pages, sheets, or quotes to look for so defaulting to similarity search")
      relavent_docs = self.chroma_db.similarity_search(query=self.source.notes)
    else:
      # get document with page number
      for pg, src in unique_page_src:
        logger.debug(f"Getting pg {pg} of {src}")

        # NOTE: must do in two steps cuz .get doesn't allow a where dict with multiple keys

        # get all doc ids with that page number
        ids = self.chroma_db.get(
          where={"page": pg}
        )['ids']

        # filter by the correct source
        res = self.chroma_db.get(
          ids=ids,
          where={'source': str(src)}
        )

        if len(res['documents']) == 0:
          logger.warning(f"Could not find page {pg} in {src}")
          raise ValueError("Couldn't find document in database")
    
        relavent_docs.append(Document(
          page_content=res['documents'][0],
          metadata=res['metadatas'][0]
        ))
    
    return relavent_docs
  
  def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> Coroutine[Any, Any, List[Document]]:
    raise NotImplementedError
  
  def check_db_contents(self, refresh_source_docs: bool = False):
    """Make sure that chroma_db has all the source ref docs and update if necessary"""

    logger.info("Checking vectorstore db contents against source ref docs")

    if not self.chroma_db:
      self.chroma_db = Chroma(
          embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
          persist_directory="data/.chroma_db"
      )

    # get all ids associated with source ref docs
    all_ids = self.chroma_db.get()['ids']
    all_ids_for_source = []
    for fpath in self.source.ref_docs:
      all_ids_for_source.extend(self.chroma_db.get(
        ids=all_ids, 
        where={'source': str(fpath)})['ids']
      )
    

    if len(all_ids_for_source) == 0:
      logger.info("No documents found in vectorstore for source ref docs. Adding documents from source ref docs")
      documents: List[Document] = []
      for fpath in self.source.ref_docs:
          loader = PyPDFLoader(str(fpath))
          documents.extend(loader.load())
          logger.debug(f"Adding document: {fpath}")
      self.chroma_db.add_documents(documents)
    
    elif refresh_source_docs == True:
        logger.info(f"Refreshing {len(self.source.ref_docs)} documents in vectorstore")
        for fpath in self.source.ref_docs:
          logger.debug(f"Updating document: {fpath}")
          loader = PyPDFLoader(str(fpath))
          document = loader.load()
          # get ids of the documents from the same source
          src_doc_ids = self.chroma_db.get(where={'source': str(fpath)})['ids']
          if src_doc_ids:  # if documents exist, delete them
              self.chroma_db.delete(src_doc_ids)
          # add the new document
          self.chroma_db.add_documents([document])