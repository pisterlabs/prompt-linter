import asyncio
import multiprocessing
import os
import traceback
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlmodel import Session, select

from fda.db.engine import engine
from fda.db.models import ApplicationDocument, DocumentSegment
from fda.functions.pdf_to_text import pdf_to_text

index = os.environ.get("JOB_COMPLETION_INDEX", "0")
num_workers = os.environ.get("NUM_WORKERS", "1")


def split_document(doc: ApplicationDocument) -> List[DocumentSegment]:
    try:
        if doc.url.endswith("pdf"):
            text = pdf_to_text(doc.url)
        elif doc.url.endswith("cfm") or doc.url.endswith("htm"):
            directory = "/".join(doc.url.split("/")[:-1])
            text = []
            res = httpx.get(doc.url, follow_redirects=True)
            soup = BeautifulSoup(res.content, "html.parser")
            links = soup.find_all(
                lambda tag: tag.has_attr("href") and tag["href"].endswith(".pdf")
            )
            for link in links:
                pdf_url = os.path.join(directory, link["href"])
                pdf_text = pdf_to_text(pdf_url)
                text += pdf_text
        else:
            raise Exception(f"Unknown document type: {doc.url}")
        splitter = RecursiveCharacterTextSplitter()
        segments = splitter.create_documents(text)
        return [
            DocumentSegment(
                document_id=doc.id,
                segment_number=i,
                content=segment.page_content,
            )
            for i, segment in enumerate(segments)
        ]
    except Exception as e:
        traceback.print_exc()
        print(f"Error splitting document {doc.id}, {doc.url}, {e}")
        return []


def split_documents(index: int = 0, num_workers: int = 1) -> None:
    with Session(engine) as session:
        rows = session.query(ApplicationDocument).count()
        offset = rows // num_workers * index
        limit = rows // num_workers
        print(f"offset: {offset}, limit: {limit}")
        statement = select(ApplicationDocument).offset(offset).limit(limit)
        docs = session.exec(statement)

    for doc in docs:
        segments = split_document(doc)
        with Session(engine) as session:
            session.add_all(segments)
            session.commit()


if __name__ == "__main__":
    index = int(index)
    num_workers = int(num_workers)
    split_documents(index=index, num_workers=num_workers)
