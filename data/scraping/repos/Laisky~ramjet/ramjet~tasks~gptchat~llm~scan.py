import base64
import os
import re
import tempfile
from concurrent.futures import Future
from textwrap import dedent
from typing import List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.schema.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import TokenTextSplitter

from ramjet.engines import thread_executor


def summary_content(
    b64content: str,
    ext: str,
    apikey: Optional[str] = None,
    api_base: str = "https://api.openai.com/v1",
) -> str:
    """Summarize the content of a document.

    Args:
        b64content (str): The base64 encoded content of the document
        ext (str): The extension of the document,
            should be one of: .docx, .pptx, .pdf, .html, .md, .txt
        apikey (str, optional): The openai api key. Defaults to None
        api_base (str, optional): The openai api base url. Defaults to "https://api.openai.com/v1"

    Returns:
        The summary of the document.
    """
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=500, chunk_overlap=30, separator="\n"
    # )
    text_splitter = TokenTextSplitter(
        chunk_size=3000,
        chunk_overlap=30,
    )

    # write to file
    file_content = base64.b64decode(b64content)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "tmpfile")
        with open(tmpfile, "wb") as f:
            f.write(file_content)

        docus: List[Document]
        loader: BaseLoader
        if ext in [".docx"]:
            loader = UnstructuredWordDocumentLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext in [".pptx"]:
            loader = UnstructuredPowerPointLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext in [".pdf"]:
            loader = PyPDFLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext in [".html"]:
            loader = BSHTMLLoader(tmpfile)
            docus = loader.load_and_split(text_splitter=text_splitter)
        elif ext in [".md"]:
            # docus = MarkdownTextSplitter(
            #     chunk_size=500, chunk_overlap=50
            # ).create_documents([file_content.decode("utf-8")])
            docus = text_splitter.create_documents([file_content.decode("utf-8")])
        elif ext in [".txt"]:
            docus = text_splitter.create_documents([file_content.decode("utf-8")])
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    return _get_question_tobe_summary(docus, apikey=apikey, api_base=api_base)


def _get_question_tobe_summary(
    docus: List[Document], apikey: Optional[str] = None, api_base: str = "https://api.openai.com/v1"
) -> str:
    """return the question can be give to LLM to summarize the documents.

    Args:
        docus (List[Document]): The documents to be summarized
        apikey (str, optional): The openai api key. Defaults to None
        api_base (str, optional): The openai api base url. Defaults to "https://api.openai.com/v1"

    Returns:
        str: The question to be summarized
    """

    summary: str = ""
    # map
    fs: List[Future] = []
    for docu in docus:
        fs.append(thread_executor.submit(summary_docu, docu, apikey=apikey, api_base=api_base))
    for f in fs:
        summary += f"* {f.result()}\n"

    # reduce
    query = dedent(
        f"""
        The following is set of summaries:

        {summary}

        Take these and distill it into a final, consolidated summary of the main themes.
        Helpful Answer:
        """
    )
    return query

    # reduce by go-ramjet, do not use it for now

    # apikey = apikey or os.environ["OPENAI_API_KEY"]
    # llm = ChatOpenAI(
    #     client=None,
    #     openai_api_key=apikey,
    #     model="gpt-3.5-turbo",
    #     temperature=0,
    #     max_tokens=1000,
    #     streaming=False,
    # )

    # return llm.predict(query)


def summary_docu(
    docu: Document,
    apikey: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    api_base: str = "https://api.openai.com/v1",
) -> str:
    """Summarize a document.

    Args:
        docu (str): A document
        apikey (str): The openai api key
        model (str, optional): The openai model
        api_base (str, optional): The openai api base url

    Returns:
        The summary of the document.
    """
    max_token = 500
    if apikey and re.match(r"\-\d+k$", apikey, re.I):
        max_token = 10000

    llm = ChatOpenAI(
        client=None,
        openai_api_key=apikey,
        openai_api_base=api_base,
        model=model,
        temperature=0,
        max_tokens=max_token,
        streaming=False,
    )

    query = dedent(
        f"""
        Write a concise summary of the following content between "@>>>>>" and "@<<<<<",
        just response the summary text in a single {'short ' if max_token <= 500 else ' '}line,
        just contains necessary key informations,
        do not contains any other words:

        @>>>>>
        {docu.page_content}
        @<<<<<

        CONCISE SUMMARY:
    """
    )
    return llm.predict(query)
