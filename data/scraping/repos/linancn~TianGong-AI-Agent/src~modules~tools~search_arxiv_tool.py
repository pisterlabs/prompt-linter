import io
from collections import Counter
from typing import Optional, Type

import arxiv
import pdfplumber
import requests
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain.vectorstores import FAISS
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed


class SearchArxivTool(BaseTool):
    name = "search_arxiv_tool"
    description = "Search arxiv.org for results."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def download_file_to_stream(self, url):
        # Fake User-Agent
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
        }

        response = requests.get(url, stream=True, headers=headers)

        if response.status_code != 200:
            response.raise_for_status()

        file_stream = io.BytesIO()

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_stream.write(chunk)

        file_stream.seek(0)
        return file_stream

    def parse_paper(self, pdf_stream):
        # logging.info("Parsing paper")
        pdf_obj = pdfplumber.open(pdf_stream)
        number_of_pages = len(pdf_obj.pages)
        # logging.info(f"Total number of pages: {number_of_pages}")
        full_text = ""
        ismisc = False
        for i in range(number_of_pages):
            page = pdf_obj.pages[i]
            if i == 0:
                isfirstpage = True
            else:
                isfirstpage = False

            page_text = []
            sentences = []
            processed_text = []

            def visitor_body(text, isfirstpage, x, top, bottom, fontSize, ismisc):
                # ignore header/footer
                if isfirstpage:
                    if (top > 200 and bottom < 720) and (len(text.strip()) > 1):
                        sentences.append(
                            {
                                "fontsize": fontSize,
                                "text": " " + text.strip().replace("\x03", ""),
                                "x": x,
                                "y": top,
                            }
                        )
                else:  # not first page
                    if (
                        (top > 70 and bottom < 720)
                        and (len(text.strip()) > 1)
                        and not ismisc
                    ):  # main text region
                        sentences.append(
                            {
                                "fontsize": fontSize,
                                "text": " " + text.strip().replace("\x03", ""),
                                "x": x,
                                "y": top,
                            }
                        )
                    elif (
                        (top > 70 and bottom < 720)
                        and (len(text.strip()) > 1)
                        and ismisc
                    ):
                        pass

            extracted_words = page.extract_words(
                x_tolerance=1,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=True,
                horizontal_ltr=True,
                vertical_ttb=True,
                extra_attrs=["fontname", "size"],
                split_at_punctuation=False,
            )

            # Treat the first page, main text, and references differently, specifically targeted at headers
            # Define a list of keywords to ignore
            # Online is for Nauture papers
            keywords_for_misc = [
                "References",
                "REFERENCES",
                "Bibliography",
                "BIBLIOGRAPHY",
                "Acknowledgements",
                "ACKNOWLEDGEMENTS",
                "Acknowledgments",
                "ACKNOWLEDGMENTS",
                "参考文献",
                "致谢",
                "謝辞",
                "謝",
                "Online",
            ]

            prev_word_size = None
            prev_word_font = None
            # Loop through the extracted words
            for extracted_word in extracted_words:
                # Strip the text and remove any special characters
                text = extracted_word["text"].strip().replace("\x03", "")

                # Check if the text contains any of the keywords to ignore
                if any(keyword in text for keyword in keywords_for_misc) and (
                    prev_word_size != extracted_word["size"]
                    or prev_word_font != extracted_word["fontname"]
                ):
                    ismisc = True

                prev_word_size = extracted_word["size"]
                prev_word_font = extracted_word["fontname"]

                # Call the visitor_body function with the relevant arguments
                visitor_body(
                    text,
                    isfirstpage,
                    extracted_word["x0"],
                    extracted_word["top"],
                    extracted_word["bottom"],
                    extracted_word["size"],
                    ismisc,
                )

            if sentences:
                for sentence in sentences:
                    page_text.append(sentence)

            blob_font_sizes = []
            blob_font_size = None
            blob_text = ""
            processed_text = ""
            tolerance = 1

            # Preprocessing for main text font size
            if page_text != []:
                if len(page_text) == 1:
                    blob_font_sizes.append(page_text[0]["fontsize"])
                else:
                    for t in page_text:
                        blob_font_sizes.append(t["fontsize"])
                blob_font_size = Counter(blob_font_sizes).most_common(1)[0][0]

            if page_text != []:
                if len(page_text) == 1:
                    if (
                        blob_font_size - tolerance
                        <= page_text[0]["fontsize"]
                        <= blob_font_size + tolerance
                    ):
                        processed_text += page_text[0]["text"]
                        # processed_text.append({"text": page_text[0]["text"], "page": i + 1})
                else:
                    for t in range(len(page_text)):
                        if (
                            blob_font_size - tolerance
                            <= page_text[t]["fontsize"]
                            <= blob_font_size + tolerance
                        ):
                            blob_text += f"{page_text[t]['text']}"
                            if len(blob_text) >= 500:  # set the length of a data chunk
                                processed_text += blob_text
                                # processed_text.append({"text": blob_text, "page": i + 1})
                                blob_text = ""
                            elif t == len(page_text) - 1:  # last element
                                processed_text += blob_text
                                # processed_text.append({"text": blob_text, "page": i + 1})
                full_text += processed_text

        # logging.info("Done parsing paper")
        return full_text

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        docs = arxiv.Search(
            query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance
        ).results()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=10
        )
        chunks = []

        for doc in docs:
            pdf_stream = self.download_file_to_stream(url=doc.pdf_url)

            page_content = self.parse_paper(pdf_stream)
            authors = ", ".join(str(author) for author in doc.authors)
            date = doc.published.strftime("%Y-%m")

            source = "[{}. {}. {}.]({})".format(
                authors,
                doc.title,
                date,
                doc.entry_id,
            )

            chunk = text_splitter.create_documents(
                [page_content], metadatas=[{"source": source}]
            )

            chunks.extend(chunk)

        embeddings = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(chunks, embeddings)

        result_docs = faiss_db.similarity_search(query, k=16)
        docs_list = []
        for doc in result_docs:
            source_entry = doc.metadata["source"]
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        docs = arxiv.Search(
            query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance
        ).results()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=10
        )
        chunks = []

        for doc in docs:
            pdf_stream = self.download_file_to_stream(url=doc.pdf_url)

            page_content = self.parse_paper(pdf_stream)
            authors = ", ".join(str(author) for author in doc.authors)
            date = doc.published.strftime("%Y-%m")

            source = "[{}. {}. {}.]({})".format(
                authors,
                doc.title,
                date,
                doc.entry_id,
            )

            chunk = text_splitter.create_documents(
                [page_content], metadatas=[{"source": source}]
            )

            chunks.extend(chunk)

        embeddings = OpenAIEmbeddings()
        faiss_db = FAISS.from_documents(chunks, embeddings)

        result_docs = faiss_db.similarity_search(query, k=16)
        docs_list = []
        for doc in result_docs:
            source_entry = doc.metadata["source"]
            docs_list.append({"content": doc.page_content, "source": source_entry})

        return docs_list
