# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import deque
from functools import lru_cache
import re
from typing import Any, Dict, List
from abc import ABC, abstractmethod
import xml.etree.ElementTree as ET

import io
import zipfile
import pdfplumber as plumber

from langchain.vectorstores import Chroma
from langchain.schema import Document as LangChainDocument
from langchain.embeddings import OpenAIEmbeddings

from app.constant import LANGCHAIN_VECTOR_DB, EMBEDDING_FUNC


class BaseDocumentLoader(ABC):
    def __init__(
        self,
        text_splitter: Any,
        vector_db: Chroma = LANGCHAIN_VECTOR_DB,
        embedding_model: OpenAIEmbeddings = EMBEDDING_FUNC,
    ):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter

    def embed_document(self, document: LangChainDocument) -> List[List[float]]:
        return self.embedding_model.embed_documents([document.page_content])

    @abstractmethod
    def load_document(self, document_content: bytes, document_name: str):
        pass

    def upload_to_db(self, document: LangChainDocument) -> None:
        self.vector_db.add_documents(document)

    def search_docs(self, query_input):
        return self.vector_db.similarity_search_by_vector(self.embedding_model.embed_query(query_input), k=2)


class PDFDocumentLoader(BaseDocumentLoader):
    LINE_HEIGHT_DIFF = (5, 30)
    LEFT_X_DIFF = 5
    LEFT_Y_DIFF = 20

    def __init__(
        self,
        text_splitter: Any,
        vector_db: Chroma = LANGCHAIN_VECTOR_DB,
        embedding_model: OpenAIEmbeddings = EMBEDDING_FUNC,
    ):
        super().__init__(text_splitter, vector_db, embedding_model)

    def sort_horizontal(self, ocr_result: List[Dict[str, str | float | int | bool]]) -> Dict[str, str | float | int]:
        """
        Sorting the OCR Result dictionary based on horizontal coordinates using the top-bottom values

        Args:
            ocr_result (List[Dict[str, str  |  float  |  int  |  bool]]): the ocr result using pdfplumber as backbone to parse

        Returns:
            Dict[str, str | float | int]: a dictionary contains all the text and coordinates after grouping them based on top-bottom values
        """
        result = {}

        for data in ocr_result:
            x0, x1, top, bottom = data["x0"], data["x1"], data["top"], data["bottom"]
            text = data["text"]

            top_left = [x0, bottom]  # x0, y1
            top_right = [x1, bottom]  # x1, y1
            bottom_left = [x0, top]  # x0, y0
            bottom_right = [x1, top]  # x1, y0

            if (top, bottom) in result:
                result[(top, bottom)]["text"] += f" {text}"
                result[(top, bottom)]["top_right"] = top_right
            else:
                result[(top, bottom)] = {
                    "text": text,
                    "top_left": top_left,
                    "top_right": top_right,
                    "bottom_left": bottom_left,
                }

            result[(top, bottom)]["bottom_right"] = bottom_right
        return result

    def sort_vertical(self, input_dicts: Dict[str, str | float | int]) -> List[str]:
        """
        Sorts the input dictionary values vertically based on their position on the page.

        Args:
            input_dicts (Dict[str, Union[str, float, int]]): A dictionary containing the text and position of each element.

        Returns:
            List[str]: A list of strings containing the sorted text elements.
        """
        result = []
        remain = deque(input_dicts.values())
        remove_indices = set()

        while remain:
            current_value = remain.popleft()
            current_text = current_value["text"]
            current_top_left = current_value["top_left"]
            current_bottom_left = current_value["bottom_left"]

            remain, remove_indices = self._merge_lines(remain, remove_indices, current_value, current_text, current_top_left, current_bottom_left)

            result.append(current_text)

        return result

    def _merge_lines(self, remain, remove_indices, current_value, current_text, current_top_left, current_bottom_left):
        for i, value in enumerate(remain):
            line_height_diff, diff_top_left, diff_bottom_left = self._calculate_distances(current_value, value)

            if self._should_merge_lines(line_height_diff, diff_top_left, diff_bottom_left):
                current_text += f" \n {value['text']}"
                current_top_left = value["top_left"]  # noqa: F841
                current_bottom_left = value["bottom_left"]  # noqa: F841
                remove_indices.add(i)

        remain = deque(val for i, val in enumerate(remain) if i not in remove_indices)
        remove_indices.clear()
        return remain, remove_indices

    def _calculate_distances(self, current_value, value):
        line_height_diff = [
            abs(current_value["bottom_left"][0] - value["top_left"][0]),
            abs(current_value["bottom_left"][1] - value["top_left"][1]),
        ]
        diff_top_left = [abs(current_value["top_left"][0] - value["top_left"][0]), abs(current_value["top_left"][1] - value["top_left"][1])]
        diff_bottom_left = [abs(current_value["bottom_left"][0] - value["bottom_left"][0]), abs(current_value["bottom_left"][1] - value["bottom_left"][1])]
        return line_height_diff, diff_top_left, diff_bottom_left

    def _should_merge_lines(self, line_height_diff, diff_top_left, diff_bottom_left):
        return line_height_diff[0] <= 5 and line_height_diff[1] <= 30 and diff_top_left[0] <= 5 and diff_top_left[1] <= 20 and diff_bottom_left[0] <= 5 and diff_bottom_left[1] <= 20

    @lru_cache
    def read_pdf(self, file_content) -> List[Dict[str, str | float | int | bool]]:
        """
        Reads a PDF file and extracts text from it using OCR.

        Args:
            file_content (bytes): The content of the PDF file.

        Returns:
            List[Dict[str, str | float | int | bool]]: A list of dictionaries, where each dictionary represents a page in the PDF file and contains the extracted text.
        """

        ocr_result = []

        with plumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                output = page.dedupe_chars().extract_words(x_tolerance=2, y_tolerance=2)
                ocr_result.append(output)

        return ocr_result

    @lru_cache
    def load_document(self, document_content: bytes, document_name: str) -> LangChainDocument:
        ocr_output = self.read_pdf(document_content)
        result = []
        for page in ocr_output:
            horizontally_sorted = self.sort_horizontal(page)
            vertically_sorted = self.sort_vertical(horizontally_sorted)
            result.extend(vertically_sorted)

        return LangChainDocument(page_content="\n".join(result), metadata={"filename": document_name})


class WordDocumentLoader(BaseDocumentLoader):
    NAMESPACE_MAP = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    def __init__(
        self,
        text_splitter: Any,
        vector_db: Chroma = LANGCHAIN_VECTOR_DB,
        embedding_model: OpenAIEmbeddings = EMBEDDING_FUNC,
    ):
        super().__init__(text_splitter, vector_db, embedding_model)

    def qn(self, tag):
        """
        Stands for 'qualified name', a utility function to turn a namespace
        prefixed tag name into a Clark-notation qualified tag name for lxml.
        """
        prefix, tagroot = tag.split(":")
        return "{{{}}}{}".format(self.NAMESPACE_MAP[prefix], tagroot)

    def xml2text(self, xml):
        """
        Convert XML content to textual content, translating specific tags to their Python equivalent.
        """
        tag_translations = {
            self.qn("w:t"): lambda el: el.text or "",
            self.qn("w:tab"): lambda el: "\t",
            self.qn("w:br"): lambda el: "\n",
            self.qn("w:cr"): lambda el: "\n",
            self.qn("w:p"): lambda el: "\n\n",
        }

        root = ET.fromstring(xml)
        return "".join(tag_translations.get(child.tag, lambda el: "")(child) for child in root.iter())

    def process(self, document) -> str:
        """
        Processes a document and extracts the main text from a DOCX file.

        Args:
            document: The DOCX file to be processed.

        Returns:
            str: The extracted main text from the document.

        """
        # unzip the docx in memory
        with zipfile.ZipFile(document) as zip_file:
            file_list = zip_file.namelist()

            # compile regular expressions for faster matching
            header_re = re.compile("word/header[0-9]*.xml")
            footer_re = re.compile("word/footer[0-9]*.xml")

            # using list comprehensions and generator expressions to minimize explicit for-loops
            headers = (self.xml2text(zip_file.read(fname)) for fname in file_list if header_re.match(fname))
            footers = (self.xml2text(zip_file.read(fname)) for fname in file_list if footer_re.match(fname))

            # get main text
            doc_xml_content = zip_file.read("word/document.xml")

            # concatenate texts
            text = "".join([*headers, self.xml2text(doc_xml_content), *footers])

        return text.strip()

    @lru_cache
    def load_document(self, document_content: bytes, document_name: str) -> LangChainDocument:
        return LangChainDocument(
            page_content=self.process(io.BytesIO(document_content)),
            metadata={"filename": document_name},
        )
