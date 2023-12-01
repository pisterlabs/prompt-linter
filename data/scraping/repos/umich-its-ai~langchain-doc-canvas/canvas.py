"""Loads Pages, Announcements, Assignments and Files from a Canvas Course site."""

import tempfile
import json
from io import BytesIO
from typing import List
from datetime import date, datetime
import pytz

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredURLLoader

from striprtf.striprtf import rtf_to_text

class CanvasLoader(BaseLoader):
    """Loading logic for Canvas Pages, Announcements, Assignments and Files."""

    def __init__(self, api_url: str, api_key: str = "", course_id: int = 0, index_external_urls: bool = False):
        """Initialize with API URL and api_key.

        Args:
            api_url: The canvas API URL endpoint.
            api_key: API Key or token.
            course_id: Course ID we want to return documents from
            index_external_urls: Whether to try and index ExternalUrls in modules - defauls is false
        """
        self.api_url = api_url
        self.api_key = api_key
        self.course_id = course_id
        self.returned_course_id = 0
        self.index_external_urls = index_external_urls

        self.invalid_files = []
        self.indexed_items = []

        self.errors = []

    def _get_page_url(self, page_url) -> str:
        return f"{self.api_url}/courses/{self.returned_course_id}/pages/{page_url}"

    def _get_file_url(self, file_id) -> str:
        return f"{self.api_url}/courses/{self.returned_course_id}/files/{file_id}"

    def load_pages(self, course) -> List[Document]:
        """Loads all published pages from a canvas course."""
        from canvasapi.exceptions import CanvasException

        page_documents = []

        try:
            pages = course.get_pages(
                published=True,
                include=[ "body" ]
            )

            for page in pages:
                if f"Page:{page.page_id}" not in self.indexed_items:
                    page_documents = page_documents + self.load_page(page)
                    self.indexed_items.append(f"Page:{page.page_id}")
        except CanvasException as error:
            self._error_logger(error=error, action="get_pages", entity_type="page", entity_id=page.page_id)

        return page_documents

    def load_page(self, page) -> List[Document]:
        """Load a specific page."""
        try:
            page_body_text = self._get_html_as_string(page.body)

            return [Document(
                page_content=page_body_text.strip(),
                metadata={ "filename": page.title, "source": self._get_page_url(page.url), "kind": "page", "page_id": page.page_id }
            )]
        except AttributeError:
            return []

    def load_announcements(self, canvas, course) -> List[Document]:
        """Loads all announcements from a canvas course."""
        from canvasapi.exceptions import CanvasException

        announcement_documents = []

        try:
            announcements = canvas.get_announcements(
                context_codes=[ course ],
                start_date="2016-01-01",
                end_date=date.today().isoformat(),
            )

            for announcement in announcements:
                page_body_text = self._get_html_as_string(announcement.message)

                announcement_documents.append(Document(
                    page_content=page_body_text,
                    metadata={ "filename": announcement.title, "source": announcement.html_url, "kind": "announcement", "announcement_id": announcement.id }
                ))
        except CanvasException as error:
            self._error_logger(error=error, action="get_announcements", entity_type="announcement", entity_id=announcement.id)

        return announcement_documents

    def load_assignments(self, course) -> List[Document]:
        """Loads all assignments from a canvas course."""
        from canvasapi.exceptions import CanvasException

        assignment_documents = []

        try:
            assignments = course.get_assignments()

            for assignment in assignments:
                if f"Assignment:{assignment.id}" not in self.indexed_items:
                    assignment_documents = assignment_documents + self.load_assignment(assignment)
                    self.indexed_items.append(f"Assignment:{assignment.id}")
        except CanvasException as error:
            self._error_logger(error=error, action="get_assignments", entity_type="assignment", entity_id=assignment.id)

        return assignment_documents

    def load_assignment(self, assignment) -> List[Document]:
        """Load a specific assignment."""
        if assignment.description:
            assignment_description = self._get_html_as_string(assignment.description)
            assignment_description = f" Assignment Description: {assignment_description}\n\n"
        else:
            assignment_description = ""

        assignment_content=f"Assignment Name: {assignment.name} \n\n Assignment Due Date: {assignment.due_at} \n\n{assignment_description}"

        return [Document(
            page_content=assignment_content,
            metadata={ "filename": assignment.name, "source": assignment.html_url, "kind": "assignment", "assignment_id": assignment.id }
        )]

    def _get_html_as_string(self, html) -> str:
        """Use BeautifulSoup 4 to parse a html string and return a simplified string."""
        try:
            # Import the html parser class
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "Could not import beautifulsoup4 python package. "
                "Please install it with `pip install beautifulsoup4`."
            ) from exc

        html_string = BeautifulSoup(html, "lxml").text.strip()

        return html_string

    def _load_text_file(self, file) -> List[Document]:
        file_contents = file.get_contents(binary=False)

        return [Document(
            page_content=file_contents.strip(),
            metadata={ "filename": file.filename, "source": file.url, "kind": "file", "file_id": file.id }
        )]

    def _load_html_file(self, file) -> List[Document]:
        file_contents = file.get_contents(binary=False)

        return [Document(
            page_content=self._get_html_as_string(file_contents),
            metadata={ "filename": file.filename, "source": file.url, "kind": "file", "file_id": file.id }
        )]

    def _load_rtf_file(self, file) -> List[Document]:
        file_contents = file.get_contents(binary=False)

        return [Document(
            page_content=rtf_to_text(file_contents).strip(),
            metadata={ "filename": file.filename, "source": file.url, "kind": "file", "file_id": file.id }
        )]

    def _load_pdf_file(self, file) -> List[Document]:
        try:
            # Import PDF parser class
            from PyPDF2 import PdfReader
            from PyPDF2 import errors
        except ImportError as exc:
            raise ImportError(
                "Could not import PyPDF2 python package. "
                "Please install it with `pip install PyPDF2`."
            ) from exc

        file_contents = file.get_contents(binary=True)

        docs = []

        try:
            pdf_reader = PdfReader(BytesIO(file_contents))

            for i, page in enumerate(pdf_reader.pages):
                docs.append(Document(
                    page_content=page.extract_text(),
                    metadata={ "filename": file.filename, "source": self._get_file_url(file.id), "kind": "file", "file_id": file.id, "page": i+1 }
                ))
        except errors.FileNotDecryptedError:
            self._error_logger(error="PyPDF2.errors.FileNotDecryptedError: File has not been decrypted", action="read_pdf", entity_type="file", entity_id=file.id)

        return docs

    def _load_docx_file(self, file) -> List[Document]:
        file_contents = file.get_contents(binary=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"

            with open(file_path, "wb") as binary_file:
                # Write bytes to file
                binary_file.write(file_contents)

            loader = Docx2txtLoader(file_path)
            docs = loader.load()

            for i, doc in enumerate(docs):
                docs[i].metadata["filename"] = file.filename
                docs[i].metadata["source"] = self._get_file_url(file.id)

        return docs

    def _load_excel_file(self, file) -> List[Document]:
        file_contents = file.get_contents(binary=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"

            with open(file_path, "wb") as binary_file:
                # Write bytes to file
                binary_file.write(file_contents)

            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()

            for i, doc in enumerate(docs):
                docs[i].metadata["filename"] = file.filename
                docs[i].metadata["source"] = self._get_file_url(file.id)

        return docs

    def _load_pptx_file(self, file) -> List[Document]:
        file_contents = file.get_contents(binary=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"

            with open(file_path, "wb") as binary_file:
                # Write bytes to file
                binary_file.write(file_contents)

            loader = UnstructuredPowerPointLoader(file_path)
            docs = loader.load()

            for i, doc in enumerate(docs):
                docs[i].metadata["filename"] = file.filename
                docs[i].metadata["source"] = self._get_file_url(file.id)

        return docs

    def _load_md_file(self, file) -> List[Document]:
        file_contents = file.get_contents(binary=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{file.filename}"

            with open(file_path, "wb") as binary_file:
                # Write bytes to file
                binary_file.write(file_contents)

            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()

            for i, doc in enumerate(docs):
                docs[i].metadata["filename"] = file.filename
                docs[i].metadata["source"] = self._get_file_url(file.id)

        return docs

    def _error_logger(self, error, action, entity_type, entity_id) -> None:
        if isinstance(error, str):
            self.errors.append({ "message": error, "action": action, "entity_type": entity_type, "entity_id": entity_id })
        elif isinstance(error.message, str):
            message_json = json.loads(error.message)
            self.errors.append({ "message": message_json["errors"][0]["message"], "action": action, "entity_type": entity_type, "entity_id": entity_id })
        else:
            self.errors.append({ "message": error.message[0]["message"], "action": action, "entity_type": entity_type, "entity_id": entity_id })

    def load_files(self, course) -> List[Document]:
        """Loads all files from a canvas course."""
        from canvasapi.exceptions import CanvasException, ResourceDoesNotExist

        file_documents = []

        try:
            files = course.get_files()

            for file in files:
                try:
                    if f"File:{file.id}" not in self.indexed_items:
                        file_documents = file_documents + self.load_file(file)
                        self.indexed_items.append(f"File:{file.id}")
                except ResourceDoesNotExist:
                    # This will happen when the file is part of a module that is hidden
                    file_content_type = getattr(file, "content-type")
                    self.invalid_files.append(f"{file.filename} ({file_content_type})")
        except CanvasException as error:
            self._error_logger(error=error, action="get_files", entity_type="course", entity_id=course.id)

        return file_documents

    def load_file(self, file) -> List[Document]:
        """Load a specific file."""
        file_documents = []

        file_content_type = getattr(file, "content-type")

        allowed_content_types = [
            "text/markdown", # md
            "text/html", # htm, html
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # docx
            "application/vnd.ms-excel", # xls
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", # xlsx
            "application/vnd.openxmlformats-officedocument.presentationml.presentation", # pptx
            "application/pdf", # pdf
            "text/rtf", # rtf
            "text/plain", # txt
        ]

        # print(f"New file:  {file.filename} ({file_content_type}) ({file.mime_class})")

        if file_content_type in allowed_content_types:
            # print(f"Processing {file.filename} {file.mime_class}")

            if file_content_type == "text/plain":
                file_documents = file_documents + self._load_text_file(file)

            if file_content_type == "text/html":
                file_documents = file_documents + self._load_html_file(file)

            elif file_content_type == "application/pdf":
                file_documents = file_documents + self._load_pdf_file(file)

            elif file_content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_documents = file_documents + self._load_docx_file(file)

            elif file_content_type in [ "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel" ]:
                file_documents = file_documents + self._load_excel_file(file)

            elif file_content_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
                file_documents = file_documents + self._load_pptx_file(file)

            elif file_content_type == "text/markdown":
                file_documents = file_documents + self._load_md_file(file)

            elif file_content_type == "text/rtf":
                file_documents = file_documents + self._load_rtf_file(file)
        else:
            self.invalid_files.append(f"{file.filename} ({file_content_type})")

        return file_documents

    def load_url(self, url) -> List[Document]:
        """Load a url."""
        loader = UnstructuredURLLoader(urls=[ url ])
        url_docs = loader.load()

        return url_docs

    def load_modules(self, course) -> List[Document]:
        """Loads all modules from a canvas course."""
        from canvasapi.exceptions import CanvasException, ResourceDoesNotExist

        module_documents = []

        try:
            modules = course.get_modules()

            for module in modules:
                locked = False

                if module.unlock_at:
                    unlock_at_datetime = datetime.strptime(module.unlock_at, '%Y-%m-%dT%H:%M:%SZ')
                    unlock_at_datetime = unlock_at_datetime.replace(tzinfo=pytz.UTC)
                    epoch_time = int(unlock_at_datetime.timestamp())
                    current_epoch_time = int(datetime.now().timestamp())

                    if current_epoch_time < epoch_time:
                        locked = True

                module_items = module.get_module_items(include=["content_details"])

                for module_item in module_items:
                    if module_item.type == "Page":
                        # print(f"  Indexing page {module_item.title} ({module_item.page_url})")

                        if f"Page:{module_item.page_url}" not in self.indexed_items:
                            if locked:
                                # Don't try indexing page
                                continue

                            try:
                                page = course.get_page(module_item.page_url)
                                module_documents = module_documents + self.load_page(page)
                                self.indexed_items.append(f"Page:{module_item.page_url}")
                            except CanvasException as error:
                                self._error_logger(error=error, action="get_page", entity_type="page", entity_id=module_item.page_url)
                    elif module_item.type == "Assignment":
                        # print(f"  Indexing assignment {module_item.title} ({module_item.content_id})")

                        if f"Assignment:{module_item.content_id}" not in self.indexed_items:
                            try:
                                assignment = course.get_assignment(module_item.content_id)
                                module_documents = module_documents + self.load_assignment(assignment)
                                self.indexed_items.append(f"Assignment:{module_item.content_id}")
                            except CanvasException as error:
                                self._error_logger(error=error, action="get_assignment", entity_type="assignment", entity_id=module_item.content_id)
                    elif module_item.type == "File":
                        # print(f"  Indexing file {module_item.title} ({module_item.content_id})")

                        if f"File:{module_item.content_id}" not in self.indexed_items:
                            try:
                                file = course.get_file(module_item.content_id)
                                module_documents = module_documents + self.load_file(file)
                                self.indexed_items.append(f"File:{module_item.content_id}")
                            except ResourceDoesNotExist:
                                # This will happen when the file is part of a module that is hidden
                                file_content_type = getattr(file, "content-type")
                                self.invalid_files.append(f"{file.filename} ({file_content_type})")
                            except CanvasException as error:
                                self._error_logger(error=error, action="get_file", entity_type="file", entity_id=module_item.content_id)
                    elif module_item.type == "ExternalUrl" and self.index_external_urls is True:
                        if locked:
                            # Don't try indexing external URL
                            continue

                        # print(f"  Indexing file {module_item.title} ({module_item.external_url})")

                        if f"ExternalUrl:{module_item.external_url}" not in self.indexed_items:
                            try:
                                module_documents = module_documents + self.load_url(url=module_item.external_url)
                                self.indexed_items.append(f"ExternalUrl:{module_item.external_url}")
                            except CanvasException as error:
                                self._error_logger(error=error, action="load_url", entity_type="externalurl", entity_id=module_item.external_url)
                    else:
                        print(f"  Module Item {module_item.title} is an unsupported type ({module_item.type})")

        except CanvasException as error:
            self._error_logger(error=error, action="get_modules", entity_type="course", entity_id=course.id)

        return module_documents

    def load(self) -> List[Document]:
        """Load documents."""

        docs = []

        try:
            # Import the Canvas class
            from canvasapi import Canvas
            from canvasapi.exceptions import CanvasException
        except ImportError as exc:
            raise ImportError(
                "Could not import canvasapi python package. "
                "Please install it with `pip install canvasapi`."
            ) from exc

        try:
            # Initialize a new Canvas object
            canvas = Canvas(self.api_url, self.api_key)

            course = canvas.get_course(self.course_id)

            # Access the course's name
            print(f"Indexing: {course.name}")
            print("")

            self.returned_course_id = course.id

            # Checking to see which tools are available?
            tabs = course.get_tabs()

            available_tabs = []

            for tab in tabs:
                available_tabs.append(tab.id)

            # load modules
            if "modules" in available_tabs:
                module_documents = self.load_modules(course=course)
                docs = docs + module_documents

            # Load pages
            if "page" in available_tabs:
                page_documents = self.load_pages(course=course)
                docs = docs + page_documents

            # load announcements
            if "announcements" in available_tabs:
                announcement_documents = self.load_announcements(canvas=canvas, course=course)
                docs = docs + announcement_documents

            # load assignments
            if "assignments" in available_tabs:
                assignment_documents = self.load_assignments(course=course)
                docs = docs + assignment_documents

            # load files
            if "files" in available_tabs:
                file_documents = self.load_files(course=course)
                docs = docs + file_documents

            # Replace null character with space
            for doc in docs:
                doc.page_content = doc.page_content.replace('\x00', ' ')

            return docs
        except CanvasException as error:
            self._error_logger(error=error, action="get_course", entity_type="course", entity_id=self.course_id)

        return docs
