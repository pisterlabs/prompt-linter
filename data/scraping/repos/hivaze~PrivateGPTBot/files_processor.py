import logging
from typing import List

from aiogram.types import User
from langchain.document_loaders import *
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app import settings
from app.bot import long_context_model
from app.internals.chat.chat_history import ChatHistory, ChatMessage, ChatRole
from app.internals.chat.chat_models import TextGenerationResult
from app.utils.misc import clean_text

EXISTING_DOCUMENT_FORMAT = "- Document ID: {doc_id}. File name: '{file_name}'. Content summary: '{content_summary}'."

SUMMARIZE_PROMPT = "You are given an excerpt of a document, your task is to determine what kind of document it is, topic the text is on, highlight the key words and make a brief summary of it. Answer in one text paragraph. The text is presented below:\n{content}"

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1_500, chunk_overlap=50)
SUMMARY_DOCS = settings.config.documents.summary_blocks


def make_summary(splits: List[Document], tg_user: User) -> str:
    document_content = "\n".join([split.page_content for split in splits[:SUMMARY_DOCS]])
    prompt = SUMMARIZE_PROMPT.format(content=document_content)
    summary_hist = ChatHistory()  # No system prompt here
    summary_hist.add_message(ChatMessage(role=ChatRole.USER, text=prompt))
    result: TextGenerationResult = long_context_model.generate_answer(summary_hist)
    logger.info(f"New document summary generated for user '{tg_user.username}' | '{tg_user.id}'. "
                f"Prompt tokens: {result.prompt_tokens_usage}, "
                f"Completion tokens: {result.completion_tokens_usage}, "
                f"Time taken: {result.time_taken}")
    summary_result = result.message.text.strip()
    summary_result = summary_result.replace("\n", " ")
    return summary_result


def check_if_extension_supported(file_path: str):
    ext = "." + file_path.rsplit(".", 1)[-1]
    return ext in LOADER_MAPPING


def load_single_document(file_path: str, file_name: str, file_id: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]

    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    documents = loader.load()

    splits: List[Document] = text_splitter.split_documents(documents)
    for split in splits:
        split.page_content = clean_text(split.page_content)
        split.metadata['file_name'] = file_name
        split.metadata['file_id'] = file_id

    return splits


def build_document_info(file_id: str, file_name: str, file_summary: str) -> str:
    return EXISTING_DOCUMENT_FORMAT.format(doc_id=file_id, file_name=file_name, content_summary=file_summary)
