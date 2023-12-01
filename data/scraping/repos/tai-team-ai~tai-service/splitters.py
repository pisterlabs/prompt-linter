"""Define custom splitters used for splitting documents."""
import copy
from typing import Optional, List, Any
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from loguru import logger
from ..shared_schemas import ChunkSize
from .data_ingestor_schema import InputFormat, IngestedDocument


class YouTubeTranscriptSplitter(RecursiveCharacterTextSplitter):
    """
    Implement a splitter for YouTube transcripts.

    We have chosen NOT to use the vanilla RecursiveCharacterTextSplitter
    as it doesn't handle timestamps. However, to keep this decoupled from
    the loader, we fall back to the RecursiveCharacterTextSplitter if the 
    appropriate metadata is not available in the document.

    NOTE: If the incoming document split sizes are already larger than
    the specified chunk size, then the document will not be split and a 
    warning will be logged.
    """
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(separators=separators, keep_separator=keep_separator, **kwargs)

    def _add_chunk_overlap_to_previous_document(self, documents: List[Document], overlap_buffer: str, overlap_duration: float) -> None:
        if documents:
            previous_duration = documents[-1].metadata["duration"]
            documents[-1].metadata.update({"duration": previous_duration + overlap_duration})
            documents[-1].page_content += overlap_buffer

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create a list of documents from the texts."""
        if not metadatas:
            metadatas = [{} for _ in range(len(texts))]
        if not all("start" in metadata and "duration" in metadata for metadata in metadatas):
            return super().create_documents(texts=texts, metadatas=metadatas)

        documents: list[Document] = []
        aggregate_content = ""
        aggregate_duration = 0
        aggregate_start = metadatas[0]["start"]
        overlap_buffer = ""
        overlap_duration = 0
        timestamp_overlap = 10

        for i, text in enumerate(texts):
            if self._length_function(text) > self._chunk_size:
                logger.warning(
                    f"Text length ({self._length_function(text)}) is greater than the target chunk size ({self._chunk_size})."
                )
            metadata = copy.deepcopy(metadatas[i])
            len_with_new_chunk = self._length_function(aggregate_content + text)
            if len_with_new_chunk <= self._chunk_overlap:
                overlap_buffer += " " + text
                overlap_duration += metadata["duration"]
            elif overlap_buffer:
                # update previous document if it exists
                self._add_chunk_overlap_to_previous_document(documents, overlap_buffer, overlap_duration)
                overlap_buffer = ""
                overlap_duration = 0

            if len_with_new_chunk <= self._chunk_size:
                aggregate_content += " " + text
                aggregate_duration += metadata["duration"]
            else:
                metadata.update({"start": aggregate_start - timestamp_overlap, "duration": aggregate_duration})
                page_content = aggregate_content if self._keep_separator else aggregate_content.strip()
                documents.append(Document(page_content=page_content, metadata=metadata))
                aggregate_content = text
                aggregate_start = metadatas[i]["start"]
                aggregate_duration = metadatas[i]["duration"]

        # Don't forget to add the last aggregate if it's non-empty
        if aggregate_content:
            self._add_chunk_overlap_to_previous_document(documents, overlap_buffer, overlap_duration)
            metadata = copy.deepcopy(metadatas[-1])
            metadata.update({"start": aggregate_start, "duration": aggregate_duration})
            documents.append(Document(page_content=aggregate_content, metadata=metadata))

        return documents


SPLITTER_STRATEGY_MAPPING: dict[str, Any] = {
    InputFormat.YOUTUBE_VIDEO: YouTubeTranscriptSplitter,
}
INPUT_TO_LANGUAGE_MAPPING = {
    InputFormat.PDF: Language.MARKDOWN,
    InputFormat.GENERIC_TEXT: Language.MARKDOWN,
    InputFormat.LATEX: Language.LATEX,
    InputFormat.MARKDOWN: Language.MARKDOWN,
    InputFormat.HTML: Language.HTML,
    InputFormat.WEB_PAGE: Language.HTML,
}
TOTAL_PAGE_COUNT_STRINGS = [
    "total_pages",
    "total_page_count",
    "total_page_counts",
    "page_count",
]
PAGE_NUMBER_STRINGS = ["page_number", "page_numbers", "page_num", "page_nums", "page"]


CHUNK_SIZE_TO_CHAR_COUNT_MAPPING = {
    ChunkSize.SMALL: 500,
    ChunkSize.LARGE: 2000,
}
OVERLAP_SIZE_TO_CHAR_COUNT_MAPPING = {
    ChunkSize.SMALL: 150,
    ChunkSize.LARGE: 350,
}


def get_total_page_count(docs: list[Document]) -> Optional[int]:
    """Get the page count and total page count."""
    for doc in docs:
        for key in TOTAL_PAGE_COUNT_STRINGS:
            if key in doc.metadata:
                return doc.metadata[key]


def get_page_number(doc: Document) -> Optional[int]:
    """Get the page number."""
    for key in PAGE_NUMBER_STRINGS:
        if key in doc.metadata:
            return doc.metadata[key]


def document_splitter_factory(ingested_document: IngestedDocument, chunk_size: ChunkSize) -> IngestedDocument:
    """Return a copy of the ingested document with the appropriate splitter."""
    input_language = INPUT_TO_LANGUAGE_MAPPING.get(ingested_document.input_format)
    kwargs = {
        "chunk_size": CHUNK_SIZE_TO_CHAR_COUNT_MAPPING[chunk_size],
        "chunk_overlap": OVERLAP_SIZE_TO_CHAR_COUNT_MAPPING[chunk_size],
    }
    if input_language:
        # currently only html is converted to generic text (this isn't great to know that here as 
        # it strongly couples us to know that we are using the BS4 loader)
        # we make the chunks smaller to allow for highlighting text in the webpage
        if ingested_document.input_format == InputFormat.GENERIC_TEXT and chunk_size == ChunkSize.SMALL:
            kwargs["chunk_overlap"] = 50
            kwargs["chunk_size"] = 200
        splitter = RecursiveCharacterTextSplitter.from_language(language=input_language, **kwargs)
    else:
        Splitter = SPLITTER_STRATEGY_MAPPING.get(ingested_document.input_format)
        if not Splitter:
            raise NotImplementedError(f"Splitting strategy for {ingested_document.input_format} not implemented.")
        elif Splitter == YouTubeTranscriptSplitter:
            kwargs["chunk_overlap"] = int(kwargs["chunk_overlap"] * 1.2)
            kwargs["chunk_size"] = int(kwargs["chunk_size"] * 1.2)
        splitter = Splitter(**kwargs)
    copy_of_ingested_document = ingested_document.copy()
    copy_of_ingested_document.splitter = splitter
    return copy_of_ingested_document
