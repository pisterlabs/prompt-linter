from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TextSplitter,
    MarkdownHeaderTextSplitter,
    NLTKTextSplitter,
    PythonCodeTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TokenTextSplitter,
)
from enum import Enum
from vectordb.selector import BaseSelector
import loguru

logger = loguru.logger


class SplitterOptions(Enum):
    recursive_character_text_splitter = RecursiveCharacterTextSplitter
    character_text_splitter = CharacterTextSplitter
    text_splitter = TextSplitter
    markdown_header_text_splitter = MarkdownHeaderTextSplitter
    nltk_text_splitter = NLTKTextSplitter
    python_code_text_splitter = PythonCodeTextSplitter
    sentence_transformers_token_text_splitter = SentenceTransformersTokenTextSplitter
    token_text_splitter = TokenTextSplitter


class SplitterSelector(BaseSelector):
    def __init__(
        self, default_splitter=SplitterOptions.recursive_character_text_splitter
    ):
        super().__init__()
        self.recursive_character_text_splitter = (
            SplitterOptions.recursive_character_text_splitter
        )
        self.character_text_splitter = SplitterOptions.character_text_splitter
        self.text_splitter = SplitterOptions.text_splitter
        self.markdown_header_text_splitter = (
            SplitterOptions.markdown_header_text_splitter
        )
        self.nltk_text_splitter = SplitterOptions.nltk_text_splitter
        self.python_code_text_splitter = SplitterOptions.python_code_text_splitter
        self.sentence_transformers_token_text_splitter = (
            SplitterOptions.sentence_transformers_token_text_splitter
        )
        self.token_text_splitter = SplitterOptions.token_text_splitter
        self.splitter = default_splitter
        self.initialize_maps(list(SplitterOptions))

    def get_recursive_character_text_splitter(self):
        logger.info("Getting recursive_character_text_splitter")
        return self.select("recursive_character_text_splitter")

    def get_character_text_splitter(self):
        logger.info("Getting character_text_splitter")
        return self.select("character_text_splitter")

    def get_text_splitter(self):
        logger.info("Getting text_splitter")
        return self.select("text_splitter")

    def get_markdown_header_text_splitter(self):
        logger.info("Getting markdown_header_text_splitter")
        return self.select("markdown_header_text_splitter")

    def get_nltk_text_splitter(self):
        logger.info("Getting nltk_text_splitter")
        return self.select("nltk_text_splitter")

    def get_python_code_text_splitter(self):
        logger.info("Getting python_code_text_splitter")
        return self.select("python_code_text_splitter")

    def get_sentence_transformers_token_text_splitter(self):
        logger.info("Getting sentence_transformers_token_text_splitter")
        return self.select("sentence_transformers_token_text_splitter")
