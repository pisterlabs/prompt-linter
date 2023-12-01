from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Splitters:
    """Splitters class provides text splitters."""

    def split_document(self, splitter_name, **kwargs):
        """Returns text splitter."""
        if splitter_name == "character":
            return CharacterTextSplitter(**kwargs)
        if splitter_name == "recursive":
            return RecursiveCharacterTextSplitter(**kwargs)
