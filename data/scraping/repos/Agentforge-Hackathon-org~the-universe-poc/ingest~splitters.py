from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Splitters:
    def __init__(self):
        pass

    def character_text_splitter(self, chunk_size=200, chunk_overlap=20):
        """
        Create a new instance of `CharacterTextSplitter` with the given `chunk_size` and `chunk_overlap`.

        Args:
            chunk_size (int, optional): The number of characters in each chunk. Defaults to 200.
            chunk_overlap (int, optional): The number of characters to overlap between chunks. Defaults to 20.

        Returns:
            CharacterTextSplitter: A new instance of `CharacterTextSplitter`.
        """
        return CharacterTextSplitter(chunk_size, chunk_overlap)

    def recursive_text_splitter(self, chunk_size=200, chunk_overlap=20):
        """
        Split the text recursively into chunks of specified size with a specified overlap.

        Parameters:
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between adjacent chunks.

        Returns:
            RecursiveCharacterTextSplitter: An instance of RecursiveCharacterTextSplitter.
        """
        return RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)