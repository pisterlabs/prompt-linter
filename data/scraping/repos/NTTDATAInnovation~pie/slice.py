from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextSplitter:
    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=[". ", ".. ", "? "],
        **kwargs,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
            **kwargs,
        )

    def split_text(self, query, limit=500):
        """Splits a text into chunks"""
        return self._split_text(query) if len(query) > limit else [query]

    def _split_text(self, text) -> list:
        """Splits a text into chunks"""
        return self.splitter.split_text(text)
