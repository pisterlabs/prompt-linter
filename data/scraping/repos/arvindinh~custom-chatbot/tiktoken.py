from langchain.text_splitter import TokenTextSplitter

class Token_TextSplitter:
    """
    A class to split a Document using the TokenTextSplitter wrapper from the langchain library.
    """
    def __init__(self, chunk_size, chunk_overlap):
        """
        Initializes a new instance of TokenTextSplitter
        
        :param chunk_size: Maximum size of chunks to return
        :param chunk_overlap: Overlap in characters between chunks
        
        """
        self.splitter = TokenTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
    
    def split_data(self, data):
        """
        Splits the given Document based on tiktoken tokens. The text is split and chunk size is measured by tiktoken tokens.
        
        :param data: The Document to be split, in the Document format returned by the langchain pdf loaders

        :return: Split Documents
        """
        docs = self.splitter.split_documents(data)
        return docs
