from langchain.text_splitter import NLTKTextSplitter

class NLTK_TextSplitter:
    """
    A class to split a Document using the NLTKTextSplitter wrapper from the langchain library
    """
    def __init__(self, chunk_size):
        """
        Initializes a new instance of NLTKTextSplitter
        
        :param chunk_size: Maximum size of chunks to return
        
        """
        self.splitter = NLTKTextSplitter(chunk_size = chunk_size)
    
    def split_data(self, data):
        """
        Splits the given Document based on NLTK tokenzer, chunk size is measured by number of characters
        
        :param data: The Document to be split, in the Document format returned by the langchain pdf loaders

        :return: Split Documents
        """
        docs = self.splitter.split_documents(data)
        return docs
