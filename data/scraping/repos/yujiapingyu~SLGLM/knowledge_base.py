from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import TextSplitter
from typing import List, Any
from config import EMBEDDING_MODEL

class MyTextSplitter(TextSplitter):
    """Implementation of splitting text that looks at characters."""

    def __init__(self, separator: str = "\n\n", **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        if self._separator:
            splits = text.split(self._separator)
        else:
            splits = list(text)
        return splits

class KnowledgeBase(object):
    def __init__(self, embedding_model):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
    def load_docs(self, path, glob='**/*.txt'):
        loader = DirectoryLoader(path, glob=glob, show_progress=True, recursive=True)
        self.docs = loader.load_and_split(text_splitter=MyTextSplitter())
        print('load docs success')
    
    def get_index_from_doc(self):
        self.db = FAISS.from_documents(self.docs, self.embeddings)
        return self.db
    
    def save_index(self, dest, index_name):
        self.db.save_local(dest, index_name)
        
    def load_doc_and_save_index(self, path, dest, index_name):
        self.load_docs(path)
        self.get_index_from_doc()
        self.save_index(dest, index_name)
    
    def get_index_from_local(self, dest, index_name):
        self.db = FAISS.load_local(dest, self.embeddings, index_name)
        
    def similarity_search(self, query, k=3):
        result = self.db.similarity_search(query, k=k)
        return result
        
if __name__ == '__main__':
    knowledge_base = KnowledgeBase(EMBEDDING_MODEL)
    knowledge_base.load_doc_and_save_index('./data/test', './index', 'test')
    # knowledge_base.get_index_from_local('./index', 'medisian')
    print(knowledge_base.similarity_search('颈椎疼痛，手脚麻木怎么办'))