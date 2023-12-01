import faiss
import openai
from llama_index.readers.file.epub_parser import EpubParser
# create an index with the text and save it to disk in data/indexes
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor
from langchain.chat_models import ChatOpenAI
from llama_index import GPTTreeIndex
import os
from llama_index import SummaryPrompt, QuestionAnswerPrompt
# set environment variable with OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = "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"

class Sage:
    def __init__(self, model_name: str = "gpt-3.5-turbo", history = None):
        """
        Initializes the Sage class with the given API key.
        """
        self.model_name = model_name
        self._index=None
        self._docs = None
        self.response = None
        self.load_model()
        

    def load_book(self, book_file_path_list: list = [""], book_dir_path: str = "") -> None:
        """
        Loads the book document from the given file path and create index.
        """
        self._docs = SimpleDirectoryReader(input_dir = book_dir_path, input_files = book_file_path_list).load_data()
        self._index = GPTSimpleVectorIndex(documents=self._docs)
        
    def load_model(self) -> None:
        """
        Load the Open AI Model, book and index embeddings
        """
        self.llm_predictor = LLMPredictor(llm=ChatOpenAI(model_name=self.model_name))

    def run(self, query: str) -> str:
        """
        Generate response.
        """
        self.response = self._index.query(query,llm_predictor=self.llm_predictor,
        similarity_top_k=3)
        return f"<b>{self.response}</b>"

if __name__ == "__main__":
  book_talker = Sage(model_name = "gpt-3.5-turbo")
  book_talker.load_book(book_file_path_list = ["test_data/epubs/SeeingLikeAState/SeeingLikeAState.epub"])
  print(book_talker.run('Summarize the book'))
