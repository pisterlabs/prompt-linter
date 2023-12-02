
from src.tubegpt.tubegpt import TubeGPT
from langchain.embeddings import HuggingFaceEmbeddings
import unittest


class TubeGPTTest(unittest.TestCase):

    def simpleuadio():
        urls = ["https://www.youtube.com/watch?v=_xASV0YmROc"]
        save_dir = "/Users/hbolak650/Downloads/YouTubeYoga"
        MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        tubegpt = TubeGPT(urls,save_dir)
        tubegpt.process_audio(urls,save_dir,embeddings=embeddings)
        

    def simplevision():
        pass

    def simpleaudiovision():
        pass

if __name__ == '__main__':
    unittest.main()