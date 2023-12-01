import numpy as np
import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter

from utils import DataLoader, Configs

class DocSplitEstimator(DataLoader, Configs):

    def __init__(self):
        super().__init__()
        self.openai_embeddings = OpenAIEmbeddings(
            openai_api_key=self.OPENAI_API_KEY,
            model=self.OPENAI_EMBEDDING_MODEL
        )
        self.documents = self.split_text()
        self.embeddings = None
        self.mid = None

    def split_text(self):
        data = self.load_csv().load()
        text_splitter = TokenTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
        )
        return text_splitter.split_documents(data)
    
    def estimate_n_docs(self):
        lowest = self.TOKEN_RANGE_BOTTOM
        highest = self.TOKEN_RANGE_TOP
        encoding = tiktoken.get_encoding(self.ENCODING_NAME)
        totals = []         
        n_documents = len(self.documents)
        left, right = 0, n_documents - 1
        while left <= right:
            counts = 0
            self.mid = left + (right - left) // 2
            for doc in self.documents[0:self.mid]:
                counts += len(encoding.encode(doc.page_content))
            print(f"Estimated token counts: {counts} based on {self.mid} data points.")
            if counts > lowest and counts < highest:
                m = n_documents // self.mid
                for i in range(m):
                    cut_off_count = 0
                    start = i * self.mid
                    end = (i + 1) * self.mid
                    for doc in self.documents[start:end]:
                        cut_off_count += len(encoding.encode(doc.page_content))
                    totals.append(cut_off_count)
                totals = np.asarray(totals)
                print(f"The ideal cut-off number is {self.mid}.")
                print(f"The maxium number of tokens of each cut-off is {totals.max()}.")
                print(f"The minimum number of tokens of each cut-off is {totals.min()}.")
                print(f"The total number of tokens of each cut-off is {totals.sum()}.")
                return totals.max()
            elif counts < lowest:
                left = self.mid + 1
            else:
                right = self.mid - 1
        return -1
    

if __name__ == "__main__":
    dse = DocSplitEstimator()
    dse.estimate_n_docs()