'''
This file contains the logic for storing and querying data from Pinecone.
'''
from typing import List
from langchain.vectorstores import Pinecone
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import GooglePalm
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.schema import Document
import pinecone
from pinecone import DescribeIndexStatsResponse

class PineconeTrainer:
    def __init__(self, gcp_api_key: str, pinecone_api_key: str, pinecone_environment: str):
        self.gcp_api_key = gcp_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.palm_config = {
            'temperature': 0.7,
            'google_api_key': self.gcp_api_key,
        }
        self.index_name = 'paragraph-summarizer'
        self.llm = GooglePalm(**self.palm_config)
        self.chain = load_summarize_chain(self.llm, chain_type='stuff')
        self.embeddings = GooglePalmEmbeddings(**self.palm_config)

        self.pinecone_init(self.index_name, 'cosine', 768)


    def pinecone_init(self, index_name: str, metric: str, dimension: int) -> None:
        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment,
        )

        # check if index exists
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, metric=metric, dimension=dimension)

        self.index = pinecone.Index(index_name=index_name)
        self.vectordb = Pinecone(index=self.index, embedding_function=self.embeddings.embed_query, text_key='text')

    def add_data(self, input_list: List[str]=[]) -> None:
        document_list = [Document(page_content=input_list[i]) for i in range(len(input_list))]
        self.vectordb = Pinecone.from_documents(document_list, embedding=self.embeddings, index_name=self.index_name)

        print('Data added successfully!, %s vectors added' % len(input_list))

    def delete_all_data(self) -> None:
        pass

    def query(self, query: str=' ', question: str='Summarize in 3 sentences') -> str:
        search = self.vectordb.similarity_search(query=query, k=3)
        summary = self.chain.run(input_documents=search, question=question)
        return summary

    def get_index_info(self) -> DescribeIndexStatsResponse:
        index = pinecone.GRPCIndex(self.index_name)
        output = index.describe_index_stats()

        return output

    def embed_text(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)
    
    def pinecone_train(self, input_file: str) -> None:
        try:
            input_list = self.extract_input_text(input_file)
            self.add_data(input_list)
        except Exception as e:
            print(e)

    @staticmethod
    def extract_input_text(input_file: str) -> List[str]:
        from logic.data_extract import extract_data, extract_text
        data = extract_data(input_file)
        texts = extract_text(data)
        return texts
    
    @staticmethod
    def extract_output_text(input_file: str) -> List[str]:
        from logic.data_extract import extract_data, extract_output_text
        data = extract_data(input_file)
        texts = extract_output_text(data)
        return texts

if __name__ == '__main__':
    pass