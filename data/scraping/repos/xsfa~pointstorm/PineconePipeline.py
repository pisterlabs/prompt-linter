import pinecone
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from pointstorm.embedding.text import Document
from tqdm.auto import tqdm
from uuid import uuid4
from typing import Union, List

load_dotenv()

class PineconePipeline:
    """
    Pipeline for interacting with the Pinecone database.
    
    Provides utilities for setting data, initializing content, and upserting data to the Pinecone database.
    
    Attributes:
    pinecone_api_key (str): Pinecone API key.
    pinecone_environment (str): Environment for Pinecone (e.g., production, staging).
    documents (Union[Document, List[Document], str]): Documents to be processed or stored.
    """

    openai_api_key: str
    pinecone_api_key: str
    pinecone_environment: str
    documents: Union[Document, List[Document], str]

    def __init__(self, openai_api_key:str = os.getenv('OPENAI_API_KEY')) -> None:
        """
        Initialize the PineconePipeline with API key and environment.
        
        Parameters:
        api_key (str): Pinecone API key.
        environment (str): Environment for Pinecone (e.g., production, staging).
        """
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENV')
        pinecone.init(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment,
        )

    def set_data(self, input_data: Union[Document, List[Document], str]) -> None:
        """
        Set the data for the pipeline.
        
        The input data can be a single Document, a list of Document objects, or a raw string.
        
        Parameters:
        input_data (Union[Document, List[Document], str]): Data to be set.
        
        Raises:
        ValueError: If input data is not of the correct type.
        """
        if isinstance(input_data, Document):
            self.documents = [input_data]
        elif isinstance(input_data, list) and all(isinstance(doc, Document) for doc in input_data):
            self.documents = input_data
        elif isinstance(input_data, str):
            self.documents = [Document(id=str(uuid4()), text=[input_data])]
        else:
            raise ValueError("Input data should be a Document object, a list of Document objects, or a raw string.")

    def initialize_contents(self, txt_path):
        """
        Initializes contents from a given text file.
        
        Reads a text file line by line, converting each line into a Document object with a unique ID and group key.
        
        Parameters:
        txt_path (str): Path to the text file to read.
        
        Returns:
        List[Document]: List of initialized Document objects.
        """
        contents = []
        with open(txt_path, 'r') as file:
            for line in file:
                doc = Document(
                    id=str(uuid4()),
                    group_key="test-doc", # verify requirements in 
                    text=[line],
                    embeddings=[]
                )
                doc = self.embedding(document=doc)
                contents.append(doc)
        return contents


    def upsert(self, index_name, txt_path):
        """
        Upsert content to the specified Pinecone index.

        If the index does not exist, it will create a new one with specified dimensions and metric.
        Then, it adds the content to the index in batches.

        Parameters:
        index_name (str): Name of the Pinecone index to upsert to.
        txt_path (str): Path to the text file containing content to upsert.
        """
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=384, metric='cosine')

        index = pinecone.Index(index_name)
        contents = self.initialize_contents(txt_path)

        batch_size = 1
        for i in tqdm(range(0, len(contents), batch_size)):
            ids = contents[i].id
            embeddings = contents[i].embeddings
            data = [{
                'group_key': contents[i].group_key,
                'file_content': contents[i].text
            }]
            to_upsert = list(zip(ids, embeddings, data))
            index.upsert(vectors=to_upsert)