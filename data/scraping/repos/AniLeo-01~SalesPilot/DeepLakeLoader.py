import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

class DeepLakeLoader:
    def __init__(self, source_data_path, token_id):
        self.source_data_path = source_data_path
        self.activeloop_token_id = token_id
        self.file_name = os.path.basename(source_data_path).split('.')[0]
        self.data = self.split_data()
        if self.check_if_db_exists():
            self.db = self.load_db()
        else:
            self.db = self.create_db()

    def split_data(self):
        with open(f'{self.source_data_path}', 'r') as f:
            text = f.read()
        # Split the text into a list using the keyword "Objection: "
        objections_list = text.split("Objection: ")[1:]  # We ignore the first split as it is empty

        # Now, prepend "Objection: " to each item as splitting removed it
        objections_list = ["Objection: " + objection for objection in objections_list]
        return objections_list 
    
    def load_db(self):
        return DeepLake(dataset_path=f'dataset/{self.file_name}', embedding_function=OpenAIEmbeddings(), read_only=True, token = self.activeloop_token_id)
    
    def create_db(self):
        return DeepLake.from_texts(self.data, OpenAIEmbeddings(), dataset_path=f'dataset/{self.file_name}', token = self.activeloop_token_id)
    
    def check_if_db_exists(self):
        """
        Check if the database already exists.

        Returns:
            bool: True if the database exists, False otherwise.
        """
        return os.path.exists(f'dataset/{self.file_name}')

    def query_db(self, query):
        if query:
            results = self.db.similarity_search(query, k=3)
            content = []
            for result in results:
                content.append(result.page_content)
            return content
        else:
            return None
    
