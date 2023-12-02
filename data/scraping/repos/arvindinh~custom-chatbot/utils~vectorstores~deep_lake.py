from langchain.vectorstores import DeepLake

class DeeplakeDB:
    """
    A class to initialize the Deep Lake vector store and perform various operations based on the DeepLake wrapper from langchain
    """
    def __init__(self, store_path, embedding_model):
        """
        Initializes the DeepLake object based on a given dataset path and embedding function/model.
        DeepLake wrapper is capable of internally computing the embedding using the given model and storing it in the path.
        
        :param store_path: path that contains vector store. will create at that path if doesn't already exist 
        :param embedding_model: langchain embedding model
        """
        self.db = DeepLake(dataset_path = store_path, embedding_function = embedding_model)

    def add_docs(self, documents):
        """
        Adds the embedded documents to the path given on initialization. returns the id, accessible if needed.
        
        :param document: langchain Document object used for computing embedding, then to be stored
        """
        for document in documents:
            id = self.db.add_documents(document)
    
    def find_similar(self, query):
        """
        Returns the document that best matches the query
        
        :param query: String that is tested for similarity search
        
        :return: most similar Document object
        """
        return self.db.similarity_search(query)

    def delete_all(self):
        """
        Deletes the vector store in the given path.
        """
        self.db.delete_dataset()
