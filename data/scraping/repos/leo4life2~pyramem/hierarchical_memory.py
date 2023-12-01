import openai
from langchain.text_splitter import CharacterTextSplitter
from pyramem.datastore import *
from pyramem.api.embeddings import Embeddings

class HierarchicalMemory:
    def __init__(self, namespace="default", datastore="chroma", separator="\n\n", chunk_size=200, chunk_overlap=20, embedding_model="text-embedding-ada-002"):
        if datastore == "chroma":
            self.datastore = ChromaDatastore(namespace=namespace)
        elif datastore == "pinecone":
            self.datastore = PineconeDatastore(namespace=namespace)
        else:
            raise Exception("Datastore not recognized.")
        
        self.namespace = namespace
        self.embedding_model = Embeddings(model=embedding_model)
        self.text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def add_memory(self, text):
        """Add an L1 memory chunk to the vectorstore.

        Args:
            text (str): a piece of text to save to memory
        """
        
        # Chunking
        texts = self.text_splitter.split_text(text)
        
        # Vectorize
        embeddings = self.embedding_model.get_embedding(texts)
        print(embeddings)
        
        # Save to vectorstore (use namespace)
        
        
        # Check if the new L1 caused there to be 5 consecutive L1s, if yes, create a new L2, then repeat for L2, L3, etc.
        
        
    
    def get_all_memories(self):
        """Get all memories from the vectorstore.
        """
        return self.datastore.query([0]*1536, top_k=1000)
        
    def get_top_memories(self, n=2):
        """Get all top N level memories from the vectorstore.
        For instance, if the current highest level of memory you have in the vectorstore is L3, and n=2, this will return L3 and L2 memories.

        Args:
            n (int, optional): Number of levels of memories to return. Defaults to 2.
        """
        pass