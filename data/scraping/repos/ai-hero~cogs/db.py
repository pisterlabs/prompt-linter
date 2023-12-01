import os
import chromadb 
import json
import openai
from typing import Any, Dict, List, Optional
openai.api_key = os.environ.get("OPENAI_API_KEY")
import skills.skills as skills_catalog



class SkillsDB:
    """A vector database for skills."""

    db_version = "01"  # Version of the skills database

    def __init__(self) -> None:
        """Initialize the database."""
        self.db_name = "chroma.db"
        self.collection_name = f"chats-{SkillsDB.db_version}"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, ".db")
        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(os.path.join(db_path, "chroma.db"))
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except ValueError:
            # Create the collection if it doesn't exist
            self.collection = self.client.create_collection(self.collection_name)

        metadata = []
        with open(os.path.join(base_dir, "skills.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.add(metadata)
            
    def _get_embedding(self, text: str, embeddings_cache: Optional[Dict[str, List[float]]] = None) -> Any:
        """Use the same embedding generator as what was used on the data!!!"""
        if embeddings_cache and text in embeddings_cache:
            return embeddings_cache[text]
        if len(text) / 4 > 8000:  # Hack to be under the 8k limit, one token ~= 4 characters
            text = text[:8000]
        response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return response.data[0].embedding

    def search(self, text: str) -> List[Dict]:
        """Search the index for the top 3 results."""
        embedding = self._get_embedding(text)
        results = self.collection.query(query_embeddings=[embedding], n_results=3)
        metadatas:list = []
        for m in results["metadatas"][0]:
            metadatas.append(json.loads(m["metadata"]))
        return metadatas

    def add(self, functions: List[Dict]) -> None:
        """Add the embeddings, metadata, and ids to the database."""

        embeddings_list:list = []
        metadata_list:list = []
        id_list:list = []

        for f in functions:
            metadata = {"metadata": json.dumps(f)}
            metadata_list.append(metadata)
            embeddings_list.append(self._get_embedding(f["description"]))
            id_list.append(f"{f.get('module', '')}{f['name']}")

        self.collection.add(embeddings=embeddings_list, metadatas=metadata_list, ids=id_list)

    def execute_command(self, function_name, args_dict):
        # Check if the function exists in the skills module
        if hasattr(skills_catalog, function_name):
            # Get the function by name
            function_to_call = getattr(skills_catalog, function_name)
            # Call the function with the provided arguments
            return function_to_call(**args_dict)
        else:
            raise Exception(f"No function named {function_name} found in skills module.")
