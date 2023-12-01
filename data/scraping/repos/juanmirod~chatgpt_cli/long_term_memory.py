import openai
import os
from dotenv import load_dotenv
import json
import numpy as np
from rich.prompt import Prompt
from annoy import AnnoyIndex

load_dotenv()
openai.api_key = os.environ.get('API_KEY')


class LongTermMemory:
    index = None
    embeddings = None

    def load(self):
        # Load the embeddings from the file
        with open("db/embeddings_db.json", "r") as f:
            self.embeddings = json.load(f)

        # Build the Annoy index
        self.index = AnnoyIndex(len(self.embeddings[0]["embedding"]), metric="angular")
        for i, item in enumerate(self.embeddings):
            self.index.add_item(i, item["embedding"])
        self.index.build(50)  # 50 trees

    def recover_memories_about(self, query):
        """Recover the more similar entries to the query."""
        response = openai.Embedding.create(
            engine="text-embedding-ada-002",
            input=query
        )
        query_embedding = response["data"][0]["embedding"]

        # Search for the most similar embeddings and return them
        similarities = self.index.get_nns_by_vector(query_embedding, 3, include_distances=True)
        results = [self.embeddings[similarities[0][i]]["text"] for i in range(3)]
        return '\n'.join(results)
