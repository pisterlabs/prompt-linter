
from typing import Dict, List
from types import SimpleNamespace
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain.prompts.example_selector.base import BaseExampleSelector

# qdrant = QdrantClient(path="db")
semantic_model = SentenceTransformer('thenlper/gte-large')

class HomerExampleSelector(BaseExampleSelector):
    """Select examples for Homer Simpson."""
    def add_example(self, example: Dict[str, str]) -> None:
        pass

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:

        """Select which examples to use based on the inputs."""
        examples = []
        qdrant = QdrantClient(path="db")

        prompts  = qdrant.search(
          collection_name="prompts",
          query_vector=semantic_model.encode(input_variables.get("human_input")).tolist(),
          limit=10,
          with_payload={"exclude": ["precontext", "postcontext", "prompting_character"]},
          score_threshold=0.75
        )

        for prompt in prompts:
          payload = SimpleNamespace(**prompt.payload)

          examples.append(prompt.payload)

          responses = qdrant.search(
            collection_name="responses",
            query_vector=semantic_model.encode(payload.response).tolist(),
            limit=3,
            with_payload={"exclude": ["precontext", "postcontext", "prompting_character"]},
            score_threshold=0.75
          )

          for response in responses:
              examples.append(response.payload)
  
        return examples
