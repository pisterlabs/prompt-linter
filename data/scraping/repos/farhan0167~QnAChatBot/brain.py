import openai
import pandas as pd
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from typing import List

class Brain:
    DOC_EMBEDDINGS_MODEL = "text-embedding-ada-002"
    QUERY_EMBEDDINGS_MODEL = "text-embedding-ada-002"

    def get_embedding(self, text: str, model: str) -> List[float]:
        result = openai.Embedding.create(
        model=model,
        input=text)
        return result["data"][0]["embedding"]

    def get_doc_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text, self.DOC_EMBEDDINGS_MODEL)

    def get_query_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text, self.QUERY_EMBEDDINGS_MODEL)

    def compute_doc_embeddings(self, df: pd.DataFrame):
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {
            idx: self.get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
        }
    def vector_similarity(self, x: List[float], y: List[float]) -> float:
        """
        We could use cosine similarity or dot product to calculate the similarity between vectors.
        In practice, we have found it makes little difference.
        """
        return np.dot(np.array(x), np.array(y))
    def order_document_sections_by_query_similarity(self, query: str, contexts):
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections.

        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_query_embedding(query)

        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)

        return document_similarities
    
    MAX_SECTION_LEN = 1000
    SEPARATOR = "\n* "

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
        """
        Fetch relevant
        """
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(question, context_embeddings)

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.
            document_section = df.loc[section_index]

            chosen_sections_len += document_section.tokens + self.separator_len
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break

            chosen_sections.append(self.SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:")
        print("\n".join(chosen_sections_indexes))

        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"