from abc import abstractmethod
from haystack import Pipeline
from haystack.nodes import reader, retriever
from haystack.nodes.base import BaseComponent
from typing import Optional, List
from pathlib import Path
import pandas as pd

from OpenAIUtils import query_to_summaries as query_to_summaries_openai, file_to_embeddings as file_to_embeddings_openai, produce_prompt as produce_prompt_openai
from CohereUtils import query_to_summaries as query_to_summaries_cohere, file_to_embeddings as file_to_embeddings_cohere, produce_prompt as produce_prompt_cohere

class Embed(BaseComponent):
    def __init__(self, module_choice = "Cohere"):
        self.module_choice = module_choice

    outgoing_edges = 1

    def get_embeddings(self, queries: List[str], batch_size: Optional[int] = None, filenames: List[str] = [], use_cache: bool = False):
        for fname in filenames:
            if self.module_choice == "OpenAI":
                file_to_embeddings_openai(Path(fname), use_cache)
            elif self.module_choice == "Cohere":
                file_to_embeddings_cohere(Path(fname), use_cache)
            else:
                raise ValueError("Invalid module choice for Embed")
        return {}

    def run(self, query: str, filenames: List[str] = [], recalc_embeddings: bool = True):  # type: ignore
        output = {}
        if recalc_embeddings:
            output = self.get_embeddings([query], filenames=filenames)
        return output, "output_1"
    
    def run_batch(self, queries: List[str], batch_size: Optional[int] = None, filenames: List[str] = [], recalc_embeddings: bool = True):  # type: ignore
        output = {}
        if recalc_embeddings:
            output = self.get_embeddings(queries, batch_size, filenames)
        return output, "output_1"

class Complete(BaseComponent):
    def __init__(self, module_choice = "OpenAI"):
        self.module_choice = module_choice

    outgoing_edges = 1

    def get_summaries(self, queries: List[str], filenames: List[str], temperature: Optional[float] = 0.5, print_responses: bool = False):
        if self.module_choice == "OpenAI":
            output = {"completions": query_to_summaries_openai(filenames, queries, temperature, print_responses)}
        elif self.module_choice == "Cohere":
            output = {"completions": query_to_summaries_cohere(filenames, queries, temperature, print_responses)}
        else:
            raise ValueError("Invalid module choice for Complete")
        return output

    def run(self, query: str, filenames: List[str] = [], temperature: Optional[float] = 0.5):  # type: ignore
        output = self.get_summaries([query], filenames, temperature)
        return output, "output_1"
        
    def run_batch(self, queries: List[str], batch_size: Optional[int] = None, filenames: List[str] = [], temperature: Optional[float] = 0.5):  # type: ignore
        output = self.get_summaries(queries, filenames, temperature)
        return output, "output_1"

def run_qap(embeddings_choice, completion_choice, temperature, relevant_questions, filenames, recalc_embeddings):
    p = Pipeline()
    embed = Embed(embeddings_choice)
    complete = Complete(completion_choice)

    p.add_node(component=embed, name="Embed", inputs=["Query"])
    p.add_node(component=complete, name="Complete", inputs=["Embed"])

    res = p.run_batch(queries=relevant_questions, params={"filenames": filenames, "temperature": temperature, "recalc_embeddings": recalc_embeddings})
    completions = res["completions"]
    completions.to_pickle("./risks_responses.pkl")

    return completions