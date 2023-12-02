from openai.error import RateLimitError, ServiceUnavailableError
import openai
import backoff
import pandas as pd
import numpy as np
import pickle
from transformers import GPT2TokenizerFast
from typing import List
from time import sleep
import pickle
import os


class Brain:
    COMPLETIONS_MODEL = "text-davinci-003"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    openai.api_key = os.getenv("OPENAI_KEY")
    @backoff.on_exception(backoff.expo, (RateLimitError, ServiceUnavailableError))
    def get_embedding(self, text: str, model: str=EMBEDDING_MODEL, idx: int=0) -> list[float]:
        result = openai.Embedding.create(
        model=model,
        input=text
        )
        return result["data"][0]["embedding"]

    def compute_doc_embeddings(self, df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
        
        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {
            idx: self.get_embedding(r.content) for idx, r in df.iterrows()
        }
        
    def compute_text_embeddings(self, text: str, start_index:int = 0) -> dict[tuple[str, str], list[float]]:
        return {
            (start_index+idx): self.get_embedding(line, self.EMBEDDING_MODEL ,idx) for idx, line in enumerate(text)
        }

    def update_text_embeddings(self, compute_embedding, text, new_text):
        print('Updating the model with data: ', new_text)
        compute_embedding.update(self.compute_text_embeddings(new_text, len(compute_embedding)))
        return text + new_text

    def load_embeddings(self, fname: str):
        """
        Read the document embeddings and their keys from a CSV.
        
        fname is the path to a CSV with exactly these named columns: 
            "title", "heading", "0", "1", ... up to the length of the embedding vectors.
        """
        
        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
        return {
            (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }
        
        
    text = []
    consumed_files = set()
    context_embeddings = {}
    def reload_models(self):
        self.text = pickle.load(open('textfile.obj', "rb"))
        self.consumed_files = pickle.load(open('consumed_files.obj', "rb"))
        self.context_embeddings = pickle.load(open('nyush_embeddings.obj', "rb"))
        
    def save_models(self):
        with open('nyush_embeddings.obj', 'wb') as fp:
            pickle.dump(self.context_embeddings, fp)
        with open('consumed_files.obj', 'wb') as fp:
            pickle.dump(self.consumed_files, fp)
        with open('textfile.obj', 'wb') as fp:
            pickle.dump(self.text, fp)
        
    def process_file(self, filename, delim="\n\n"):
        self.reload_models()
        if filename not in self.consumed_files:
            update = open (filename, "r").read().split(delim)
            self.text = self.update_text_embeddings(self.context_embeddings, self.text, update)
            self.consumed_files.add(filename)
            with open('nyush_embeddings.obj', 'wb') as fp:
            	pickle.dump(self.context_embeddings, fp)
            with open('consumed_files.obj', 'wb') as fp:
            	pickle.dump(self.consumed_files, fp)
            with open('textfile.obj', 'wb') as fp:
            	pickle.dump(self.text, fp)
            
        else:
            print("File already processed")
    # document_embeddings = self.load_embeddings("olympics_sections_document_embeddings.csv")

    # context_embeddings = self.compute_doc_embeddings(df)

    # context_embeddings = self.compute_text_embeddings(text)
    # with open('nyush_embeddings.obj', 'wb') as fp:
    # 	pickle.dump(context_embeddings, fp)
    # print(len(context_embeddings))

    # update = open ("update.txt", "r").read().split("\n\n")

    # text = self.update_text_embeddings(context_embeddings, text, update)


    def vector_similarity(self, x: List[float], y: List[float]) -> float:
        """
        We could use cosine similarity or dot product to calculate the similarity between vectors.
        In practice, we have found it makes little difference. 
        """
        return np.dot(np.array(x), np.array(y))


    def order_document_sections_by_query_similarity(self, query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_embedding(query)
        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
        
        return document_similarities

    
    
    MAX_SECTION_LEN = 500
    SEPARATOR = "\n\n* "

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    separator_len = len(tokenizer.tokenize(SEPARATOR))

    f"Context separator contains {separator_len} tokens"


    def construct_prompt_with_text(self,question: str, context_embeddings: dict, text: list, previous_context: str = None) -> str:
        """
        Fetch relevant 
        """
        if previous_context is not None:
            most_relevant_document_sections = self.order_document_sections_by_query_similarity(question + previous_context, context_embeddings)
        else:
            most_relevant_document_sections = self.order_document_sections_by_query_similarity(question, context_embeddings)
            
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        
        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = "\n".join(text[section_index-3:section_index+3])
            
            chosen_sections_len += len(document_section.split()) + self.separator_len
            if chosen_sections_len > self.MAX_SECTION_LEN:
                break
                
            chosen_sections.append(self.SEPARATOR + document_section)
            chosen_sections_indexes.append(str(section_index))
                
        # Useful diagnostic information
        print(f"Selected {len(chosen_sections)} document sections:","\t".join(chosen_sections_indexes))
        header = """Answer the questions as truthfully as possible using the provided context, and if the answer is not contained within the context, say "I don't know." Put ``` before and after the code.\n\nGeneralized Information:\n"""
        
        prompt = header + "".join(chosen_sections) 
        if previous_context is not None:
            prompt = prompt + "\n\n" + previous_context
        prompt = prompt +"\n " + question
        return prompt

    COMPLETIONS_API_PARAMS = {
        # We use temperature of 0.0 because it gives the most predictable, factual answer.
        "temperature": 1.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
    }

    def answer_query_with_context(
        self,
        query: str,
        df: pd.DataFrame,
        document_embeddings,
        previous_context = None,
        show_prompt: bool = True
    ) -> str:
        prompt = self.construct_prompt_with_text(
            query,
            document_embeddings,
            df,
            previous_context
        )
        
        if show_prompt:
            with open('temp.txt', 'w') as f:
                for line in prompt:
                    f.write(f"{line}")
                f.flush()

        response = openai.Completion.create(
                    prompt=prompt,
                    **self.COMPLETIONS_API_PARAMS
                )

        return response["choices"][0]["text"].strip(" \n")


    # answer= answer_query_with_context("How old was he when he won?",text, context_embeddings, previous_context)
    # print(answer)
import os
if __name__ == "__main__":
    b = Brain()
    b.reload_models()
    # directory = '../training_data/'
    # for filename in os.listdir(directory):
    #     if filename.endswith(".txt"):
            # b.process_file(directory+filename,'all right')
    # b.process_file('../training_data/overview.txt')
    # print([i for i, j in enumerate(b.text) if 'Junru He' in j])
    # b.save_models()
    