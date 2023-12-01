from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
from os.path import dirname, join 
from dotenv import load_dotenv
import numpy as np

#.env adjustments
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
PINECONE_API = os.environ.get("PINECONE_API")
OPENAI_API = os.environ.get("OPENAI_API")


#the database class
class Embedder:
    def __init__(self):
        self.embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    def get_embedding(self, text:str):
        embedding = self.embed_model.embed_query(text) 
          
        return embedding
    
        
def cossimhist(vec1, vec_dict:dict, thresh=0.5):
    """
    A function to calculate cosine similarity between a vector and a list of vectors.
    Designed to take a vector and a list of vectors. 
    Returns the n most similar vectors to the input one + answers to them. 
    """
    n_prompts_answers = {}

    for key in vec_dict.keys(): 
        cos_sim = np.dot(vec1, vec_dict[key][0])/(np.linalg.norm(vec1)*np.linalg.norm(vec_dict[key][0]))
        if cos_sim >= thresh:
            n_prompts_answers[key] = vec_dict[key][1]

    return n_prompts_answers

def cossim(vec1, vec2, thresh=0.5):
    result = []
    cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    if cos_sim >= thresh:
        result.append(vec2)
    
    return result

#a function to retreive all the previous prompts from some history 
def retreive_hist(inp:dict):
    """
    A function to retreive the entire chat history from an input dictionary.
    Specific to the pipeline of the project.
    """
    vec_dict = {}
    history = inp['history'] #a list of all prompt-answer pairs

    for pair in history:
        vec_dict[pair['prompt']] = [pair['vectorized_prompt'], pair['answer']]

    return vec_dict