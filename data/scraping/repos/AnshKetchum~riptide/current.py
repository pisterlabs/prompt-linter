from .openai import OpenAILLM
from .llama import LLaMALLM
import json 


class Provider:
    """
        A wrapper selector class to pick our desired LLM 
    """

    def __init__(self) -> None:

        with open('prefs.json', 'r') as f:
            dct = json.load(f)

            if dct['llm'] == OpenAILLM.SELECTOR:
                self.provider = OpenAILLM()
            
            elif dct['llm'] == LLaMALLM.SELECTOR:
                self.provider = LLaMALLM()

            else:
                raise ValueError("The llm you want hasn't been added yet")

    def get_llm_and_embeddings(self):
        return self.provider
    
provider = Provider().get_llm_and_embeddings()
            