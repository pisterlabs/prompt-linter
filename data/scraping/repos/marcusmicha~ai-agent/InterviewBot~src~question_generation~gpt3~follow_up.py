from abc import ABC, abstractmethod
import os
from typing import List
import openai
from src.memory.agent_memory import AgentMemory
import pandas as pd

class TextGenerator(ABC):
    @abstractmethod
    def generate_text(prompt:str)->str:
        pass

class IntentQuestionGenerator(object):
    def __init__(self, **kwargs) -> None:
        self.init_model()
        self.params = {
            'engine': "text-davinci-001",
            'max_tokens': 64,
            'n': 1,
            'stop': '?'
        }
        
    def init_model(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def generate_question(self, intent: str, **kwargs) -> str:

        context = self.get_context_from_memory(**kwargs)
        intent_natural_text = intent.replace('/', ' and ')

        if context is None:
                prompt = f"""
                    Generate an interview question regarding {intent_natural_text}.
                    Question:"""  
        else:
            prompt = f"""Question: {context['question']}
                         Answer: {context['answer']}
                         Generate next question regarding {intent_natural_text}.
                         Question:"""

        api_params = {**self.params, 'prompt': prompt}
        response = openai.Completion.create(**api_params)
        response_texts = [f'{item["text"].strip()}{self.params["stop"]}' for item in response["choices"]]

        return response_texts[0]


    def get_context_from_memory(self, memory: AgentMemory, session_id='default', **kwargs) -> str:
        df = memory.get_text(session_id=session_id)
        df = df[df['text'].str.len() > 0]
        if df.empty:
            return None
        last_turn_df = df[df['turn'] == df['turn'].max()]
        prev_question = last_turn_df[last_turn_df['is_agent'] == True]['text'].values[0]
        prev_answers_list = last_turn_df[last_turn_df['is_agent'] == False]['text'].values
        
        prev_answer = ' '.join(prev_answers_list)
        if len(prev_answer.split()) > 300:
            raise Exception('GPT3 question generator: prev_answer is too long')
        return {'question': prev_question, 'answer': prev_answer}

class ImprovisedQuestionGenerator(TextGenerator):
    def __init__(self,params:dict=None) -> None:
        super().__init__()
        self.params = params
    
    def init_model(self, params:dict):
        self.params = params 
        openai.api_key = os.getenv('OPENAI_API_KEY')

    def generate_text(self, prompt:str)->List[str]:
        api_params = {**self.params, 'prompt':prompt}
        response = openai.Completion.create(**api_params)
        response_texts = [f'{item["text"].strip()}{self.params["stop"]}' for item in response["choices"]]
        return response_texts
