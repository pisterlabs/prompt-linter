"""
scikit-learn Model Wrapper
--------------------------
"""


import pandas as pd
import numpy as np
from statistics import mean
from .model_wrapper import ModelWrapper
import tiktoken
import guidance

class LLMModelWrapper(ModelWrapper):
    """Loads a LLM model and tokenizer (tokenizer implements
    `transform` and model implements `predict_proba`).

    May need to be extended and modified for different types of
    tokenizers.
    """

    def __init__(self, prompts, labels, model_name='text-ada-001', encoding_name='r50k_base'):
        print('hello')
        self.prompts = prompts
        self.labels = labels
        self.model = model_name
        self.encoding_name = tiktoken.get_encoding(encoding_name)

    def __call__(self, text_input_list, batch_size=None):
        print(text_input_list)
        probs = []
        for text in text_input_list:
            all_resp = {'answer': [], 'logprobs':[]}
            for user_message in self.prompts:
                guidance.llm = guidance.llms.OpenAI(self.model)
                prompt = guidance("""
                {{user_message}}
                "{{input_text}}"
                {{select "answer" logprobs='logprobs' options=labels}}
                """)
                output = prompt(input_text=text, labels=self.labels, user_message=user_message)
                all_resp['answer'].append(output['answer'])
                all_resp['logprobs'].append(output['logprobs'])
            probs.append(np.array([np.exp(mean([k[label] for k in all_resp['logprobs']])) for label in self.labels]))
        print(text_input_list, probs)
        return np.array(probs)
    def get_grad(self, text_input):
        raise NotImplementedError()
    
    def _tokenize(self, inputs):
        """Helper method that for `tokenize`
        Args:
            inputs (list[str]): list of input strings
        Returns:
            tokens (list[list[str]]): List of list of tokens as strings
        """
        return [self.tokenizer.convert_ids_to_tokens(self.tokenizer(x)) for x in inputs]
