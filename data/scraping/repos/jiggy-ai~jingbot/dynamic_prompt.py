"""
Dynamic Prompt Decorator
Copyright (C) 2023 Jiggy AI

Dynamically adjust dynamic prompt elements to constrain the size of list elements 
such that the total prompt length in tokens is below the model maximum,
leaving room for the specified number of response_tokens.

davinci3 = OpenAI(model_name="text-davinci-003", max_tokens=-1, temperature=.1)

@dynamic_prompt(llm=davinci3, response_tokens=200)
def qa_prompt(context : List[str],  question : str):
    prompt  = "Use the following Context to answer the specified question. "
    prompt += "It is important to answer the question correctly based on the Context. "
    prompt += "If the Context does not contain the required information for answering the question "
    prompt += "then answer 'No answer'.\n"    
    prompt += "Context:\n"
    prompt += "\n".join(context)  
    prompt +=  f"\nQuestion: {question}\n"
    prompt += "Answer: "
    return prompt

"""


from langchain.llms.base import BaseLLM
from functools import wraps

def dynamic_prompt(llm : BaseLLM, response_tokens : int):
    def deco_prompt(f):
        @wraps(f)
        def f_prompt(*args, **kwargs):
            # while the prompt is too long, remove elements the end of the list args
            # until everything fits
            def reduce_prompt(listitem):
                if isinstance(listitem, list):
                    return listitem[:-1]
                return listitem
            while llm.max_tokens_for_prompt(f(*args, **kwargs)) < response_tokens:                
                args = tuple([reduce_prompt(a) for a in list(args)])
                kwargs = {k: reduce_prompt(v) for k, v in kwargs.items()}                
            return f(*args, **kwargs)
        return f_prompt  # true decorator
    return deco_prompt

