import inspect

def get_source_code(function_name):
    # Get the source code of the function 
    return inspect.getsource(function_name)

from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator

class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that table in the function name as input, and formats the prompt template to provide the source code of the function."""
    
    @validator('input_variables')
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "function_name" not in v:
            raise ValueError("function_name must be the only input_variable.")
        return v 
    
    def format(self, **kwargs) -> str:
        # Get the source code of the function 
        source_code = get_source_code(kwargs['function_name'])
        
        # Generate the prompt to be send to the language model
        prompt = f"""
        Given the function name and source code, generate an English language explanation of the function.
        Function_name: {kwargs["function_name"].__name__}
        Source code:
        {source_code}
        Explanation:
        """
        return prompt
    
    def _prompt_type(self):
        return "function_explainer"