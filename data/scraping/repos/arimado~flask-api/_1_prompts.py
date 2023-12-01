import inspect
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator


def get_source_code(function_name):
    # Get the source code of the function
    return inspect.getsource(function_name)


class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function."""

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "function_name" not in v:
            raise ValueError("function_name must be the only input_variable.")
        return v

    def format(self, **kwargs):
        # Get the soure code of the function
        source_code = get_source_code(kwargs["function_name"])

        # Generate the prompt to be sent to the language model
        prompt = f"""
        Given the function name and source code, generate an English language explanation of the function.
        Function Name: {kwargs["function_name"].__name__}
        Source Code:
        {source_code}
        Explanation:
        """
        return prompt

    def _prompy_type(self):
        return "function_explainer"


def get_prompt_for_function(function_name, custom_function):
    """Generate an English language explanation of the function."""
    # Generate a prompt for the function "get_source_code"

    fn_explainer = FunctionExplainerPromptTemplate(
        input_variables=[function_name])

    prompt = fn_explainer.format(function_name=custom_function)
    print(prompt)

    return prompt
