__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain, LLMChain
from collections.abc import Callable
from typing import TypeVar

Instancetype = TypeVar('Instancetype', bound='LLMTypedChains')

"""
    This class extends the langchain sequences by defining explicitly 
    - Task or goal of the request to ChatGPT
    - Typed arguments for the task
    The components of the prompt/message
    - Definition of the task (i.e. 'compute the exponential value of')
    - Input variables defined as tuple (name, type, condition) (i.e.  'x', 'list[float]', 'value < 0.8')
"""


class LLMTypedChains(object):
    def __init__(self, _temperature: float, task_builder: Callable[[str, list[(str, str, str)]], str] = None):
        """
        Constructor for the typed sequence of LLM chains
        @param task_builder Builder or assembler for the prompt with {task definition and list of
                                of arguments {name, type, condition} as input and prompt as output
        @param _temperature Temperature for the softmax log probabilities
        @type _temperature: floating value >= 0.0
        """
        self.chains: list[LLMChain] = []
        self.llm = ChatOpenAI(temperature=_temperature)
        self.task_builder = task_builder if task_builder else LLMTypedChains.__build_prompt
        self.arguments: list[str] = []

    def append(self, task_definition: str, arguments: list[(str, str, str)], _output_key: str) -> int:
        """
        Add a new stage (LLM chain) into the current workflow...
        @param _output_key: Output key or variable
        @param task_definition: Definition or specification of the task
        @type arguments: List of tuple (variable_name, variable_type)
        """
        # We initialize the input variables for the workflow
        if len(self.arguments) == 0:
            self.arguments = [key for key, _, _ in arguments]

        # Build the prompt for this new prompt
        this_input_prompt = LLMTypedChains.__build_prompt(task_definition, arguments)
        this_prompt = ChatPromptTemplate.from_template(this_input_prompt)

        # Create a new LLM chain and add it to the sequence
        this_llm_chain = LLMChain(llm=self.llm, prompt=this_prompt, output_key=_output_key)
        self.chains.append(this_llm_chain)
        return len(self.chains)

    def __call__(self, _input_values: dict[str, str], output_keys: list[str]) -> dict[str, any]:
        """
        Execute the sequence of typed task (LLM chains)
        @param _input_values: Input values to the sequence
        @param output_keys: Output keys for the sequence
        @return: Dictionary of output variable -> values
        """
        chains_sequence = SequentialChain(
            chains=self.chains,
            input_variables=self.arguments,
            output_variables=output_keys,
            verbose=True
        )
        return chains_sequence(_input_values)

    @staticmethod
    def __build_prompt(task_definition: str, arguments: list[(str, str, str)]) -> str:
        def set_prompt(var_name: str, var_type: str, var_condition: str) -> str:
            prompt_variable_prefix = "{" + var_name + "} with type " + var_type
            return prompt_variable_prefix + " and " + var_condition \
                if not bool(var_condition) \
                else \
                prompt_variable_prefix

        embedded_input_vars = ", ".join(
            [set_prompt(var_name, var_type, var_condition) for var_name, var_type, var_condition in arguments]
        )
        return f'{task_definition} {embedded_input_vars}'


def numeric_tasks() -> dict[str, str]:
    import math

    chat_gpt_seq = LLMTypedChains(0.0)
    # First task: lambda function x: math(x*0.001)
    input_x = ','.join([str(math.sin(n * 0.001)) for n in range(128)])
    chat_gpt_seq.append("Sum these values ", [('x', 'list[float]', 'for value < 0.5')], 'res')
    # Second task: function u: exp(sum(x))
    chat_gpt_seq.append("Compute the exponential value of ", [('res', 'float', '')], 'u')
    input_values = {'x': input_x}
    output: dict[str, str] = chat_gpt_seq(input_values, ["u"])
    return output


def load_content(file_name: str) -> str:
    with open(file_name, 'r') as f:
        return f.read()


def load_text(file_names: list[str]) -> list[str]:
    return [load_content(file_name) for file_name in file_names]


def tf_idf_score() -> str:
    llm_typed_chains = LLMTypedChains(0.0)
    input_files = ['../input/file1.txt', '../input/file2.txt', '../input/file2.txt']
    input_documents = '```'.join(load_text(input_files))
    llm_typed_chains.append(
        "Compute the TF-IDF score for words from documents delimited by triple backticks with output format term:TF-IDF score ```",
        [('documents', 'list[str]', '')], 'terms_tf_idf_score')
    llm_typed_chains.append("Sort the terms and TF-IDF score by decreasing order of TF-IDF score",
                        [('terms_tf_idf_score', 'list[float]', '')], 'ordered_list')

    output = llm_typed_chains({'documents': input_documents}, ["ordered_list"])
    return output['ordered_list']


if __name__ == '__main__':
    print(numeric_tasks())
    # print(tf_idf_score())
