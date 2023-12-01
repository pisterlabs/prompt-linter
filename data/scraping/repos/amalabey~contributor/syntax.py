from abc import ABC, abstractmethod
import os
from langchain.llms.base import BaseLLM
from langchain import PromptTemplate
from core.code_review.models import MethodInfoCollection
from langchain.chains.openai_functions import create_structured_output_chain


class BaseSyntaxProvider(ABC):
    """Base class for syntax providers"""

    @abstractmethod
    def get_method_blocks(self, code_contents: str, lang: str) -> MethodInfoCollection:
        """Returns a list of method/function blocks in the given code file contents"""
        raise NotImplementedError()


class SyntaxProvider(BaseSyntaxProvider):
    """Provides syntax information for a given code file"""

    def __init__(self, llm: BaseLLM, verbose: bool = False) -> None:
        self.llm = llm
        self.verbose = verbose

    """Returns a list of method/function blocks in the given code file contents"""

    def get_method_blocks(self, code_contents: str, lang: str) -> MethodInfoCollection:
        script_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(script_path)
        prompt_abs_path = os.path.join(parent_directory, "prompts", "methods.txt")
        contents_with_line_numbers = self._add_line_numbers(code_contents)

        prompt = PromptTemplate.from_file(prompt_abs_path, ["input", "lang"])
        chain = create_structured_output_chain(
            MethodInfoCollection, llm=self.llm, prompt=prompt, verbose=self.verbose
        )
        result_dict = chain.run({"input": contents_with_line_numbers, "lang": lang})
        return MethodInfoCollection.parse_obj(result_dict)

    def _add_line_numbers(self, original_text: str) -> str:
        lines = original_text.splitlines()
        updated_lines = list()
        for line_number, line in enumerate(lines, start=1):
            line_with_number = f"{line_number} {line}"
            updated_lines.append(line_with_number)
        return "\n".join(updated_lines)
