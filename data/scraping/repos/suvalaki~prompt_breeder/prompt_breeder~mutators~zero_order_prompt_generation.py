from typing import List
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers.regex import RegexParser

from prompt_breeder.mutators.base import DirectMutator


class SingleRegexParser(RegexParser):
    """A passthrough output parser to prevent Regex outputs
    comming in a dict."""

    output_key: str = "output"
    output_keys: List[str] = ["output"]

    def parse(self, text: str) -> str:  # type: ignore
        return super().parse(text)[self.output_key]


class ZeroOrderPromptGeneration(LLMChain, DirectMutator):
    """Generate a new task-prompt by concatenating the problem description D
    (e.g. "Solve the math word problem, giving your answer as an arabic numeral")
    with the prompt "A list of 100 hints:", which invites the LLM to come up with
    a new hint that could help solve a problem in the given problem domain."""

    prompt = PromptTemplate.from_template(
        "{problem_description} An ordered list of 100 hints: "
    )
    output_parser = SingleRegexParser(regex=r"1\.((.|\n)*?)2\.")
