from typing import List
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

from ..base_validator import Validator


class FactsValidator(Validator):
    def __init__(
        self,
        openai_key: str,
    ):
        self._model = OpenAI(
            temperature=0,
            openai_api_key=openai_key,
            model_name="text-davinci-003",
        )

        self._template = """
            In the next answer only address the data that was given to answer yes/no.
            Given the following facts:
            {facts}
            assert if the following is factually true:
            {response}
            respond with yes/no

            YOUR RESPONSE:
            """

        self._prompt = PromptTemplate(
            template=self._template, input_variables=["facts", "response"]
        )

    def validate(
        self,
        facts: List[str],
        model_output: str,
    ):
        parsed_facts = ", ".join(facts)

        fact_check_chain = LLMChain(
            prompt=self._prompt,
            llm=self._model,
        )
        entails = fact_check_chain.predict(
            facts=parsed_facts,
            response=model_output,
        )

        entails = entails.lower().strip()

        assert (
            "yes" in entails
        ), "llm validation check validation failed on fact check"  # noqa: E501
