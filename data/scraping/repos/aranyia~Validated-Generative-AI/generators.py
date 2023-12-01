from langchain.chains import SequentialChain, LLMChain
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate


class ValidatedAnswerGenerator:
    chain: SequentialChain

    def __init__(self, llm: LLM,
                 prompt: PromptTemplate,
                 validation_prompt: PromptTemplate):
        answer_llm_chain = LLMChain(prompt=prompt,
                                    llm=llm,
                                    output_key="answer")

        validation_chain = LLMChain(prompt=validation_prompt,
                                    llm=llm,
                                    output_key="validation_result")

        input_variables = prompt.input_variables + validation_prompt.input_variables
        input_variables.remove(answer_llm_chain.output_key)

        output_variables = [answer_llm_chain.output_key, validation_chain.output_key]

        self.chain = SequentialChain(chains=[answer_llm_chain, validation_chain],
                                     input_variables=input_variables,
                                     output_variables=output_variables)

    def get_answer(self, input_values: dict, question: str) -> dict:
        input_values["question"] = question

        results = self.chain(input_values)
        results["is_valid"] = self.is_valid(results)
        return results

    @staticmethod
    def is_valid(results: dict) -> bool:
        return results["validation_result"].strip().startswith("No")


class OrderAnswerGenerator(ValidatedAnswerGenerator):
    validation_criteria: str
    assistant_style: str
    heavy_limit: int

    def __init__(self, llm: LLM,
                 prompt: PromptTemplate,
                 validation_prompt: PromptTemplate,
                 validation_criteria,
                 assistant_style,
                 heavy_limit=5000):
        super().__init__(llm, prompt, validation_prompt)
        self.validation_criteria = validation_criteria
        self.assistant_style = assistant_style
        self.heavy_limit = heavy_limit

    def get_answer(self, order: dict, provider: str, question: str) -> dict:
        input_params = {
            "order": order,
            "provider": provider,
            "heavy_limit": self.heavy_limit,
            "assistant_style": self.assistant_style,
            "validation_criteria": self.validation_criteria
        }
        return super().get_answer(input_params, question)
