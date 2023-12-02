from typing import Callable, List

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import Question, Hypothesis
from ai.question_answering.generators.base import HypothesisGenerator


LLM_HYPOTHESIS_GENERATOR_PROMPT_TEMPLATE = """
You are to generate a hypothesis for the following question:
{question}

Your hypothesis should be a statement which can be used to assist with answering the question based on the data available. 
The data available is: 
{data_sources}

Be exploratory and thorough. You need to build a corpus of hypothesis which, when answered, will answer the question.
Your hypothesis should be more detailed that looking for simple corelation. 
Think of hypothesis that make a positive statement about the data.

{format_instructions}
Answer with only one hypothesis. The hypothesis must be answerable using the data provided.
"""


class HypothesisQuery(BaseModel):
    hypothesis: str


OUTPUT_PARSER = PydanticOutputParser(pydantic_object=HypothesisQuery)


LLM_HYPOTHESIS_GENERATOR_PROMPT = PromptTemplate(
    template=LLM_HYPOTHESIS_GENERATOR_PROMPT_TEMPLATE,
    input_variables=["question", "data_sources"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
    output_parser=OUTPUT_PARSER,
)


class LLMHypothesisGenerator(HypothesisGenerator):
    def __init__(
        self,
        comparison_filter: Callable[[Hypothesis], List[Hypothesis]],
        llm: BaseLanguageModel,
    ):
        super().__init__(comparison_filter)
        self.chain = LLMChain(
            llm=llm,
            prompt=LLM_HYPOTHESIS_GENERATOR_PROMPT,
            output_parser=OUTPUT_PARSER,
        )

    def _generate_single(self, question: Question) -> Hypothesis:
        reply = self.chain.predict(
            **{
                "question": question.question,
                "data_sources": "\n".join(
                    [data_source.name for data_source in question.data_sources]
                ),
            }
        )

        return Hypothesis(
            hypothesis=reply.hypothesis,
            data_sources=question.data_sources,
        )
