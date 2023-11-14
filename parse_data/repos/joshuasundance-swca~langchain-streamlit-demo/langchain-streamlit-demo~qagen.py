from typing import List

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from pydantic import BaseModel, Field


class QuestionAnswerPair(BaseModel):
    question: str = Field(..., description="The question that will be answered.")
    answer: str = Field(..., description="The answer to the question that was asked.")

    def to_str(self, idx: int) -> str:
        question_piece = f"{idx}. **Q:** {self.question}"
        whitespace = " " * (len(str(idx)) + 2)
        answer_piece = f"{whitespace}**A:** {self.answer}"
        return f"{question_piece}\n\n{answer_piece}"


class QuestionAnswerPairList(BaseModel):
    QuestionAnswerPairs: List[QuestionAnswerPair]

    def to_str(self) -> str:
        return "\n\n".join(
            [
                qap.to_str(idx)
                for idx, qap in enumerate(self.QuestionAnswerPairs, start=1)
            ],
        )


PYDANTIC_PARSER: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=QuestionAnswerPairList,
)


templ1 = """You are a smart assistant designed to help college professors come up with reading comprehension questions.
Given a piece of text, you must come up with question and answer pairs that can be used to test a student's reading comprehension abilities.
Generate as many question/answer pairs as you can.
When coming up with the question/answer pairs, you must respond in the following format:
{format_instructions}

Do not provide additional commentary and do not wrap your response in Markdown formatting. Return RAW, VALID JSON.
"""
templ2 = """{prompt}
Please create question/answer pairs, in the specified JSON format, for the following text:
----------------
{context}"""
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", templ1),
        ("human", templ2),
    ],
).partial(format_instructions=PYDANTIC_PARSER.get_format_instructions)


def get_rag_qa_gen_chain(
    retriever: BaseRetriever,
    llm: BaseLanguageModel,
    input_key: str = "prompt",
) -> RunnableSequence:
    return (
        {"context": retriever, input_key: RunnablePassthrough()}
        | CHAT_PROMPT
        | llm
        | OutputFixingParser.from_llm(llm=llm, parser=PYDANTIC_PARSER)
        | (lambda parsed_output: parsed_output.to_str())
    )
