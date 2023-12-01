from dataclasses import dataclass
from typing import Optional

import langchain
from decouple import config
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser


langchain.debug = True


chatgpt_api = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0, openai_api_key=config("OPENAI_API_KEY")
)


@dataclass
class RCI_log:
    question: str
    initial_answer: Optional[str] = None
    constructive_criticism: Optional[str] = None
    final_answer: Optional[str] = None

    def dict(self):
        return self.__dict__.copy()


def run_rci_chain(question: str) -> RCI_log:
    log = RCI_log(question)

    def combine_system_plus_human_chat_prompt(
        sys_template: str, human_template: str
    ) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(sys_template),
                HumanMessagePromptTemplate.from_template(human_template),
            ]
        )

    initial_chat_prompt = combine_system_plus_human_chat_prompt(
        "You are a helpful assistant that provides people with correct and accurate answers.",
        "{question}",
    )

    critique_chat_prompt = combine_system_plus_human_chat_prompt(
        "You are a helpful assistant that looks at a question and it's given answer. You will find out what is wrong with the answer and give a critique.",
        "Question:\n{question}\nAnswer Given:\n{initial_answer}\nReview the answer and find out what is wrong with it.",
    )

    improvement_chat_prompt = combine_system_plus_human_chat_prompt(
        "You are a helpful assistant that will look at a question, its answer and a critique on the answer. Based on this answer and the critique, you will write a new improved answer.",
        "Question:\n{question}\nAnswer Given:\n{initial_answer}\nConstructive Criticism:\n{constructive_criticism}\nBased on this information, give only the correct answer.\nFinal Answer:",
    )

    initial_chain = initial_chat_prompt | chatgpt_api | StrOutputParser()
    log.initial_answer = initial_chain.invoke(log.dict())

    critique_chain = critique_chat_prompt | chatgpt_api | StrOutputParser()
    log.constructive_criticism = critique_chain.invoke(log.dict())

    improvement_chain = improvement_chat_prompt | chatgpt_api | StrOutputParser()
    log.final_answer = improvement_chain.invoke(log.dict())

    print(
        f"""
        Question:
        {log.question}

        Answer Given:
        {log.initial_answer}

        Constructive Criticism:
        {log.constructive_criticism}

        Final Answer:
        {log.final_answer}
        """
    )
    return log


question = "who was the first man to win 9 consecutive races in formula 1?"
print(run_rci_chain(question))
