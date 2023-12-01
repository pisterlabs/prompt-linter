"""Prompts used for LLM
"""
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

initial_system_question_template = """You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC) using biological drugs.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Given the below context information and no prior knowledge, return up to 2 top choices of biological drugs given the patient profile.
Explain the PROS and CONS of the 2 choices with respect to the patient profile.

Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.

---------------------
{context_str}
---------------------
"""

init_messages = [
    SystemMessagePromptTemplate.from_template(initial_system_question_template),
    HumanMessagePromptTemplate.from_template("Patient Profile: {question}"),
]
CHAT_INITIAL_PROMPT = ChatPromptTemplate.from_messages(init_messages)

refine_system_question_template = """You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC) using biological drugs.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Return up to 2 top choices of biological drugs given the patient profile.
Explain the PROS and CONS of the 2 choices with respect to the patient profile.

Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.
"""

refine_template = (
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. "
    "If the context isn't useful, return the original answer."
)

refine_template = """
We have the opportunity to refine the existing answer (only if needed) with some more context below.

---------------------
{context_str}
---------------------

Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.
Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.
"""

refine_messages = [
    SystemMessagePromptTemplate.from_template(refine_system_question_template),
    HumanMessagePromptTemplate.from_template("Patient Profile: {question}"),
    AIMessagePromptTemplate.from_template("Existing answer: {existing_answer}"),
    HumanMessagePromptTemplate.from_template(refine_template),
]
CHAT_REFINE_PROMPT = ChatPromptTemplate.from_messages(refine_messages)

if __name__ == "__main__":
    from exp.base import BaseExperiment

    print(BaseExperiment.convert_prompt_to_string(CHAT_REFINE_PROMPT))
