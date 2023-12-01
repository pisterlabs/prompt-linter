"""Prompts used for LLM
"""
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Map
system_question_template = """You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC) using biological drugs.
Use the following portion of a long document to see if any of the text is relevant to treatment of given patient profile using biological drugs.
Return any relevant text verbatim.
______________________
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_question_template),
    HumanMessagePromptTemplate.from_template("Patient Profile: {question}"),
]
CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)

# Reduce

system_combine_template = """You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC) using biological drugs.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Given the following extracted information of a long document, return up to 2 top choices of biological drugs given the patient profile.
Explain the PROS and CONS of the 2 choices with respect to the patient profile.

Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.
______________________
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_combine_template),
    HumanMessagePromptTemplate.from_template("Patient Profile: {question}"),
]
CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(messages)

CHAT_COLLAPSE_PROMPT = CHAT_COMBINE_PROMPT

if __name__ == "__main__":
    from exp.base import BaseExperiment

    print(BaseExperiment.convert_prompt_to_string(CHAT_COMBINE_PROMPT))
