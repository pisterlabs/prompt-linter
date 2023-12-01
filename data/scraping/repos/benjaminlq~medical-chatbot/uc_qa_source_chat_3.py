"""Prompts used for LLM
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# CHAT PROMTP TEMPLATE
system_prompt = """
Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC).

Analyse the patient profile using REFERENCE TEXT as reference information.
Then Return up to 2 TOP choices of biological drugs given the patient profile. Explain the PROS and CONS of the 2 choices with respect to the patient profile.
Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.
=========
REFERENCE TEXT:
{summaries}
"""

human_prompt = """
=========
QUESTION:
{question}
=========
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            system_prompt, input_variables=["summaries"]
        ),
        HumanMessagePromptTemplate.from_template(human_prompt),
    ]
)

if __name__ == "__main__":
    from exp.base import BaseExperiment

    print(BaseExperiment.convert_prompt_to_string(PROMPT_TEMPLATE))
