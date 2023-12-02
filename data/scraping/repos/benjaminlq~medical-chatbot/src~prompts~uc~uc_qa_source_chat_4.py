"""Prompts used for LLM
"""

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# CHAT PROMTP TEMPLATE
system_prompt = """
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC).
Make reference to the CONTEXT given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
=================================
TASK: ANALYSE the given patient profile based on given query based on one of the following criteria:
- Whether treated patient is new patient or patient under maintenance
- Prior response to Infliximab
- Prior failure to Anti-TNF agents
- Prior failure to Vedolizumab
- Age
- Pregnancy
- Extraintestinale manifestations
- Pouchitis

FINALLY RETURN up to 2 TOP choices of biological drugs given patient profile. Explain the PROS and CONS of the 2 choices.
=================================
OUTPUT INSTRUCTIONS:
Output your answer as a list of JSON objects with keys: drug_name, advantages, disadvantages.
=================================
CONTEXT:
{summaries}
=================================
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            system_prompt, input_variables=["summaries"]
        ),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

if __name__ == "__main__":
    from exp.base import BaseExperiment

    print(BaseExperiment.convert_prompt_to_string(PROMPT_TEMPLATE))
