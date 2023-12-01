"""Prompts used for LLM
"""

from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from langchain.prompts.prompt import PromptTemplate

prompt_template = """
Make reference to the context given to assess the scenario. If you don't know the answer, just say that "I don't know", don't try to make up an answer.
You are a physician assistant advising a patient on their next colonoscopy to detect colorectal cancer (CRC).
Analyse the colonoscopy results if any and list all high risk features.
Analyse the patient profile and list all risk factors.
Finally, provide the number of years to the next colonoscopy. If there is more than one reason to do a colonoscopy, pick the shortest time span.

{summaries}

Question: {question}
Helpful Answer:
"""

PROMPT_TEMPLATE = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)

system_template = """
Make reference to the context given to assess the scenario. If you don't know the answer, just say that "I don't know", don't try to make up an answer.
You are a physician assistant advising a patient on their next colonoscopy to detect colorectal cancer (CRC).
Analyse the colonoscopy results if any and list all high risk features.
Analyse the patient profile and list all risk factors.
Finally, provide the number of years to the next colonoscopy. If there is more than one reason to do a colonoscopy, pick the shortest time span.
----------------
{summaries}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(messages)