"""Prompts used for LLM
"""

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from custom_parsers import DrugOutput

drug_parser = PydanticOutputParser(pydantic_object=DrugOutput)

prompt_template = """Make reference to the context given to assess the scenario. If you do not know the answer. just say that "I don't know", don't try to make up an answer.
You are a physician assistant giving advice on treatment for moderate to severe ulcerative colitis (UC). Use the information given in the REFERENCE TEXT to answer the question.

=========
REFERENCE TEXT:
{summaries}

=========
QUESTION:
Analyse the given patient profile based on the following criteria:
{question}

Return up to 2 TOP choices of biological drugs given patient profile. Explain the PROS and CONS of the 2 choices.

{format_instructions}
=========
Answer:
"""

PROMPT_TEMPLATE = PromptTemplate(
    template=prompt_template,
    input_variables=["summaries", "question"],
    partial_variables={"format_instructions": drug_parser.get_format_instructions()},
)

if __name__ == "__main__":
    from exp.base import BaseExperiment

    print(BaseExperiment.convert_prompt_to_string(PROMPT_TEMPLATE))
