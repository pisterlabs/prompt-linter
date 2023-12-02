from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0.9)

examples = [
    {
        "query": "How do I become a better programmer?",
        "answer": "Try talking to a rubber duck; it works wonders.",
    },
    {
        "query": "Why is the sky blue?",
        "answer": "It's nature's way of preventing eye strain.",
    },
]

example_template = """
User: {query}
AI: {answer}
"""
example_variables = re.findall(r"\{([^}]+)\}", example_template)
example_prompt = PromptTemplate(
    template=example_template, input_variables=example_variables
)

prefix = """The following are excerpts from conversations with an AI assistant. The assistant is typically sarcastic and witty, producing creative and funny responses to users' questions. Here are some examples:"""

suffix = """
User: {query}
AI: 
"""

few_shot_prompt_template = FewShotPromptTemplate(
    prefix=prefix,
    suffix=suffix,
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["query"],
    example_separator="\n\n",
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

input_data = {"query": "How can I learn quantum computing?"}
response = chain.run(input_data)

print(f"Input: {input_data['query']}\n")
print(f"Response: {response}\n")
