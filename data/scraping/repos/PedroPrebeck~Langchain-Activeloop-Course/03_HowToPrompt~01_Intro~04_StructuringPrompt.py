from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": "Finding balance in life and learning to enjoy the small momennts.",
    },
    {
        "query": "How can I become more productive?",
        "answer": "Try prioritizing tasks, setting goals, and maintaining a healthy work-life balance.",
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

prefix = """
The following are excerpts from conversations with an AI life coach. The assistant provides insightful advice to users's questions. Here are some examples:
"""

suffix = """
User: {query}
AI: 
"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n",
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

user_query = "What are some tips for improving communication skills?"

response = chain.run({"query": user_query})

print(f"User Query: {user_query}")
print(f"AI Response: {response}\n")
