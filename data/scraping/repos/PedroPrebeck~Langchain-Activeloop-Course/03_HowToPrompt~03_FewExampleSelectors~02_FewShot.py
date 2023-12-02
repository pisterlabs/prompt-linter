from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)

examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!",
    },
    {"query": "How old are you?", "answer": "Age is just a number, but I'm timeless."},
]

example_template = """
User: {query}
AI: {answer}
"""
example_variables = re.findall(r"\{([^}]+)\}", example_template)
example_prompt = PromptTemplate(
    template=example_template, input_variables=example_variables
)

prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n",
)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
print(chain.run("What's the secret to happiness?"))
