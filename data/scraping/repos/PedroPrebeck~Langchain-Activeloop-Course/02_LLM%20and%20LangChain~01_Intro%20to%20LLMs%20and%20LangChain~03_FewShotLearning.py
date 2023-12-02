from dotenv import load_dotenv

load_dotenv()

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

examples = [
    {
        "querry": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!",
    },
    {"querry": "How old are you?", "answer": "Age is just a number, but I'm timeless."},
]

exemple_template = """
User: {querry}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["querry", "answer"], template=exemple_template
)

prefix = """
The following are excerpts from conversations with an AI assistant. The assistant is know for its humor and wit, providing entertaining and amusing responses to users' questions. Here are some examples:
"""

suffix = """
User: {querry}
AI: 
"""

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["querry"],
    example_separator="\n\n",
)

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
print(chain.run("What's the meaning of life?"))
