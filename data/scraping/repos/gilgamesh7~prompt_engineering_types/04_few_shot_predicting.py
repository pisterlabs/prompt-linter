import os
import openai
import dotenv

from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

# generate Synthetic data
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "sunny", "antonym": "cloudy"}
]

example_formatter_template = """
Word: {word}
Antonym: {antonym}\n
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Add three other examples.",
    input_variables=[],
)

llm = OpenAI()
chain = LLMChain(llm=llm, prompt=few_shot_prompt)

print(f"\n{chain.predict()}\n")