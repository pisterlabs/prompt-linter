import os
import openai
import dotenv

from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

# First, create the list of few shot examples.
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# Next, we specify the template to format the examples we have provided.
# We use the `PromptTemplate` class for this.
example_formatter_template = """
Word: {word}
Antonym: {antonym}\n
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template
)

# Finally, we create the `FewShotPromptTemplate` object.
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,  # format the examples
    prefix="Give the antonym of every input", # instructions that goes before the examples in the prompt
    suffix="Word: {input}\nAntonym:", # Suffix is where the user input goes
    input_variables=["input"], # ariables that the overall prompt expects
    example_separator="\n\n" # use to join the prefix, examples, and suffix 
)

# instantiate the openai default model - text-davinci-003
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=few_shot_prompt)

print(f"\nAntonym of Dark : {chain.run('Dark')}\n")