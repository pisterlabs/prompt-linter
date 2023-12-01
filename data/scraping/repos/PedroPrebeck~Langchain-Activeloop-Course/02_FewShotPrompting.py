from dotenv import load_dotenv
import re

load_dotenv()

# variable_pattern = r"\{([^}]+)\}"
# input_variables = re.findall(variable_pattern, template)

from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]

example_formatter_template = """
Color: {color}
Emotion: {emotion}\n
"""

input_variables = re.findall(r"\{([^}]+)\}", example_formatter_template)

example_prompt = PromptTemplate(
    input_variables=input_variables, template=example_formatter_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    input_variables=["input"],
    example_separator="\n",
)

color_input = input("Choose a color to learn its associated emotion: ")

template = few_shot_prompt.format(input=color_input)
prompt = PromptTemplate(template=template, input_variables=[])
chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run({})

print(f"Color:{color_input}")
print(f"Emotion:{response}")