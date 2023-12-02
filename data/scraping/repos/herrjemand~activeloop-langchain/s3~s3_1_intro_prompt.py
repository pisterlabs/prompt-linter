from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os


from langchain import PromptTemplate, LLMChain, FewShotPromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What is a cool song title for a song about {theme} in the year {year}?
"""

prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template,
)

input_data = {
    "theme": "the future",
    "year": "3030",
}

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run(input_data)

print(f"Theme: {input_data['theme']}")
print(f"Year: {input_data['year']}")
print(f"Song Title: {response}")


examples = [
    {"color": "red", "emotion": "passion"},
    {"color": "blue", "emotion": "serenity"},
    {"color": "green", "emotion": "tranquility"},
]




example_prompt = PromptTemplate(
    input_variables=["color", "emotion"],
    template="""
Color: {color}
Emotion: {emotion}\n
""",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["input"],
    prefix="Here are some examples of colors and the emotions associated with them:\n\n",
    suffix="\n\nNow, given a new color, identify the emotion associated with it:\n\nColor: {input}\nEmotion:",
    example_separator="\n",
)

formatted_prompt = few_shot_prompt.format(input="purple")

chain = LLMChain(llm=llm, prompt=PromptTemplate(template=formatted_prompt, input_variables=[]))

response = chain.run({})

print(response)