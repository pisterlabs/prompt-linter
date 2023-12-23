import os
from apikey import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Text model example

llm = OpenAI(temperature=0.1)

template = """
    Write a title for a Youtube video about {content} with {style} style.
"""

prompt_template = PromptTemplate(
    input_variables=["content", "style"],
    template=template,
)

# Print the template after format
print(prompt_template.format(content="Deep Learning in 1 minutes", style="funny"))

# Save prompt to json
prompt_template.save("video_title.json")

# Define a chain
chain = LLMChain(llm=llm, prompt=prompt_template)

print(chain.run(content="Deep Learning in 1 minutes", style="funny"))

# First, create the list of few shot examples.
examples = [
    {
        "command": "Turn on the kitchen light",
        "action": "turn on",
        "object": "light",
        "location": "kitchen",
        "value": "",
    },
    {
        "command": "Turn off the TV in the living room",
        "action": "turn off",
        "object": "TV",
        "location": "living room",
        "value": "",
    },
    {
        "command": "Increase the bedroom temperature by 2 degrees",
        "action": "Increase temperture",
        "object": "air-conditional",
        "location": "bed room",
        "value": "2 degrees",
    },
]

example_formatter_template = """
    Input command from user: {command}
    The information extracted from above command::\n
    ----
    Action: {action}\n
    Object: {object}\n
    Location: {location}\n
    Value: {value}\n
"""

example_prompt = PromptTemplate(
    input_variables=["command", "action", "object", "location", "value"],
    template=example_formatter_template,
)


few_shot_prompt = FewShotPromptTemplate(
    # These are the examples we want to insert into the prompt.
    examples=examples,
    # This is how we want to format the examples when we insert them into the prompt.
    example_prompt=example_prompt,
    # The prefix is some text that goes before the examples in the prompt.
    # Usually, this consists of intructions.
    prefix="Extract the detail information for an IoT input command. Return the corresponding action, object, location and value. Below are some examples:",
    # The suffix is some text that goes after the examples in the prompt.
    # Usually, this is where the user input will go
    suffix="Input command from user: {command}\nThe information extracted from above command:",
    # The input variables are the variables that the overall prompt expects.
    input_variables=["command"],
    # The example_separator is the string we will use to join the prefix, examples, and suffix together with.
    example_separator="\n\n",
)

print(few_shot_prompt.format(command="Turn off the bath room light"))

chain = LLMChain(llm=llm, prompt=few_shot_prompt)

print(chain.run(command="Turn off the bath room light"))
