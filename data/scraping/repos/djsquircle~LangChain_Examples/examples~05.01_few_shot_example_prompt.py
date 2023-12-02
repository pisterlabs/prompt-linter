from dotenv import load_dotenv
from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain
from langchain.llms import OpenAI

# Load .env variables
load_dotenv()

# Initialize OpenAI's language model
language_model = OpenAI(model_name="text-davinci-003", temperature=0.7)

# Define examples as a constant
EXAMPLES = [
    {"profession": "detective", "trait": "observant"},
    {"profession": "doctor", "trait": "compassionate"},
    {"profession": "warrior", "trait": "brave"},
]

# Define template for formatting examples
example_formatter_template = """Profession: {profession}
Trait: {trait}
"""

# Define the prompt template based on the formatter
example_prompt = PromptTemplate(
    input_variables=["profession", "trait"],
    template=example_formatter_template,
)

# Define a few shot prompt template, using the examples and the example prompt defined above
few_shot_prompt = FewShotPromptTemplate(
    examples=EXAMPLES,
    example_prompt=example_prompt,
    prefix="Here are some examples of professions and the traits associated with them:\n\n",
    suffix="\n\nNow, given a new profession, identify the trait associated with it:\n\nProfession: {input}\nTrait:",
    input_variables=["input"],
    example_separator="\n",
)

# Use the template to format a new prompt
formatted_prompt = few_shot_prompt.format(input="wizard")

# Create an LLMChain using the formatted prompt and the language model
chain = LLMChain(llm=language_model, prompt=PromptTemplate(template=formatted_prompt, input_variables=[]))

# Run the LLMChain to get the model's response
response = chain.run({})

# Print the results
print("Profession: wizard")
print("Trait:", response)
