from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for equipment troubleshooting
troubleshooting_template = '''Troubleshoot the issue with the following equipment:
Equipment: {equipment}
Problem Description: {description}'''

troubleshooting_prompt = PromptTemplate(
    input_variables=["equipment", "description"],
    template=troubleshooting_template
)

# Format the equipment troubleshooting prompt
troubleshooting_prompt.format(
    equipment="Excavator",
    description="Engine not starting"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
troubleshooting_chain = LLMChain(llm=llm, prompt=troubleshooting_prompt)

# Run the equipment troubleshooting chain
troubleshooting_chain.run({
    "equipment": "Excavator",
    "description": "Engine not starting"
})




