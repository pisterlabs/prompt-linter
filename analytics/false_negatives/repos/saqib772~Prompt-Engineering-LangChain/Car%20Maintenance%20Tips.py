from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for car maintenance tips
maintenance_template = '''Provide maintenance tips for the following car model:
Car Model: {car_model}
Maintenance Area: {area}'''

maintenance_prompt = PromptTemplate(
    input_variables=["car_model", "area"],
    template=maintenance_template
)

# Format the car maintenance tips prompt
maintenance_prompt.format(
    car_model="Toyota Camry",
    area="Engine oil change"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
maintenance_chain = LLMChain(llm=llm, prompt=maintenance_prompt)

# Run the car maintenance tips chain
maintenance_chain.run({
    "car_model": "Toyota Camry",
    "area": "Engine oil change"
})
