from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for training routine recommendation
recommendation_template = '''Recommend a training routine based on the following criteria:
Sport: {sport}
Fitness Level: {fitness_level}
Duration: {duration}'''

recommendation_prompt = PromptTemplate(
    input_variables=["sport", "fitness_level", "duration"],
    template=recommendation_template
)

# Format the training routine recommendation prompt
recommendation_prompt.format(
    sport="Basketball",
    fitness_level="Intermediate",
    duration="1 hour"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Run the training routine recommendation chain
recommendation_chain.run({
    "sport": "Basketball",
    "fitness_level": "Intermediate",
    "duration": "1 hour"
})
