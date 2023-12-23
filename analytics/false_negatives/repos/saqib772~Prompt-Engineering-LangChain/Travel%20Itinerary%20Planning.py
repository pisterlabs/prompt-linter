from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for travel itinerary planning
itinerary_template = '''Create a customized travel itinerary for the following destination:
Destination: {destination}
Duration: {duration}
Interests: {interests}'''

itinerary_prompt = PromptTemplate(
    input_variables=["destination", "duration", "interests"],
    template=itinerary_template
)

# Format the travel itinerary planning prompt
itinerary_prompt.format(
    destination="Paris, France",
    duration="5 days",
    interests="Art, Cuisine, Historical landmarks"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
itinerary_chain = LLMChain(llm=llm, prompt=itinerary_prompt)

# Run the travel itinerary planning chain
itinerary_chain.run({
    "destination": "Paris, France",
    "duration": "5 days",
    "interests": "Art, Cuisine, Historical landmarks"
})
