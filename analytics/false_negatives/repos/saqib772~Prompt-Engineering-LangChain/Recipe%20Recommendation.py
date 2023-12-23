from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for recipe recommendation
recommendation_template = '''Recommend a recipe based on the following preferences:
Cuisine: {cuisine}
Dietary Restrictions: {restrictions}
Cooking Time: {cooking_time}'''

recommendation_prompt = PromptTemplate(
    input_variables=["cuisine", "restrictions", "cooking_time"],
    template=recommendation_template
)

# Format the recipe recommendation prompt
recommendation_prompt.format(
    cuisine="Italian",
    restrictions="Vegetarian",
    cooking_time="30 minutes"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Run the recipe recommendation chain
recommendation_chain.run({
    "cuisine": "Italian",
    "restrictions": "Vegetarian",
    "cooking_time": "30 minutes"
})
