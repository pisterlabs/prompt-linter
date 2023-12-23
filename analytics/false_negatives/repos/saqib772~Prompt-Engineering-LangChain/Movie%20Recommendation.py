from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for movie recommendation
recommendation_template = '''Recommend a movie based on the following preferences:
Genre: {genre}
Mood: {mood}
Rating: {rating}'''

recommendation_prompt = PromptTemplate(
    input_variables=["genre", "mood", "rating"],
    template=recommendation_template
)

# Format the movie recommendation prompt
recommendation_prompt.format(
    genre="Thriller",
    mood="Suspenseful",
    rating="8/10"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Run the movie recommendation chain
recommendation_chain.run({
    "genre": "Thriller",
    "mood": "Suspenseful",
    "rating": "8/10"
})
