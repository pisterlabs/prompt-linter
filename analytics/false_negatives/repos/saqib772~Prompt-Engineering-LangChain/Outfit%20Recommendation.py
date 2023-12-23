from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for outfit recommendation
recommendation_template = '''Recommend an outfit based on the following criteria:
Occasion: {occasion}
Style: {style}
Color: {color}'''

recommendation_prompt = PromptTemplate(
    input_variables=["occasion", "style", "color"],
    template=recommendation_template
)

# Format the outfit recommendation prompt
recommendation_prompt.format(
    occasion="Formal event",
    style="Classic",
    color="Black"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Run the outfit recommendation chain
recommendation_chain.run({
    "occasion": "Formal event",
    "style": "Classic",
    "color": "Black"
})
