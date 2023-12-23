from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for product recommendation
recommendation_template = '''Recommend a product based on the following criteria:
Category: {category}
Price Range: {price_range}
Features: {features}'''

recommendation_prompt = PromptTemplate(
    input_variables=["category", "price_range", "features"],
    template=recommendation_template
)

# Format the product recommendation prompt
recommendation_prompt.format(
    category="Electronics",
    price_range="$500-$1000",
    features="Wireless, Bluetooth"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Run the product recommendation chain
recommendation_chain.run({
    "category": "Electronics",
    "price_range": "$500-$1000",
    "features": "Wireless, Bluetooth"
})
