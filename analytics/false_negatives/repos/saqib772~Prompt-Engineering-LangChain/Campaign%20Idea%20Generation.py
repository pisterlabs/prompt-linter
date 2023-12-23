from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for marketing campaign idea generation
campaign_template = '''Generate a creative marketing campaign idea for the following product:
Product: {product}
Target Audience: {audience}'''

campaign_prompt = PromptTemplate(
    input_variables=["product", "audience"],
    template=campaign_template
)

# Format the marketing campaign idea prompt
campaign_prompt.format(
    product="Smartphones",
    audience="Millennials"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
campaign_chain = LLMChain(llm=llm, prompt=campaign_prompt)

# Run the marketing campaign idea generation chain
campaign_chain.run({
    "product": "Smartphones",
    "audience": "Millennials"
})
