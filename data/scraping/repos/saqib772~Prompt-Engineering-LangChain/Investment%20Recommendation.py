from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for investment recommendation
recommendation_template = '''Provide an investment recommendation based on the following information:
Investment Amount: {amount}
Risk Tolerance: {risk_tolerance}
Investment Horizon: {horizon}'''

recommendation_prompt = PromptTemplate(
    input_variables=["amount", "risk_tolerance", "horizon"],
    template=recommendation_template
)

# Format the investment recommendation prompt
recommendation_prompt.format(
    amount="$10,000",
    risk_tolerance="Medium",
    horizon="5 years"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
recommendation_chain = LLMChain(llm=llm, prompt=recommendation_prompt)

# Run the investment recommendation chain
recommendation_chain.run({
    "amount": "$10,000",
    "risk_tolerance": "Medium",
    "horizon": "5 years"
})
