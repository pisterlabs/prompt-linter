from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Define the prompt template for customer support ticket resolution
resolution_template = '''Resolve the following customer support ticket:
Ticket ID: {ticket_id}
Issue Description: {issue}'''

resolution_prompt = PromptTemplate(
    input_variables=["ticket_id", "issue"],
    template=resolution_template
)

# Format the customer support ticket resolution prompt
resolution_prompt.format(
    ticket_id="123456",
    issue="Product not delivered"
)

# Initialize the language model and chain
llm = OpenAI(temperature=0.7)
resolution_chain = LLMChain(llm=llm, prompt=resolution_prompt)

# Run the customer support ticket resolution chain
resolution_chain.run({
    "ticket_id": "123456",
    "issue": "Product not delivered"
})
