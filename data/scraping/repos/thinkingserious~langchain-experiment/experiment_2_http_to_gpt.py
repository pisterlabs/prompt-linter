from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Define a prompt template that will be used to extract information from the search results
template = """Between >>> and <<< are the raw search result text from google.
Extract the answer to the question '{query}' or say "not found" if the information is not contained.
Use the format
Extracted:<answer or "not found">
>>> {requests_result} <<<
Extracted:"""
PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)

# Create a new LLMRequestsChain object with an LLMChain object as its argument
chain = LLMRequestsChain(llm_chain=LLMChain(
    llm=OpenAI(temperature=0), prompt=PROMPT))

# Define a few questions to ask and their corresponding search URLs
question = "What are the Three (3) best Customer Data Platforms (CDPs)?"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+"),
}

# Use the LLMRequestsChain object to extract the answer from the search results
print(chain(inputs))
# >> {
# >>    "query": "What are the Three (3) best Customer Data Platforms (CDPs)?",
# >>    "url": "https://www.google.com/search?q=What+are+the+Three+(3)+best+Customer+Data+Platforms+(CDPs)?",
# >>    "output": "Segment, Bloomreach, SAP Customer Data Platform"
# >> }

question = "What are the Three (3) best SMS APIs?"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+"),
}
print(chain(inputs))
# >> {
# >>    "query": "What are the Three (3) best SMS APIs?",
# >>    "url": "https://www.google.com/search?q=What+are+the+Three+(3)+best+SMS+APIs?",
# >>    "output": "Twilio, MessageBird B.V., Nexmo, Telnyx"
# >> }

question = "What are the Three (3) best Communications APIs?"
inputs = {
    "query": question,
    "url": "https://www.google.com/search?q=" + question.replace(" ", "+"),
}
print(chain(inputs))
# >> {
# >>    "query": "What are the Three (3) best Communications APIs?",
# >>    "url": "https://www.google.com/search?q=What+are+the+Three+(3)+best+Email+APIs?",
# >>    "output": " not found"
# >> }
