# import
import os
from langchain.llms import OpenAI



# pulls Open AI key from Secrets
os.environ["OPENAI_API_KEY"]





from langchain.chains.question_answering import load_qa_chain

# function that takes in question and gives out response
def callAPI(documents, query):
    # configure()
    chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")

    testing = chain.run(input_documents=documents, question=query)
    return testing