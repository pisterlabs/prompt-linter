""" import openai
from dotenv import load_dotenv
import os
from openai_multi_client import OpenAIMultiOrderedClient

# Load environment variables
load_dotenv('.env')

# Use the environment variables for the API keys if available
openai.api_key = os.getenv('OPENAI_API_KEY')

def stepMaker(phaseList):
    for phase in phaseList:
        api2.request(data={
            "messages": [{
                "role": "user",
                "content": f"Break this phase of the bio research project into three steps. {phase}"
            }]
        }, metadata={'phase': phase})

api2 = OpenAIMultiOrderedClient(endpoint="chats", data_template={"model": "gpt-3.5-turbo"})
phaseList = ["Create the plasmid","Clone the plasmid into E. coli","Check for Expression of the Protein"]
api2.run_request_function(stepMaker, phaseList)


for result in api2:
    phase = result.metadata['phase']
    response = result.response['choices'][0]['message']['content']
    print(f"\n{phase}\n:", response) """