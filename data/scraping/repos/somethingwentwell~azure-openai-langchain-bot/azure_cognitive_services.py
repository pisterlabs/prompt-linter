import os
from dotenv import load_dotenv
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
load_dotenv(os.path.join(BASEDIR, '.env'), override=True)

toolkit = AzureCognitiveServicesToolkit()

def azure_cognitive_services():
    tools = []
    tools.extend(toolkit.get_tools())
    return tools