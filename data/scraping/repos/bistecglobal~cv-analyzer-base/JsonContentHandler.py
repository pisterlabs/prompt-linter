from langchain.chat_models import ChatOpenAI
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
import json
#The summary of the request body is "
def ProcessJsonData(data):
    analysisData ="";
    try:    
        analysisData = AnalyseJsonData(data);
        analysisData = analysisData.replace('The summary of the request body is "', "");
        analysisData = analysisData.replace('"', "");
    except Exception as e:
        analysisData ="";
    return analysisData;


def AnalyseJsonData(data):    
    spec = JsonSpec(dict_= data, max_value_length=4000)
    toolkit = JsonToolkit(spec=spec)
    agent = create_json_agent(
        llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo"),
        toolkit=toolkit,
        max_iteration=1000,
        verbose=True)
   
    return agent.run("Please analyze this request body of the /compitions endpoint for the post of python developer");