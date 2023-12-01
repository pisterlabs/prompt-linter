import os
import openai
import requests
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.base import Document
from langchain.utilities import ApifyWrapper


def chatgpt_browse_web(url):
    plugins = [ "https://gochitchat.ai/linkreader/openapi.yaml",
                "https://webreader.webpilotai.com/openapi.yaml",
                "https://api.gafo.tech/openapi.yaml"]
    # Example usage
    query = "Summarise this url for me in the form a an article with subheaders atleast 600 words:" + url
    
    # Load the OpenAI Chat model
    llm = ChatOpenAI(temperature=0, max_tokens=700)

    # Load your tools
    tools = load_tools(["requests_all"])
    for plugin in plugins:
        try:
            tool = AIPluginTool.from_plugin_url(plugin)
            tools.append(tool)
        except Exception as e:
            print(f"Failed to load plugin from {plugin}: {e}")

    
    # Run the agent with a task
    # This is a placeholder task and should be replaced with the actual task you want to perform

    
    # Initialize the agent with the tools and model
    agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    query = "Extract text from this URL:" + url
    plugin_url = "https://openai.com/blog/chatgpt-plugins"
    result = agent_chain.run(f"{query} {plugin_url}")

    print("resulrssss",result)
    return result


def scrape_text_from_url(url):
    # apify = ApifyWrapper()
    # loader = apify.call_actor(
    #     actor_id="apify/website-content-crawler",
    #     run_input={"startUrls": [{"url": url}]},
    #     dataset_mapping_function=lambda item: Document(
    #         page_content=item["text"] or "", metadata={"source": item["url"]}
    #     ),
    # )
    response = requests.get('https://www.geeksforgeeks.org/python-programming-language/')
    print(response)
 
# print content of request
    print("content",response.content)
    return response



chatgpt_browse_web("https://arxiv.org/pdf/2306.00008.pdf")
