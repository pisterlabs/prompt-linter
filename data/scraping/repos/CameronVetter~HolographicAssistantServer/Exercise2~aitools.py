from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.tools import GooglePlacesTool
from langchain import LLMMathChain
from langchain.agents import Tool

def get_tools():

    # search = SerpAPIWrapper(search_engine="google")
    # places = GooglePlacesTool()
    llm_davinci = OpenAI(model_name="text-davinci-003", temperature=0, client=None)
    math = LLMMathChain(llm=llm_davinci, verbose=True)
    tools = [
        # Tool(
        #     name = "Current Search",
        #     func = search.run,
        #     description = "useful for when you need to answer questions about current evernts or the current state of the world.  the input to this should be a single search term."
        # ),
        # Tool(
        #     name="Google Places",
        #     func=places.run,
        #     description="useful for when you need to answer questions about places.  the input to this should be a single search term."
        # ),
        Tool(
            name="Math",
            func=math.run,
            description="useful for when you need to answer questions about math.  the input to this should be a single math equation."
        )
    
    ]
    return tools