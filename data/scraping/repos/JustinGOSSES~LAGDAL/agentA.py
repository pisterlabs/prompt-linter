from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent, tool
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.tools import BaseTool

from langchain.chains import LLMChain
from langchain.llms import OpenAI

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# import openai as openaiNotLC

from util import append_experiment_results_agent_version

from prompts import promptCombineAndRewordInStyle, promptIsThisAbout, promptGeologicalRegions, promptLocationWithinRegions

from prompts import promptDecideIfCountySmall, promptCombineAndRewordInStyleB

### functions functions
from native_skills.macrostrat.macrostrat import getPointLocationStratColumn, macrostratOnlyReturnFirstTwoLayers, macrostratOnlyReturnFirstLayer, ifNoSurfaceGeology
#### macrostrat prompts
from native_skills.macrostrat.macrostrat import macroStratColSummarizationTop, macroStratColSummarizationSecondMostLayer, macroStratColSummarization, macroStratColSummarizationWhenNoColumn

from native_skills.bing.geocoding import getStateAndCountyFromLatLong, getAddressFromLatLong, getPointLocationFromCityStateAndCounty

from native_skills.wikipedia.wikipedia import getWikipediaPageAndProcess, extractContentFromWikipediaPageContent

# llm = ChatOpenAI(temperature=0)
# llm = OpenAI(model_name="text-davinci-003",temperature=0.2, max_tokens=256)
#llm = OpenAI(model_name="text-davinci-003",temperature=0.2) ### works mostly
# llm = OpenAI(model_name="text-davinci-003",temperature=0.2,max_tokens=4096) ### does not work as too short!
#llm = OpenAI(model_name="gpt-3.5-turbo",temperature=0.2) ### can only do chat? not text?
# llm = OpenAI(model_name="gpt-4",temperature=0.2, max_tokens=4096)
llm = OpenAI(model_name="text-davinci-003",temperature=0.0)

llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# llm_4a = OpenAI(model_name="gpt-4",temperature=0.2, max_tokens=4096)

chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0)
chatLong = ChatOpenAI(model_name="gpt-4",temperature=0)

def callChatGPT4(inputString:str):
    location, yearsOld = inputString.split("|")
    request = "What is the geologic story around "+location+" ? Be sure to discuss the youngest uppermost rocks around "+yearsOld+" million years old. Break it down by time period and < 9 sentences."
    print('chatGPT request is',request)
    messages = [
    SystemMessage(content="You are a helpful assistant that summarizes regional geology at the side of the road."),
    HumanMessage(content=request)
    ]
    completion = chat(messages)
    #completion = openaiNotLC.ChatCompletion.create( model="gpt-4", messages=[{"role": "user", "content": "What is the geologic story around  Estes Park, Colorado USA ? Break it down by time period and keep it under 12 sentences."} ] ) 
    print(completion)
    return completion

def callChatGPTSummary(inputString:str):
    location, local_geology, regional_geology = inputString.split("|")
    request = """
        Combine the following information into a summary of how the local point geology fits into regional geology picture for """+location+""" 
        --- start uppermost local point geology ---
        """+local_geology+"""
        --- end local geology --- 
        --- start regional geology ---
        """+regional_geology+"""
        --- end regional geology --- 
        """
    print('callChatGPTSummary request is',request)
    messages = [
    SystemMessage(content="You are a helpful assistant that summarizes regional geology at the side of the road."),
    HumanMessage(content=request)
    ]
    completion = chatLong(messages)
    #completion = openaiNotLC.ChatCompletion.create( model="gpt-4", messages=[{"role": "user", "content": "What is the geologic story around  Estes Park, Colorado USA ? Break it down by time period and keep it under 12 sentences."} ] ) 
    print(completion)
    return completion

def getPointLocationFromCityStateAndCountyMod(inputString:str):
    try: 
        city, state, country = inputString.split(",")
        response_object = getPointLocationFromCityStateAndCounty(city, state, country)
    except: 
        try:
            arrayOfStrings = inputString.split(",")
            city = arrayOfStrings[-2]
            state = city
            country = arrayOfStrings[-1]
            response_object = getPointLocationFromCityStateAndCounty(city, state, country)
        except:
            response_object = getPointLocationFromCityStateAndCounty(city, country)
    return response_object

def getMacroStratAPIBasic(latLong:str):
    a, b = latLong.split(",")
    macrostrat_column_json  = getPointLocationStratColumn(float(a),float(b))
    latitude = float(a)
    longitude = float(b)
    return macrostratGeologyForLocationMod(macrostrat_column_json,latitude,longitude)
                                

def macrostratGeologyForLocationMod(macrostrat_column_json,latitude,longitude):
    # macrostrat_column_json = getPointLocationStratColumn(latitude,longitude)
    if macrostrat_column_json == "No stratigraphic column data available for this location.":
        #print("No stratigraphic column data available for this location of: ",latitude,longitude, " so we will try to get surface geology data.")
        macrostrat_map_json = ifNoSurfaceGeology(latitude,longitude)
        #print("macrostrat_map_json map geologic data is",macrostrat_map_json)
        #### Using prompt for map data when there is no stratigraphic column data
        chainMacroStratWhenNotColum = LLMChain(llm=llm, prompt=macroStratColSummarizationWhenNoColumn)
        response = chainMacroStratWhenNotColum.run(macrostrat_map_json)
        
    else:
        #print("Found a stratigraphic column data available for this location of. ",latitude,longitude)
        macrostrat_column_json = macrostratOnlyReturnFirstTwoLayers(macrostrat_column_json)
        #### Using prompt for stratigraphic column data
        chainMacroStrat = LLMChain(llm=llm, prompt=macroStratColSummarization)
        response = chainMacroStrat.run(macrostrat_column_json)
    return response





def checkIfTextAbout(stringInput:str):
    subject, response = stringInput.split(",")
    objectInput = {"subject":subject,"text":response}
    checkIfTextAbout = LLMChain(llm=llm, prompt=promptIsThisAbout)
    print("The objectInput is",objectInput)
    print("The objectInput type is",type(objectInput))
    checkIfTextAbout = checkIfTextAbout.run(objectInput)
    return checkIfTextAbout


def regionalGeologyOfStateFromWikipedia(inputString:str):
    chainWiki = LLMChain(llm=llm, prompt=extractContentFromWikipediaPageContent)
    state, country, regional_geology_subarea = inputString.split(",")
    stateAndCountry = {"state":state,"country":country}
    #search_term = "Geology of "+stateAndCountry["state"]+" state, "+stateAndCountry["country"]
    search_term = "Geology of "+ stateAndCountry["country"]
    if("United States" in country or "USA" in country):
        search_term = "Geology of "+stateAndCountry["state"]
    else:
        search_term = "Geology of "+stateAndCountry["state"]+ country
    
    print("wikipedia search_term = ",search_term)
    wikipedia_page_object = getWikipediaPageAndProcess(search_term,stateAndCountry)
    page_content = wikipedia_page_object["content"]
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(page_content)
    docs = [Document(page_content=t) for t in texts[:3]]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summarized_wikipedia = chain.run(docs)
    wikipedia_page_title = wikipedia_page_object["title"]
    return {"wikipedia_page_title":wikipedia_page_title,"wikipedia_page_content":summarized_wikipedia,"subject_to_extract":regional_geology_subarea}
    # response = chainWiki.run({"subject_to_extract":regional_geology_subarea,"wikipedia_page_content":summarized_wikipedia})
    # if "No" in checkIfTextAbout.run({"subject":"geology","text":response}) or "No" in checkIfTextAbout.run({"subject":stateAndCountry["country"],"text":response}):
    #     ### deferring to direct prompt
    #     ### decide if we want to use the prompt for the state or the country
    #     sizeOfCountryInKilometers_chain = LLMChain(llm=llm, prompt=promptDecideIfCountySmall)
    #     sizeOfCountryInKilometers = sizeOfCountryInKilometers_chain.run(stateAndCountry["country"])
    #     if int(sizeOfCountryInKilometers) > 500000:
    #         geology_regions_chain = LLMChain(llm=llm, prompt=promptGeologicalRegions)
    #         response = geology_regions_chain.run(stateAndCountry["state"])
    #     else:
    #         response = geology_regions_chain.run(stateAndCountry["country"])
    #     return {"summary":response,"wikipedia_page_title":"regional geology areas prompt"}
        
    # else: 
    #     return {"summary":response,"wikipedia_page_title":wikipedia_page_title}

# You can also define an args_schema to provide more information about inputs
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    query: str = Field(description="should be a math expression")

        
tools = [
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you need to answer questions about math",
        # args_schema=CalculatorInput
    ),
]

tools.append(
    Tool(
        name="Check-if-text-about-subject",
        func=checkIfTextAbout,
        description="""
        Useful for determining if a text is about a particular subject.
        The input to this tool should be a comma separated list of strings.
        It should take the form of `"subject","text to check if about subject"`.
        """,
    )
)
tools.append(
    Tool(
        name="get-state-country-from-lat-long",
        func=getStateAndCountyFromLatLong,
        description="""
        Useful for finding the state and country for a given point location defined by latitude and longitude.
        The input to this tool should be a comma separated list of numbers of length two,representing latitude and longitude. 
        For example, `40.7128,-74.0060` would be the input for a location at latitude 40.7128 and longitude -74.0060
        """,
    )
)
tools.append(
    Tool(
        name="get-street-address-from-lat-long",
        func=getPointLocationFromCityStateAndCountyMod,
        description="""
        Useful for finding the street address include state and country for a given point location defined by latitude and longitude.
        The input to this tool should be a comma separated list of strings representing city, state, and country. 
        For example, "Houston, Texas, USA""
        """
    )
)
# tools.append(
#     Tool(
#         name="find-regional-geology-of-state-using-wikipedia",
#         func=regionalGeologyOfStateFromWikipedia,
#         description="""
#         Useful for finding the regional geology of a geographic area using wikipedia.
#         The input to this tool should be a comma separated list of strings 
#         It should contain state, country, and the string 'regional geologic history'. 
#         For example, `"Texas", "United States of America", "regional geologic history"`.
#         """
#     )
# )
tools.append(
    Tool(
        name="Macrostrat-Geology-For-Location",
        func=getMacroStratAPIBasic,
        description="""
        Useful for finding the uppermost bedrock geology at a given point location in latitude and longitude
        The input to this tool should be a comma separated list of numbers of length two,representing latitude and longitude. 
        For example, `40.7128,-74.0060` would be the input for a location at latitude 40.7128 and longitude -74.0060
        """,
    ),
)
tools.append(
    Tool(
        name="get-point-location-from-city-state-and-country",
        func=getPointLocationFromCityStateAndCountyMod,
        description="""
        Useful for finding the latitude and longitude for a given point location defined by city, state, and country.
        The input to this tool should be a comma separated list of strings 
        It should contain city, state, and country'. 
        For example, `"Houston, Texas, United States of America"`.
        """
    )
)
tools.append(
    Tool(
        name="get-regional-geology-from-chatGPT4",
        func=callChatGPT4,
        description="""
        The best way to find the regional geology of an area. Should call after finding local geology.
        The input to this tool should be a | separated list of strings of length 2.
        The first string should describe the location. It can be a city, state, and country or a latitude and longitude if not near a city. For example, `"Houston, Texas, USA"` or `"Oslo, Norway"` or `"51.36° N, 91.62° W"`.
        The second string should describe the age of the younger geology in millions of years. For example, "0-0.5" or "10-65"
        The full input should look like "Houston, Texas, USA | 0-10 ".
        """
    )
)
# tools.append(
#     Tool(
#         name="get-summary-local-regional-geology-from-chatGPT4",
#         func=callChatGPTSummary,
#         description="""
#         Useful when summarizing local & regional geology into one paragraph. 
#         Should only call after finding local geology & regional geology via other tools!
#         The input to this tool should be a | separated list of strings of length 3.
#         The first string should describe the location. It can be a city, state, and country or a latitude and longitude if not near a city. For example, `"Houston, Texas, USA"` or `"51.36° N, 91.62° W"`.
#         The second string should describe the uppermost bedrock geology at a given point location . The third string describes regional geology from the get-regional-geology-from-chatGPT4 tool.
#         The full input should look like "Houston, Texas, USA | words about uppermost geology | words about regional geology ".
#         """
#     )
# )


agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run("""
#           Get the latitude and longitude of Port Clinton, Ohio and 
#           tell me the geology of that location, 
#           if there is any age gap between the top two layers, 
#           and how that local geology fits into regional geologic story.
#           Do so as if you are talking to students on a geology field trip looking at an outcrop.
#           """)

# agent.run("""
#           Get the latitude and longitude of Port Clinton, Ohio.
#           Is there any volcanic rocks there?
#           """)

# agent.run("""
#           Get the latitude and longitude of Houston, Texas.
#           Is there any sedimentary rocks there?
#           """)

### This one does not work out well!
# agent.run("""
#           Get the latitude and longitude of Olso, Norway and the local geology.
#           Determine the age of any rocks there that are metamorphic.
#           """)

# agent.run("""
#           Get the latitude and longitude of Estes Park, Colorado, United States and 
#           tell me the geology of that location, 
#           and how the local geology fits into regional geologic story of Colarado.
#           """)

# agent.run("""
#           Tell me the geology Estes Park, Colorado, United States
#           and how the local geology fits into regional geologic story
#           """)

# agent.run("""
#           Tell me how the surface geology of Estes Park, Colorado, United States
#           fits into recent regional geologic history of the area.
#           """)

# agent.run("""
#           Tell me how the surface geology of Port Clinton, Ohio, United States
#           fits into geologic history of the area. Do so in a way that is understandable to students on a geology field trip.
#           Tell me at least 15 to 30 sentences.
#           """)

# agent.run("""
#           Tell me the geology of Port Clinton, Ohio, USA
#           Tell me how the uppermost geology at that specific location fits into regional geology story.
#           Say at least 8 to 20 sentences.
#           """)

# agent.run("""
#           Tell me the geology of Houston, Texas, USA
#           Tell me how the uppermost geology at that specific location fits into regional geology story.
#           Say at least 8 to 20 sentences.
#           """)
## Try to add memory: https://python.langchain.com/en/latest/modules/memory/examples/agent_with_memory.html



def runAgent(agent_prompt_string,location):
    if location != "default":
        adjusted_agent_prompt_string = agent_prompt_string.replace("_______",location)
    else:
        adjusted_agent_prompt_string = agent_prompt_string.replace("_______","Houston, Texas, USA")
    geology_response = agent.run(adjusted_agent_prompt_string)
    return {"adjusted_agent_prompt_string":adjusted_agent_prompt_string, "location":location, "geology_response":geology_response}

def goAgent(agent_prompt_string,location):
    answerObject = runAgent(agent_prompt_string,location)
    append_experiment_results_agent_version(filepath, answerObject)
    return answerObject
  
# agent_prompt_string = """
#           Tell me the geology of _______
#           Tell me how the uppermost geology at that specific location fits into regional geology story.
#           Say at least 10 to 18 full sentences.
#           """

# agent_prompt_string = """
#           As a professor leading a field trip, describe how the youngest uppermost geology at the point location of _______ fits into regional geology story.
#           """

agent_prompt_string = """
          As a professor leading a field trip, describe how the surface geology at the point location of _______ fits into regional geology story.
          """
          
filepath = "../experiments/results_of_tests/experiment_results_agent.json"  

import sys

argOne = sys.argv[1]

if argOne:
    print("argOne = ",argOne)
else: 
    argOne = "Houston, Texas, USA"
goAgent(agent_prompt_string,location=argOne)
    