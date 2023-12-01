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

from .util import append_experiment_results_agent_version

from .prompts import promptCombineAndRewordInStyle, promptIsThisAbout, promptGeologicalRegions, promptLocationWithinRegions

from .prompts import promptDecideIfCountySmall, promptCombineAndRewordInStyleB

### functions functions
from .native_skills.macrostrat.macrostrat import getPointLocationStratColumn, macrostratOnlyReturnFirstTwoLayers, macrostratOnlyReturnFirstLayer, ifNoSurfaceGeology
#### macrostrat prompts
from .native_skills.macrostrat.macrostrat import macroStratColSummarizationTop, macroStratColSummarizationSecondMostLayer, macroStratColSummarizationB, macroStratColSummarizationWhenNoColumn

from .native_skills.bing.geocoding import getStateAndCountyFromLatLong, getAddressFromLatLong, getPointLocationFromCityStateAndCounty

from .native_skills.wikipedia.wikipedia import getWikipediaPageAndProcess, extractContentFromWikipediaPageContent

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
    request = """
    What is the geologic story around """+location+""" ? 
    Be sure to discuss the youngest uppermost rocks that range from """+yearsOld+""" million years old. Break it down by time period and < 10 sentences.
    Describe the geologic history in narrative fashion over 6-10 sentences as a professor leading a geology field trip. 
    Assume you have just described the rocks in front of you at the outcrop and are now talking about regional geology.
    """
    print('chatGPT request is',request)
    messages = [
    SystemMessage(content="You are a helpful AI assistant with knowledge about regional geology pretending to be a professor leading a geology 101 field trip."),
    HumanMessage(content=request)
    ]
    completion = chat(messages)
    #completion = openaiNotLC.ChatCompletion.create( model="gpt-4", messages=[{"role": "user", "content": "What is the geologic story around  Estes Park, Colorado USA ? Break it down by time period and keep it under 12 sentences."} ] ) 
    print(completion)
    return completion

def callChatGPT4regionalGeologyWithLocal(inputString:str):
    location, yearsOld , localGeology = inputString.split("|")
    request = """
    What is the geologic story around """+location+""" ? 
    Be sure to discuss the youngest uppermost rocks that range from """+yearsOld+""" million years old. Break it down by time period and < 10 sentences.
    Describe the geologic history in narrative fashion over 6-10 sentences as a professor leading a geology field trip.
    Start by including the following information about the local rock outcrop in front of you: [" """+localGeology+""" "] and continue on from it talking about how that fits into the regional geology.
    """
    print('chatGPT request is',request)
    messages = [
    SystemMessage(content="You are a helpful AI assistant with knowledge about regional geology pretending to be a professor leading a geology 101 field trip."),
    HumanMessage(content=request)
    ]
    completion = chat(messages)
    #completion = openaiNotLC.ChatCompletion.create( model="gpt-4", messages=[{"role": "user", "content": "What is the geologic story around  Estes Park, Colorado USA ? Break it down by time period and keep it under 12 sentences."} ] ) 
    print(completion)
    return completion

def callChatGPT4regionalGeologyWithLocal_B(inputString:str):
    location, localGeology = inputString.split("|")
    request = """
    What is the geologic story around """+location+""" ? 
    Describe the geologic history in narrative fashion over 6-10 sentences as a professor leading a geology field trip.
    Start by including the following information about the local rock outcrop in front of you: [" """+localGeology+""" "] and continue on from it talking about how that fits into the regional geology.
    """
    print('chatGPT request is',request)
    messages = [
    SystemMessage(content="You are a helpful AI assistant with knowledge about regional geology pretending to be a professor leading a geology 101 field trip."),
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

def getLatitudeLongitudeFromAmbiguousLocationStringViaGPT4(inputString):
    request = """
        Given the following string description of a location, return the latitude and longitude that approximates the location
        --- start input location ---
        --- end input location ---
        Provide a latitude and longitude for the given location in format: `latitude: 100.000 longitude -10.000` 
        """
    messages = [
    SystemMessage(content="You are a helpful assistant that always provides a reasonable latitude and longitude for a given location description."),
    HumanMessage(content=request)
    ]
    completion = chat(messages)
    print("longitude and latitude from location via gpt chat 4: ",completion)
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
            response_object = getLatitudeLongitudeFromAmbiguousLocationStringViaGPT4(inputString)
    return response_object

def getMacroStratAPIBasic(latLong:str):
    print("Within getMacroStratAPIBasic function the incoming latLong = ",latLong)
    a, b = latLong.split(",")
    if "S" in a:
        a = "-"+a
    if "W" in b:
        b = "-"+b
    a = a.replace("N","").replace("°","").replace("W","").replace("E","").replace("S","").replace("North","").replace("South","").replace("East","").replace("West","")
    b = b.replace("N","").replace("°","").replace("W","").replace("E","").replace("S","").replace("North","").replace("South","").replace("East","").replace("West","")
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
        chainMacroStrat = LLMChain(llm=llm, prompt=macroStratColSummarizationB)
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
        description="Useful for when you need to answer questions about math. Do not add percentages together without using this!",
        # args_schema=CalculatorInput
    ),
]

# tools.append(
#     Tool(
#         name="Check-if-text-about-subject",
#         func=checkIfTextAbout,
#         description="""
#         Useful for determining if a text is about a particular subject.
#         The input to this tool should be a comma separated list of strings.
#         It should take the form of `"subject","text to check if about subject"`.
#         """,
#     )
# )
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
        Useful for finding the uppermost bedrock geology at a point location. Not useful for finding regional geology.
        The input to this tool needs tos be a comma separated list of numbers of length two. 
        The numbers should represent latitude and longitude. No W or N included only numbers. If latitude is South use '-' instead. If longitude is West use '-' instead.
        For example, `40.7128,-74.0060` would be the input for a location at latitude 40.7128 and longitude -74.0060
        """,
    ),
)
tools.append(
    Tool(
        name="get-point-location-from-city-state-and-country",
        func=getPointLocationFromCityStateAndCountyMod,
        description="""
        Useful for finding the latitude and longitude for a given point location defined by a string of 'city, state, and country'.
        The input to this tool should be a comma separated list of strings 
        It should contain city, state, and country'. 
        For example, `"Houston, Texas, United States of America"`.
        Do not take '-' out of latitude or longitude.
        """
    )
)
# tools.append(
#     Tool(
#         name="get-regional-geology-from-chatGPT4",
#         func=callChatGPT4,
#         description="""
#         The best way to find the regional geology of an area. Should call after finding local geology.
#         The input to this tool should be a | separated list of strings of length 2.
#         The first string should describe the location. It can be a city, state, and country or a latitude and longitude if not near a city. For example, `"Houston, Texas, USA"` or `"Oslo, Norway"` or `"51.36° N, 91.62° W"`.
#         The second string should describe the age of the younger geology in millions of years. For example, "0-0.5" or "10-65"
#         The full input should look like "Houston, Texas, USA | 0-10 ".
#         """
#     )
# )
tools.append(
    Tool(
        name="get-regional-geology-from-chatGPT4-with-local",
        func=callChatGPT4regionalGeologyWithLocal,
        description="""
        The best way to find the regional geology of an area. Should call after finding local geology.
        The input to this tool should be a | separated list of strings of length 3.
        The first string should describe the location. It can be a city, state, and country or a latitude and longitude if not near a city. For example, `"Houston, Texas, USA"` or `"Oslo, Norway"` or `"51.36° N, 91.62° W"`.
        The second string should describe the age of the younger geology in millions of years. For example, "0-0.5" or "10-65"
        The third string should describe the local geology at a point location that was found with the Macrostrat-Geology-For-Location tool
        The full input should look like "Houston, Texas, USA | 0-10 | The location has sandstone of jurassic age".
        """
    )
)



agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


## Try to add memory: https://python.langchain.com/en/latest/modules/memory/examples/agent_with_memory.html



def runAgent(agent_prompt_string,location,point_geology="not known"):
    if location != "default":
        adjusted_agent_prompt_string = agent_prompt_string.replace("_______",location)
        adjusted_agent_prompt_string = adjusted_agent_prompt_string.replace("========",point_geology)
    else:
        adjusted_agent_prompt_string = agent_prompt_string.replace("_______","Houston, Texas, USA")
        adjusted_agent_prompt_string = adjusted_agent_prompt_string.replace("========",point_geology)
    geology_response = agent.run(adjusted_agent_prompt_string)
    return {"adjusted_agent_prompt_string":adjusted_agent_prompt_string, "location":location, "geology_response":geology_response}

def goAgent(agent_prompt_string_1, agent_prompt_string_2, location):
    answerObject1 = runAgent(agent_prompt_string_1,location)
    # append_experiment_results_agent_version(filepath, answerObject1)
    combined_string = location+" | "+answerObject1["geology_response"]
    localPlusRegionalVia4 = callChatGPT4regionalGeologyWithLocal_B(combined_string )
    print("final answer = ",localPlusRegionalVia4.content)
    # print("final answer type = ",type(localPlusRegionalVia4))
    # print("  asdfaf = ",localPlusRegionalVia4.content)
    # location, yearsOld , localGeology = inputString.split("|")
    # answerObject2 = runAgent(agent_prompt_string_2,location,point_geology=answerObject1["geology_response"])
    # answerObject2["geology_response"] = [answerObject1["geology_response"],answerObject2["geology_response"]]
    # #answerObject2["geology_response"] = [answerObject1["geology_response"],answerObject2["geology_response"]]
    answerObject1["geology_response"] = localPlusRegionalVia4.content.replace('\n','')
    # answerObject2["adjusted_agent_prompt_string"] = [answerObject1["adjusted_agent_prompt_string"],answerObject2["adjusted_agent_prompt_string"]]
    # append_experiment_results_agent_version(filepath, answerObject2)
    # return [answerObject1, answerObject2]
    print("answerObject1",answerObject1)
    #append_experiment_results_agent_version(filepath, answerObject1)
    return answerObject1
  
# agent_prompt_string = """
#           Tell me the geology of _______
#           Tell me how the uppermost geology at that specific location fits into regional geology story.
#           Say at least 10 to 18 full sentences.
#           """

# agent_prompt_string = """
#           As a professor leading a field trip, describe how the youngest uppermost geology at the point location of _______ fits into regional geology story.
#           """

# agent_prompt_string = """
#           As a professor leading a field trip, describe how the surface geology at the point location of _______ fits into regional geology story.
#           """
          
agent_prompt_string_1 = """
          Describe the uppermost surface geology at the local point location at center of _______ as if it was a rock outcrop and you are a professor leading a geology field trip. 
          Be sure to mention the rock's age, composition, and thickness. 
          Later you will talk about how this fits into the regional geology story so set up for that but do not talk about regional geology yet!
          Only if lithology is sedimentary should the interpreted depositional environment be mentioned. If some piece of information is not present, simply say 'I do not know'. Do not add percentages! 
          Refer to the geology at this point location defined by a latitude and longitude as if it was a rock outcrop and you are a professor leading a geology field trip.
          """      
          
agent_prompt_string_2 = """
          As a professor leading a field trip, you have just described the uppermost surface geology at a point location by saying:
          --- start outcrop geology of point location ---
          ========
          --- end outcrop geology of point location ---
          You do not need to repeat any information about those rocks already discussed! 
          You will put that local outcrop geology within a larger narrative about the regional geology in the surrounding region around _______ . 
          If the location is not well known and there is not a lot written about it, you may need to first find latitude and longitude of the approximates the center point of the geography.
          Tell a story that flows from what you previously said about the outcrop geology of the point location. Do so as a professor leading a geology field trip.
          Start with the phrase 'Now stepping back to talk about the regional geology around this area'
          """       
          
          
#filepath = "../experiments/results_of_tests/experiment_results_agent_F.json"  

# import sys
# argOne = sys.argv[1]
# if argOne:
#     print("argOne = ",argOne)
# else: 
#     argOne = "Houston, Texas, USA"

def startWithInputOfLatLongString(latLongString):
    ### This function assumes latLongObject is an object like latLongAsString = "latitude = "+latitude+", longitude = "+longitude
    #}
    response = goAgent(agent_prompt_string_1, agent_prompt_string_2, location=latLongString)
    return response
    
    
### this version is used for running local and calling via terminal    
#combinedAnswerArray = goAgent(agent_prompt_string_1, agent_prompt_string_2, location=argOne)

### this version is for used with webpage
# response = startWithInputOfLatLongString(latLongString)