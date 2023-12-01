from langchain.prompts import PromptTemplate

class TemplateGenerator:
    def __init__(self):
       pass
    
    def getTemplates(self) -> list:
       
        locprompt =  (PromptTemplate(
            input_variables = ['location'],
            template = 'Give me a description of {location}'
        ), "desc")

        cuisineinfo = (PromptTemplate(
            input_variables = ["location"],
            template = "I am tourist, give me 3 food recomendations that I must try when visiting {location}"
        ), "cuisine")

        accomsinfo = (PromptTemplate(
            input_variables = ["location"],
            template = "I am tourist, give me 3 good accomodations at {location} along with their price per night"
        ), "accoms")

        transportinfo = (PromptTemplate(
            input_variables = ["location"],
            template = "I am tourist, give me 3 best transport options when I am travelling at {location}"
        ), "transportation")

        weatherinfo = (PromptTemplate(
            input_variables = ["location"],
            template = "What is the weather like at {location}, format it nicely."
        ), "weather")

        ecoinfo = (PromptTemplate(
            input_variables = ["location"],
            template = "I am a tourist, give me 2 good ways to travel to {location} in a eco-friendly and sustainable manner."
        ), "ecoinfo")

        budgetinfo = (PromptTemplate(
            input_variables = ["location"],
            template = "I am a tourist, give me 2 good ways to save money when travelling to {location}."
        ), "budget")

        return [locprompt, cuisineinfo, accomsinfo, 
                transportinfo, weatherinfo, 
                ecoinfo, budgetinfo]
    
    def getCheckTemplates(self) -> PromptTemplate:
        validLoc = PromptTemplate(
            input_variables = ["location"],
            template = "place = {location}, If place is fictional or cannot be found, return 0. If the place is not fictional and exists, return 1."
        )
        return validLoc

    def iternaryGen(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables = ["location", "days", "month"],
            template = "Create an Itinerary for a trip to {location} in {month}. Spread the activities across {days} days, since I also want some time to relax. Format it nicely."
        )

    