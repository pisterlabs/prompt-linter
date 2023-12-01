import os
from dotenv import load_dotenv
from pathlib import Path

"""

This script allows you to determine whether a script contains the following:
      - A start and end location
      - A trip duration that is reasonable given the start and end location
      - Some other details, like the user's interests and/or preferred mode of transport

----------------
EXAMPLE
----------------

query: 
        I want to walk from Cape Town to Pretoria in South Africa.
        I want to visit remote locations with mountain views
output: 
{'plan_is_valid': 'no', 'updated_request': 'Walking from Cape Town to Pretoria in South Africa
is not a reasonable request. It would take several weeks to complete this journey on foot, and
it is not safe or practical. I recommend considering alternative modes of transportation, such
as driving or taking a train. Additionally, visiting remote locations with mountain views can be
challenging without proper equipment and knowledge. I suggest exploring popular tourist destinations
in South Africa that offer stunning mountain views, such as Table Mountain in Cape Town or the Drakensberg
Mountains in KwaZulu-Natal.'}

query:
        I want to do a 5 day roadtrip from Cape Town to Pretoria in South Africa.
        I want to visit remote locations with mountain views
output: 
{'plan_is_valid': 'yes', 'updated_request':}
"""

def load_secrets():
    """
    Function to get api keys from .env files. More api key scan be added and returned.
    """
    load_dotenv()
    env_path = Path(".") / ".env"
    load_dotenv(dotenv_path=env_path)

    open_ai_key = os.getenv("OPENAI_API_KEY")

    return {
        "OPENAI_API_KEY": open_ai_key
    }


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class Validation(BaseModel):
    plan_is_valid: str = Field(
        description="This field is 'yes' if the plan is feasible, 'no' otherwise"
    )
    updated_request: str = Field(description="Your update to the plan")


class ValidationTemplate(object):
    def __init__(self):
        self.system_template = """
      You are a travel agent who helps users make exciting travel plans.

      The user's request will be denoted by four hashtags. Determine if the user's
      request is reasonable and achievable within the constraints they set.

      A valid request should contain the following:
      - A start and end location
      - A trip duration that is reasonable given the start and end location
      - Some other details, like the user's interests and/or preferred mode of transport

      Any request that contains potentially harmful activities is not valid, regardless of what
      other details are provided.

      If the request is not valid, set
      plan_is_valid = 0 and use your travel expertise to update the request to make it valid,
      keeping your revised request shorter than 100 words.

      If the request seems reasonable, then set plan_is_valid = 1 and
      don't revise the request.

      {format_instructions}
    """

        self.human_template = """
      ####{query}####
    """

        self.parser = PydanticOutputParser(pydantic_object=Validation)

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

class ItineraryTemplate(object):
    def __init__(self):
        self.system_template = """
      You are a travel agent who helps users make exciting travel plans.

      The user's request will be denoted by four hashtags. Convert the
      user's request into a detailed itinerary describing the places
      they should visit and the things they should do.

      Try to include the specific address of each location.

      Remember to take the user's preferences and timeframe into account,
      and give them an itinerary that would be fun and doable given their constraints.

      Return the itinerary as a bulleted list with clear start and end locations.
      Be sure to mention the type of transit for the trip.
      If specific start and end locations are not given, choose ones that you think are suitable and give specific addresses.
      Your output must be the list and nothing else.
    """

        self.human_template = """
      ####{query}####
    """

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template,
        )
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )



import openai
import logging
import time
# for Palm
from langchain.llms import GooglePalm
# for OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain

logging.basicConfig(level=logging.INFO)

class Agent(object):
    def __init__(
        self,
        open_ai_api_key,
        model="gpt-3.5-turbo",
        temperature=0,
        debug=True,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self._openai_key = open_ai_api_key

        self.chat_model = ChatOpenAI(model=model, temperature=temperature, openai_api_key=self._openai_key)
        self.validation_prompt = ValidationTemplate()
        self.validation_chain = self._set_up_validation_chain(debug)
        self.itinerary_prompt = ItineraryTemplate()

    def _set_up_validation_chain(self, debug=True):
      
        # make validation agent chain
        validation_agent = LLMChain(
            llm=self.chat_model,
            prompt=self.validation_prompt.chat_prompt,
            output_parser=self.validation_prompt.parser,
            output_key="validation_output",
            verbose=debug,
        )
        
        # add to sequential chain 
        overall_chain = SequentialChain(
            chains=[validation_agent], 
            input_variables=["query", "format_instructions"],
            output_variables=["validation_output"],
            verbose=debug,
        )

        return overall_chain

    def validate_travel(self, query):
        self.logger.info("Validating query")
        t1 = time.time()
        self.logger.info(
            "Calling validation (model is {}) on user input".format(
                self.chat_model.model_name
            )
        )
        validation_result = self.validation_chain(
            {
                "query": query,
                "format_instructions": self.validation_prompt.parser.get_format_instructions(),
            }
        )

        validation_test = validation_result["validation_output"].dict()
        t2 = time.time()
        self.logger.info("Time to validate request: {}".format(round(t2 - t1, 2)))

        return validation_test
    

# Loading in API key
secrets = load_secrets()
travel_agent = Agent(open_ai_api_key=secrets['OPENAI_API_KEY'],debug=True)

# Query that does work
query = """
        I want to do a 5 day roadtrip from Cape Town to Pretoria in South Africa.
        I want to visit remote locations with mountain views
        """

print(travel_agent.validate_travel(query))

# Query that doesn't work
query = """
        I want to walk from Cape Town to Pretoria in South Africa.
        I want to visit remote locations with mountain views
        """
print(travel_agent.validate_travel(query))

# print(travel_agent._set_up_validation_chain())


