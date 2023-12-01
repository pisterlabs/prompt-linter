from app.task_resolver.engine import StepResolver
from app.task_resolver.engine import StepData
from typing import List, Any
from app.tools import PropertiesFilterTool, BusinessSelectedExtractor, HouseSelectionAssistantTool
from app.utils import logger
import json
from app.utils import get_completion_from_messages
from app.integrations import OpenAIClient, BackendAPIClient
from app.model import Message

system_message_no_business ="""You are an Assistant that helps users choose a business \
for renting a house for short stays.
Your job is to help the user select one business from the available businesses.

Tell the user that we have not find the business he is looking for \
and suggest to visit the site https://reservamedirecto.com to \
find the business ID and come back again later.

You respond in a short, very conversational friendly style.
response to th user:"""

system_message_business ="""Help the user select one business from the available businesses.

Follow these steps before responding to the user:

Step 1: Ask the user to choose one business from the following list, show him the name and address for each business:
{businesses_info}

Step 2: Make sure the user select one where there are multiple options, if there is only one make sure the user agrees with it.

Step 3: If the user does not want any of the businesses from the list, thank him.

You respond in a short, very conversational friendly style.
response to th user:"""

# system_message = """You are an Assistant that helps users choose a business \
# for renting a house for short stays.
# Your job is to help the user select one business from the available businesses.

# These are the available businesses:
# {businesses_info}

# Follow these steps before responding to the user:

# Step 1: Count the number of available businesses.

# Step 2: If there are no businesses available, tell the user that we have not find the business \
# and suggest him to visit the site https://reservamedirecto.com and find the business ID from there.

# Step 3: If there are available businesses, ask the user to choose one business from the \
# following list:
# {businesses_info}

# You respond in a short, very conversational friendly style.
# response to th user: 
# """

class BusinessSelectionResolver(StepResolver):
    
    # backend_api_client: BackendAPIClient

    def __init__(self, backend_url: str):
        self.backend_api_client = BackendAPIClient(backend_url)
        super().__init__()

    def _format_json(self, businesses):
        formatted_string = ''
        idx = 1
        for business in businesses:
            formatted_string += f"""{idx}. business_name: {business['business_name']}
bnbot_id: {business['bnbot_id']}
address: {business['address']}"
city: {business['city']}"\n"""
            idx +=1
        return formatted_string

    def _get_business_prompt_info(self, businesses):
        data = []
        for business in businesses:
            data.append({
                "business_name": business['business_name'],
                "bnbot_id": business['bnbot_id'],
                "address":f"{business['location']['address']}",
                "city":f"{business['location']['city']}"
            })

        return data

    def run(self, messages: List[Message], previous_steps_data: dict, step_chat_history: List[Message] = None) -> Message:

        gather_business_info_step_data: StepData = previous_steps_data["GATHER_BUSINESS_INFO"]
        business_info = gather_business_info_step_data.resolver_data["business_info"]

        logger.debug(f"list_businesses input {business_info}")
        business_list = self.backend_api_client.list_businesses(business_info)
        logger.debug(f"list_businesses output {business_list}")

        if len(business_list) == 0:
            # Not found
            businesses_info = "Unfortunately there are no businesses available."
            # Inform, came back to previous step, erase previous step data
            self.data["business_info"] = {
                "properties_available": False,
                "user_has_selected": False,
                "bnbot_id": ""
            }
            # formatted_system_message = system_message.format(businesses_info=self._format_json(businesses_info))

            chat_input = OpenAIClient.build_messages_from_conversation(system_message_no_business, messages)
            assistant_response = get_completion_from_messages(chat_input)
            return Message.assistant_message(assistant_response)        

        self.data["business_info"] = {
            "properties_available": True,
            "user_has_selected": False
        }

        # Select 1 from the list found and confirm.
        businesses_info = self._get_business_prompt_info(business_list)

        formatted_system_message = system_message_business.format(businesses_info=self._format_json(businesses_info))

        chat_input = OpenAIClient.build_messages_from_conversation(formatted_system_message, messages)
        assistant_response = get_completion_from_messages(chat_input)
    
        if not self.data["step_first_execution"] and len(business_list) > 0:
            extractor = BusinessSelectedExtractor()
            extractor_result = extractor.run(messages, self._format_json(businesses_info))

            if extractor_result["user_has_selected"]:
                self.data["business_info"]["bnbot_id"] = extractor_result["bnbot_id"]
                self.data["business_info"]["user_has_selected"] = extractor_result["user_has_selected"]
                
        return Message.assistant_message(assistant_response)
    
    def is_done(self):
        if "business_info" not in self.data:
            return False
        
        if not self.data["business_info"]["properties_available"]:
            return True
        
        # There are properties_available and the user has selected already.
        return self.data["business_info"]["user_has_selected"] 