from app.task_resolver.engine import StepResolver, StepData
from app.utils import logger
from app.utils import get_completion_from_messages
from app.tools import BusinessSearchDataExtractor
from typing import List, Any
from datetime import datetime, timedelta
from app.integrations import OpenAIClient
from app.model import Message


system_message="""You are an Assistant that gathers information from the user about booking an accommodation. 
Your task is only to help the user find the property for booking, any other tasks must not be handled by you.

Follow these steps before responding to the user:

Step 1: Check if the user provided a business ID.\
NOTE: The ID of the business does not have spaces and always starts with the @ character. i.e: '@casa.de.alejandro'.

Step 2: If the user has not provided the business ID and has not provided the location of the property ask him to provide it. \ 
If the location was provided, make sure that it is one of the available ones. \
If the location is not available, then apologize with the user and tell him that it will be available soon. \
The only available locations are:
1. Mercedes, Uruguay
2. Montevideo, Uruguay

Step 3: If the user has not provided the name of the business, ask him to provide so you can search for it. Explain him that the name \
is something similar to the title of the publication in other platforms like booking or airbnb.

Here is the conversation: 
{chat_history}

You respond in a short, very conversational friendly style.
response to th user: """


class GatherBusinessInfoResolver(StepResolver):
    
    
    def run(self, messages: List[Message], previous_steps_data: dict, step_chat_history: List[Message] = None) -> Message:
        
        business_search_data_extractor = BusinessSearchDataExtractor()
        chat_input = OpenAIClient.build_messages_from_conversation(system_message, messages)
        assistant_response = get_completion_from_messages(chat_input)

        business_info = business_search_data_extractor.run(messages)
        
        self.data["business_info"] = business_info

        return Message.assistant_message(assistant_response)
    
    def is_done(self):
        if "business_info" not in self.data:
            return False
        
        business_info = self.data["business_info"]

        return (("bnbot_id" in business_info and business_info["bnbot_id"] is not None) or 
                (("location" in business_info and business_info["location"] is not None) and 
                 (("business_name" in business_info and business_info["business_name"] is not None) or 
                  ("business_owner" in business_info and business_info["business_owner"]))))