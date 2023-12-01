from app.task_resolver.engine import StepResolver, StepData
from app.utils import logger
from app.utils import get_completion_from_messages
from app.tools import SearchDataExtractor
from typing import List, Any
from datetime import datetime, timedelta
from app.integrations import OpenAIClient
from app.model import Message

#The only information you need is: check-in date, check-out date and number of guests staying.
system_message =f"""
You are an Assistant that gathers information from the user about booking an accommodation. 
You respond allways in Spanish.
The only information you need is: check-in date, check-out date and number of guests staying.

Follow these Steps before responding to the user new message:

Step 1: Make sure the user provided the check-in date.

Step 2: Make sure the user has provided either the check-out date or the number of nights they are staying.

Step 3: Make sure the user has provided the number of guests that are staying.

You respond in a short, very conversational friendly style.

REMEMBER: Only asked for the information needed, nothing else."""


class GatherBookingInfoResolver(StepResolver):

    def _calculate_checkout_date(self, checkin_date, num_nights):
        checkin_datetime = datetime.strptime(checkin_date, '%Y-%m-%d')
        checkout_datetime = checkin_datetime + timedelta(days=num_nights)
        checkout_date = checkout_datetime.strftime('%Y-%m-%d')
        return checkout_date
    
    
    def run(self, messages: List[Message], previous_steps_data: dict, step_chat_history: List[Message] = None) -> Message:
        
        # exit_task_step_data: StepData = previous_steps_data["EXIT_TASK_STEP"]
        # if exit_task_step_data.resolver_data["conversation_finished"] == True:
        #     logger.debug("Conversation finished. Responding None")
        #     return None

        # chat_history = self.build_chat_history(messages)
        
        search_data_extractor = SearchDataExtractor()
        chat_input = OpenAIClient.build_messages_from_conversation(system_message, messages)
        assistant_response = get_completion_from_messages(chat_input)

        booking_info = search_data_extractor.run(messages)
        
        if booking_info is not None:
            checkin_date = booking_info.get("check_in_date", None)
            checkout_date = booking_info.get("check_out_date", None)
            # num_nights = booking_info["num_nights"]
            num_guests = booking_info.get("num_guests", None)

            if checkin_date is not None and checkout_date is not None and num_guests > 0:
                self.data["booking_information"] = booking_info

        return Message.assistant_message(assistant_response)
    
    def is_done(self):
        if "booking_information" not in self.data:
            return False
        
        booking_information = self.data["booking_information"]

        return (booking_information["check_in_date"] is not None and 
                booking_information["check_out_date"] is not None and 
                booking_information["num_guests"] > 0)