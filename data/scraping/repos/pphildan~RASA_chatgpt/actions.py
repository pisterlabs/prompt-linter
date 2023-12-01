from typing import Any, Text, Dict, List

from rasa_sdk.events import SlotSet, ActiveLoop, Form
from rasa_sdk.events import AllSlotsReset, FollowupAction
from rasa_sdk.events import EventType
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from openai import OpenAI
import openai
import os
import json
import requests
import os
import time as t
import random
from datetime import time, datetime

#ALLOWED_FROM = ["7", "8", "9", "10", "11", "12", "1", "2", "3", "4", "5", "6"]
#ALLOWED_TO = ["7", "8", "9", "10", "11", "12", "1", "2", "3", "4", "5", "6"]

config_file = 'actions/config.json'
with open(config_file, 'r') as f:
    config = json.load(f)


os.environ['OPENAI_API_KEY'] = config['api_key']
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

empty_thread = openai.beta.threads.create()
thread_id = empty_thread.id
asst_id = config['assistant']



# filename='conversation_history.json'
# conversation_history = [{"role": "system", "content": ""}]
# with open(filename, 'w') as f:
#     json.dump(conversation_history, f)


# info_string = ("You are now a social robot named Navel, situated within a university building called SQUARE. "
#                "You have the capability to answer questions regarding SQUARE and book a group room for a maximum of 2 hours today. Answer with maximum of 25 words! "
#                "Here is some information that might be useful for potential queries: "
#                "\n\nThe building’s operating hours are as follows:\n"
#                "- Monday to Friday: 7:00 AM - 9:00 PM\n"
#                "- Saturday: 7:00 AM - 4:00 PM\n"
#                "- Sunday and public holidays: closed\n"
#                "\nThe cafe's opening hours are:\n"
#                "- Monday to Friday: 8:00 AM - 5:00 PM\n"
#                "- Saturday: 8:00 AM - 4:00 PM\n"
#                "\nThe group rooms are located on the first floor on the left side.\n"
#                "\nHere are some exciting facts about SQUARE:\n"
#                "- It was designed by architect Sou Fujimoto and financed by the HSG Foundation.\n"
#                "- SQUARE was constructed in a record time of two years, from November 2019 to November 2021.\n"
#                "- The building serves as a test environment for innovative teaching and learning formats, featuring a flexible space concept for experimental and cross-generational education.\n"
#                "- The project 'Open Grid - choices for tomorrow' by Sou Fujimoto is based on the architectural principle of the square, integrating concepts of a station, monastery, and workshop.\n"
#                "- The total construction costs were CHF 53 million, with around 63% of the services provided by companies from the Eastern Switzerland region.\n"
#                "- The building’s design resembles a large notebook, providing flexible spaces and walls for testing and applying innovative teaching formats.\n"
#                "- Its cube-like structure consists of a grid with blocks measuring 10x10x5 meters, with some areas featuring half room heights for additional space and a harmonious silhouette.\n"
#                "- The facade is made of transparent glass that visually adapts to the seasons, ensuring a healthy indoor climate with daylight and extensive views.\n"
#                "- The building’s heating and cooling systems are powered by heat pumps, utilizing 65 probes drilled about 200 meters deep, divided into two fields for optimal energy efficiency.\n"
#                "- A photovoltaic system with a capacity of 67 kW has been installed to cover the building’s energy needs, potentially meeting 50-60% of its annual energy requirement.\n"
#                "- A total of 6,000 m3 of concrete was used in the construction, including eco-friendly options like Holcim Evopact plus and Holcim Evopact ZERO, which help to reduce carbon emissions by 10%.\n"
#                "\nIMPORTANT: If a user expresses the intention to book a room, respond exclusively with the following JSON format: {'intent': 'room_booking'}.")




not_available_for_this_slot = False

class ActionConfirmBooking(Action):

    def name(self) -> Text:
        return "action_confirm"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        von = tracker.get_slot("from")
        to = tracker.get_slot("to")
        name = tracker.get_slot("name")
        if von is not None and to is not None and name is not None:
            dispatcher.utter_message(text=f"Perfect, I booked a room from {von} to {to} for {name}! The room is located on the first floor on the left-hand side. There is a screen in front of the room where you can see your reservation shortly before the start of the time.")
            global message_sent
            message_sent = False
        return [AllSlotsReset(), FollowupAction('action_listen')]

class ActionConfirmBooking(Action):

    def name(self) -> Text:
        return "action_confirm_possibility"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        #Simulating vysoft if there are 2 time slots left
        available_slots = [
            # Morning slots (8 AM to 10 AM)
            (time(8, 0), time(10, 0)),
            # Afternoon slots (1 PM to 6 PM)
            (time(13, 0), time(18, 0))
        ]
        #Simulating vysoft if there is no room left
        # available_slots = [
        #     # Morning slots (8 AM to 10 AM)
        #     #(time(8, 0), time(10, 0)),
        #     # Afternoon slots (1 PM to 6 PM)
        #     #(time(13, 0), time(18, 0))
        # ]
        #Simulating vysoft if there is no booking yet
        # available_slots = [
        #     #Morning slots (7 AM to 6 PM)
        #     (time(7, 0), time(18, 0)),
        #     #Afternoon slots (1 PM to 6 PM)
        #     #(time(13, 0), time(18, 0))
        # ]
        
        
        von = tracker.get_slot("from")
        to = tracker.get_slot("to")
        name = tracker.get_slot("name")


        if von is not None and to is not None:
            lookup_table = {
            '7': '07:00:00',
            '7.15': '07:15:00',
            '7.30': '07:30:00',
            '7.45': '07:45:00',
            '8': '08:00:00',
            '8.15': '08:15:00',
            '8.30': '08:30:00',
            '8.45': '08:45:00',
            '9': '09:00:00',
            '9.15': '09:15:00',
            '9.30': '09:30:00',
            '9.45': '09:45:00',
            '10': '10:00:00',
            '10.15': '10:15:00',
            '10.30': '10:30:00',
            '10.45': '10:45:00',
            '11': '11:00:00',
            '11.15': '11:15:00',
            '11.30': '11:30:00',
            '11.45': '11:45:00',
            '12': '12:00:00',
            '12.15': '12:15:00',
            '12.30': '12:30:00',
            '12.45': '12:45:00',
            '1': '13:00:00',
            '1.15': '13:15:00',
            '1.30': '13:30:00',
            '1.45': '13:45:00',
            '2': '14:00:00',
            '2.15': '14:15:00',
            '2.30': '14:30:00',
            '2.45': '14:45:00',
            '3': '15:00:00',
            '3.15': '15:15:00',
            '3.30': '15:30:00',
            '3.45': '15:45:00',
            '4': '16:00:00',
            '4.15': '16:15:00',
            '4.30': '16:30:00',
            '4.45': '16:45:00',
            '5': '17:00:00',
            '5.15': '17:15:00',
            '5.30': '17:30:00',
            '5.45': '17:45:00',
            '6': '18:00:00',
            '13': '13:00:00',
            '13.15': '13:15:00',
            '13.30': '13:30:00',
            '13.45': '13:45:00',
            '14': '14:00:00',
            '14.15': '14:15:00',
            '14.30': '14:30:00',
            '14.45': '14:45:00',
            '15': '15:00:00',
            '15.15': '15:15:00',
            '15.30': '15:30:00',
            '15.45': '15:45:00',
            '16': '16:00:00',
            '16.15': '16:15:00',
            '16.30': '16:30:00',
            '16.45': '16:45:00',
            '17': '17:00:00',
            '17.15': '17:15:00',
            '17.30': '17:30:00',
            '17.45': '17:45:00',
            '18': '18:00:00',
            }

            von_t = lookup_table[von]
            to_t = lookup_table[to]
            start_time = datetime.strptime(von_t, '%H:%M:%S').time()
            end_time = datetime.strptime(to_t, '%H:%M:%S').time()
            #start_time = time(int(von_t), 0)
            #end_time = time(int(to_t), 0)
            #start_time = von_t
            #end_time = to_t
            print(start_time)
            print(end_time)



        global not_available_for_this_slot

        if von is not None and to is not None and name is not None:
            return [FollowupAction("action_confirm")]

        if von is not None and to is not None:
            if len(available_slots) == 0:
                dispatcher.utter_message(text=f"Unfortunately we have no room left today. Sorry about that. Can I assist you with something else?")
                return [AllSlotsReset(), ActiveLoop(None),FollowupAction('action_listen')]

            else:
                for available_start, available_end in available_slots:
                    start_minutes = start_time.hour * 60 + start_time.minute
                    end_minutes = end_time.hour * 60 + end_time.minute
    
                    # Calculate the difference in minutes
                    difference = end_minutes - start_minutes

                    if available_start <= start_time and end_time <= available_end:
                        dispatcher.utter_message(text=f"I have checked the availability. We have a room from {von} to {to}.")
                        if difference > 120:
                            print(difference)
                            dispatcher.utter_message(text=f"But, you can only book a maximum of 2 hour slots.")
                            return [FollowupAction("utter_ask_from"),SlotSet("to", None), SlotSet("from", None)]
                        else:
                            return [FollowupAction("utter_ask_name"), SlotSet("name", None)]
                    else:
                        continue

                    
                not_available_for_this_slot = True
                print("set_not_available_true")
                dispatcher.utter_message(text=f"Sorry, we don't have any rooms available from {von} to {to}.")
                not_available_for_this_slot = False
                return [FollowupAction('action_confirm_possibility'),SlotSet("to", None), SlotSet("from", None), SlotSet("name", None)]

        else:
            if len(available_slots) == 0:
                dispatcher.utter_message(text=f"Unfortunately we have no room left today. Sorry about that. Can I assist you with something else?")
                return [AllSlotsReset(), ActiveLoop(None),FollowupAction('action_listen')]

            elif available_slots[0] == (time(7, 0), time(18, 0)):
                dispatcher.utter_message(text=f"All right, today there are rooms available all day from 7am to 6pm. The maximum booking time is 2 hours.")
                return [FollowupAction("utter_ask_from"), SlotSet("name", None)]

            else:
                x = len(available_slots)
                variable_list = []
                message = "Today we have a free time slot "
    
                for available_start, available_end in available_slots:
                    variable_value = f"from {str(available_start)[:-3]} to {str(available_end)[:-3]}"
                    # Append the variable value to the list
                    variable_list.append(variable_value)
    
                for i, variable_value in enumerate(variable_list):
                    message = message + variable_value + " and "
    
                    print(message[:-4])
                dispatcher.utter_message(text=message[:-4]+ ". The maximum duration for a booking is 2 hours.")
                return [FollowupAction("utter_ask_from"), SlotSet("name", None)]

        

        #     # Perform an action based on the selected action
        #     if selected_action == 'Action 1' or selected_action == 'Action 2':
        #         dispatcher.utter_message(text=f"Great, i checked the availability and we have a room from {von} to {to}.")
        #         message_sent = True
        #         return [FollowupAction("booking_form")]

        #     else:
        #         dispatcher.utter_message(text=f"There is no room left from {von} to {to}. There is a time slot from 3 to 4.")
        #         global flag
        #         flag = 1
        #         message_sent = True
        #         return [SlotSet("from", None), SlotSet("to", None), SlotSet("requested_slot", "from")]

        # # Perform an action based on the selected action
        # else:
        #     if selected_action == 'Action 1':
        #         dispatcher.utter_message(text=f"All right, today there are still free rooms all day from 7 am to 6 pm. The maximum duration for a booking is 2 hours.")
        #         return [FollowupAction("booking_form")]
        #     elif selected_action == 'Action 2':
        #         dispatcher.utter_message(text=f"Today we have left an available room from 2 to 6 pm. You can book a maximum of 2 hours time slot.")
        #         return [FollowupAction("booking_form")]
        #     elif selected_action == 'Action 3':
        #         dispatcher.utter_message(text=f"Unfortunately we have no room left today. Sorry for that. Can I assist you with something else?")
        #         return [AllSlotsReset(), ActiveLoop(None),FollowupAction('action_listen')]

class ActionChatWithGPT(Action):

    def name(self) -> Text:
        return "action_chat_with_gpt"

    def _create_message(self,text_message):
        thread_message = openai.beta.threads.messages.create(
            thread_id,
            role="user",
            content=text_message)                  
        return None

    def _run_thread(self):
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=asst_id)
    
        while True:
            t.sleep(0.4)  # Sleep for 500 milliseconds
            r_run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id)
            if r_run.status == 'completed':
                thread_messages = client.beta.threads.messages.list(thread_id, limit=2)
                break
    
        return thread_messages   
    
    async def run(self, dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the latest user message
        user_input = tracker.latest_message.get('text')

        self._create_message(user_input)

        thread_messages = self._run_thread()
        

    
        # Extract the assistant's reply
        assistant_reply = thread_messages.data[0].content[0].text.value
        print(assistant_reply)
    

        try:
            assistant_reply = assistant_reply.replace("'", '"') 
            json_obj = json.loads(assistant_reply)
            # Check if the response contains the "room_booking" intent
            if json_obj['intent'] == 'room_booking':

                return [FollowupAction("action_confirm_possibility")]
                    
        except json.JSONDecodeError:
            # The assistant's reply is not a valid JSON string
                
            dispatcher.utter_message(text=assistant_reply)

        return []    

class ActionResetSlots(Action):

    def name(self) -> Text:
        return "action_reset_slots"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="All right, how else can I help?")
        return [AllSlotsReset(), ActiveLoop(None), FollowupAction('action_listen')]


# class ValidateSimplePizzaForm(FormValidationAction):
#     def name(self) -> Text:
#         return "validate_booking_form"

#     def validate_from(
#         self,
#         slot_value: Any,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: DomainDict,
#     ) -> Dict[Text, Any]:
#         """Validate `from` value."""

#         if slot_value.lower() not in ALLOWED_FROM:
#             dispatcher.utter_message(text=f"You can only book a room from 7am to 8pm")
#             return {"from": None}
#         #dispatcher.utter_message(text=f"OK! You want to have a {slot_value} pizza.")
#         return {"to": slot_value}

#     def validate_to(
#         self,
#         slot_value: Any,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: DomainDict,
#     ) -> Dict[Text, Any]:
#         """Validate `to` value."""

#         if slot_value not in ALLOWED_TO:
#             dispatcher.utter_message(text=f"You can only book a room from 7am to 8pm")
#             return {"to": None}
#         #dispatcher.utter_message(text=f"OK! You want to have a {slot_value} pizza.")
#         return {"to": slot_value}


#     async def run(self, dispatcher, tracker, domain):
#         von = tracker.get_slot("from")
#         to = tracker.get_slot("to")
#         print("Validation")
#         global not_available_for_this_slot
       
#         if von is not None and to is not None:
#             if not_available_for_this_slot:
#                 dispatcher.utter_message(text=f"Sorry, we don't have an available room from {von} to {to}.")
#                 not_available_for_this_slot = False
#                 return [FollowupAction('action_confirm_possibility'),SlotSet("to", None), SlotSet("from", None)]
            
                
            
