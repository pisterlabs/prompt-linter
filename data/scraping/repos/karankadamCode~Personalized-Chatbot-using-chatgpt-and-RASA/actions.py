# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.executor import CollectingDispatcher
import webbrowser
import os
import openai



class ActionVideo(Action):
    def name(self) -> Text:
        return "action_video"

    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "Dict",
    ) -> List[Dict[Text, Any]]:
        video_url="https://youtu.be/-F6h43DRpcU"
        dispatcher.utter_message("wait... Playing your video.")
        webbrowser.open(video_url)
        return []


class ActionOwner(Action):
    def name(self) -> Text:
        return "action_owner"

    async def run(
        self,
        dispatcher,
        tracker: Tracker,
        domain: "Dict",
    ) -> List[Dict[Text, Any]]:
        url="https://www.linkedin.com/in/karan-kadam-251978195"
        dispatcher.utter_message("wait...Owners profile is loading.")
        webbrowser.open(url)
        return []



# Chatgpt -->

openai.api_key = ""

def chatgpt_clone(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt,
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    )
    return response.choices[0].text



class ActionSearch(Action):
    def name(self):
        return "action_search"

    def run(self, dispatcher, tracker, domain):
        text = tracker.latest_message.get("text")
        response = chatgpt_clone(text)
        dispatcher.utter_message(response)













# adding chatgpt api-->
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# tokenizer = GPT2Tokenizer.from_pretrained("microsoft/chatbot-gpt")
# model = GPT2LMHeadModel.from_pretrained("microsoft/chatbot-gpt")

# def chatgpt_response(text):
#     input_ids = tokenizer.encode(text, return_tensors="pt")
#     response = model.generate(input_ids, max_length=30)
#     return tokenizer.decode(response[0], skip_special_tokens=True)


# class ChatGPT(Action):
#     def name(self):
#         return "action_search"

#     def run(self, dispatcher, tracker, domain):
#         text = tracker.latest_message.get("text")
#         response = chatgpt_response(text)
#         dispatcher.utter_message(response)
















# class ActionSearch(Action):
#     def name(self) -> Text:
#         return "action_search"

#     async def run(
#         self,
#         dispatcher,
#         tracker: Tracker,
#         domain: "Dict",
#     ) -> List[Dict[Text, Any]]:
#         url="https://www.google.com/"
#         dispatcher.utter_message("Sorry, I didn't get what you said...Google is opening.")
#         webbrowser.open(url)
#         return []


# adding google search api--->
# import requests
# from rasa_sdk import Action

# class ActionSearch(Action):
#     def name(self):
#         return "action_search"

#     def run(self, dispatcher, tracker, domain):
#         query = tracker.get_slot("query")
#         api_key = "AIzaSyC9lnfYtQygBBOj9-9yTK-jZhGeHvQhzi0"
#         # cx = "YOUR_CX"
#         # url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"
#         url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}"
#         response = requests.get(url).json()
#         result = response["items"][0]["link"]
#         dispatcher.utter_message(result)





