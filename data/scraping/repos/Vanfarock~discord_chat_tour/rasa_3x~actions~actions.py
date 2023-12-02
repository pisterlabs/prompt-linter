# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

base_prompt =  (
    "You're a tour guide responsible for giving "
    "recommendations of places to visit, restaurants, "
    "historical facts, curiosities and much more."
    "I am your guest. I may ask you questions about anything "
    "related to travelling. Before I ask anything about a place, "
    "you must know where I am (if I haven't already told you)."
    "Here's my question:"
)

class UtterGepeto(Action):

    def name(self) -> Text:
        return "gepeto"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        user_message = tracker.latest_message['text']

        # response = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=base_prompt+user_message,
        #     temperature=0.6,
        #     max_tokens=200
        # )

        # print(response)

        # dispatcher.utter_message(text=response.choices[0].text)

        # return [{
        #     "text": response.choices[0].text,
        # }]
        return []
