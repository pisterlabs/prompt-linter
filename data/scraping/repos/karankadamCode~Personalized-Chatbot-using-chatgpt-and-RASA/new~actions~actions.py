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
# npm i openai

# ->>>
# In your Rasa project's actions.py file, import the OpenAI library and create a function that calls 
# the GPT-3 API using the openai.Completion.create() method. 
# The function should take in the user's input and use it as the prompt parameter in the API call.

# import openai

# # export OPENAI_API_KEY='sk-...'
# openai.api_key = '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'

# # openai.api_type = "azure"
# # openai.api_key = "..."
# # openai.api_base = "https://example-endpoint.openai.azure.com"
# # openai.api_version = "2022-12-01"

# def generate_response(input_text):
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=input_text,
#         max_tokens=1024,
#         n = 1,
#         stop=None,
#         temperature=0.5
#     )

#     return response["choices"][0]["text"]
