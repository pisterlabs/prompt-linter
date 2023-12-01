import csv
#import os
#import requests
#import openai
from typing import Dict, Text, Any, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionFetchCSVData(Action):

    def name(self) -> str:
        return "action_fetch_csv_data"

    def run(self, dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Assuming the CSV file is named "data.csv" and is in the root folder
        with open("D:\project98\data\ExportersDirectory_delhi.csv" ,"r") as f:
            reader = csv.reader(f)
            data = list(reader)
        
        # Convert the CSV data into a string format
        response = "\n".join([", ".join(row) for row in data])
        
        dispatcher.utter_message(text=response)
        
        return []